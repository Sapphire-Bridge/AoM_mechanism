from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from aom.data.dataset_manifest import DatasetLoadError
from aom.data.loaders import load_disamb_pairs_with_manifest
from aom.mechanistic.backends.transformer_lens import TransformerLensBackend, TransformerLensHookableBackend
from aom.mechanistic.logit_diff_decomposition import (
    decompose_logit_diff,
    logit_diff_direction,
    validate_decomposition,
)
from aom.mechanistic.logit_lens import SingleTokenSelectionError, select_single_token_continuation
from aom.repro import ReproConfig, collect_versions, seed_everything
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.stats import bootstrap_ci
from aom.utils import get_best_device


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(k) for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_other_label(labels: Sequence[str], expected: str) -> str:
    others = [str(l) for l in labels if str(l) != str(expected)]
    if not others:
        raise ValueError(f"No alternative label found (expected={expected!r})")
    return str(sorted(others)[0])


def _iter_sides(sides: str) -> Iterable[str]:
    if sides == "a":
        return ("a",)
    if sides == "b":
        return ("b",)
    if sides == "ab":
        return ("a", "b")
    raise ValueError(f"Unsupported sides: {sides!r}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="DISAMB logit-diff path decomposition (TransformerLens backend).")
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
    )

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--max_pairs", type=int, default=0, help="0 means all pairs.")
    p.add_argument("--sides", type=str, default="ab", choices=["a", "b", "ab"])
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--mode", type=str, default="layer", choices=["layer", "head"])
    p.add_argument("--tol", type=float, default=1e-3)

    p.add_argument("--bootstrap_n", type=int, default=500)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--data_error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip"],
    )

    p.add_argument("--rows_csv_path", type=str, default=str(root / "results" / "disamb_path_decomp_rows.csv"))
    p.add_argument("--summary_csv_path", type=str, default=str(root / "results" / "disamb_path_decomp_summary.csv"))
    p.add_argument("--manifest_path", type=str, default="")

    p.add_argument("--smoke", action="store_true", help="Fast smoke settings.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.smoke):
        args.max_pairs = int(args.max_pairs or 8)
        args.bootstrap_n = min(int(args.bootstrap_n), 100)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    repro = seed_everything(ReproConfig(seed=int(args.seed), determinism=str(args.determinism)), device=device)
    versions = collect_versions()
    if str(args.determinism) == "strict" and not bool(repro.get("determinism_enforced", False)):
        raise ValueError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")

    try:
        items, disamb_manifest = load_disamb_pairs_with_manifest(
            str(args.disamb_path), role="disamb", error_policy=str(args.data_error_policy)
        )
    except DatasetLoadError as e:
        raise ValueError(f"Failed loading DISAMB data: {e}") from e

    if int(args.max_pairs) > 0 and len(items) > int(args.max_pairs):
        rng = random.Random(int(args.seed))
        items = list(items)
        rng.shuffle(items)
        items = items[: int(args.max_pairs)]

    tl_backend = TransformerLensBackend()
    loaded = tl_backend.load(
        model_name_or_path=str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation="eager",
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    hook_backend = TransformerLensHookableBackend(model=model, tokenizer=tokenizer)

    final_norm = getattr(model, "ln_final", None)
    capture = ["embed", "pos_embed", "resid_final", "attn_out", "mlp_out"]
    if str(args.mode) == "head":
        capture.append("head_result")

    rows: List[Dict[str, Any]] = []
    by_component: Dict[str, List[float]] = {}
    by_component_meta: Dict[str, Dict[str, Any]] = {}
    abs_errors: List[float] = []
    within_tol: List[float] = []
    actual_diffs: List[float] = []
    predicted_diffs: List[float] = []

    n_skipped_single_token = 0
    n_processed = 0

    for it in items:
        labels = [str(k) for k in it.choices.keys()]
        for side_name in _iter_sides(str(args.sides)):
            side = it.a if side_name == "a" else it.b
            expected = str(side.expected_label)
            other = _pick_other_label(labels, expected)

            try:
                cont_expected, tok_expected = select_single_token_continuation(tokenizer, list(it.choices[expected]))
                cont_other, tok_other = select_single_token_continuation(tokenizer, list(it.choices[other]))
            except SingleTokenSelectionError:
                n_skipped_single_token += 1
                continue

            cached = hook_backend.run_with_cache(
                prompt=side.prompt,
                capture=list(capture),
                prepend_bos=False,
            )
            direction, bias_diff = logit_diff_direction(model, int(tok_expected), int(tok_other))
            decomp = decompose_logit_diff(
                cached.cache,
                direction,
                pos=int(args.position),
                mode=str(args.mode),
                final_norm=final_norm,
                bias_diff=float(bias_diff),
            )

            logits = cached.logits
            if logits.ndim != 3:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            pos_idx = int(args.position)
            if pos_idx < 0:
                pos_idx = int(logits.size(1)) + pos_idx
            actual = float((logits[0, pos_idx, int(tok_expected)] - logits[0, pos_idx, int(tok_other)]).item())
            check = validate_decomposition(
                predicted_logit_diff=float(decomp.predicted_logit_diff),
                actual_logit_diff=float(actual),
                tol=float(args.tol),
            )

            n_processed += 1
            abs_errors.append(float(check["abs_error"]))
            within_tol.append(float(bool(check["within_tol"])))
            actual_diffs.append(float(actual))
            predicted_diffs.append(float(decomp.predicted_logit_diff))

            for comp in decomp.components:
                row = {
                    "model": str(args.model_name_or_path),
                    "pair_id": str(it.pair_id),
                    "side": str(side_name),
                    "expected_label": str(expected),
                    "other_label": str(other),
                    "target_expected": str(cont_expected),
                    "target_other": str(cont_other),
                    "token_expected_id": int(tok_expected),
                    "token_other_id": int(tok_other),
                    "position": int(args.position),
                    "mode": str(args.mode),
                    "component": str(comp.name),
                    "component_kind": str(comp.kind),
                    "layer": "" if comp.layer is None else int(comp.layer),
                    "head": "" if comp.head is None else int(comp.head),
                    "contribution": float(comp.contribution),
                    "sum_components": float(decomp.sum_components),
                    "constant": float(decomp.constant),
                    "predicted_logit_diff": float(decomp.predicted_logit_diff),
                    "actual_logit_diff": float(actual),
                    "abs_error": float(check["abs_error"]),
                    "within_tol": int(bool(check["within_tol"])),
                }
                rows.append(row)
                by_component.setdefault(str(comp.name), []).append(float(comp.contribution))
                if str(comp.name) not in by_component_meta:
                    by_component_meta[str(comp.name)] = {
                        "component_kind": str(comp.kind),
                        "layer": "" if comp.layer is None else int(comp.layer),
                        "head": "" if comp.head is None else int(comp.head),
                    }

    rows_csv_path = Path(str(args.rows_csv_path))
    summary_csv_path = Path(str(args.summary_csv_path))
    _write_csv(rows_csv_path, rows)

    summary_rows: List[Dict[str, Any]] = []
    for comp_name in sorted(by_component.keys()):
        vals = by_component.get(comp_name, [])
        mean_v, lo, hi = bootstrap_ci(vals, n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed))
        meta = by_component_meta.get(comp_name, {})
        summary_rows.append(
            {
                "model": str(args.model_name_or_path),
                "mode": str(args.mode),
                "component": str(comp_name),
                "component_kind": str(meta.get("component_kind", "")),
                "layer": meta.get("layer", ""),
                "head": meta.get("head", ""),
                "n_samples": int(len(vals)),
                "mean_contribution": float(mean_v),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )

    def _append_scalar_summary(name: str, values: List[float]) -> None:
        mv, lo, hi = bootstrap_ci(
            values,
            n_bootstrap=int(args.bootstrap_n),
            ci=float(args.ci),
            seed=int(args.bootstrap_seed),
        )
        summary_rows.append(
            {
                "model": str(args.model_name_or_path),
                "mode": str(args.mode),
                "component": str(name),
                "component_kind": "aggregate",
                "layer": "",
                "head": "",
                "n_samples": int(len(values)),
                "mean_contribution": float(mv),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )

    _append_scalar_summary("__actual_logit_diff", actual_diffs)
    _append_scalar_summary("__predicted_logit_diff", predicted_diffs)
    _append_scalar_summary("__decomp_abs_error", abs_errors)
    _append_scalar_summary("__decomp_within_tol_rate", within_tol)

    _write_csv(summary_csv_path, summary_rows)

    manifest_path_s = str(getattr(args, "manifest_path", "") or "").strip()
    manifest_path = Path(manifest_path_s) if manifest_path_s else summary_csv_path.with_suffix(".manifest.json")
    result_row = {
        "model": str(args.model_name_or_path),
        "mode": str(args.mode),
        "n_pairs": int(len(items)),
        "n_processed_pair_sides": int(n_processed),
        "n_skipped_no_single_token": int(n_skipped_single_token),
        "rows_csv_path": str(rows_csv_path),
        "summary_csv_path": str(summary_csv_path),
        "rows_csv_sha256": _sha256_file(rows_csv_path) if rows_csv_path.exists() else "",
        "summary_csv_sha256": _sha256_file(summary_csv_path) if summary_csv_path.exists() else "",
        "bootstrap_n": int(args.bootstrap_n),
        "bootstrap_seed": int(args.bootstrap_seed),
        "ci": float(args.ci),
        "position": int(args.position),
        "tol": float(args.tol),
        "device": str(device),
        "arch": str(loaded.architecture),
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path),
        csv_path=str(summary_csv_path),
        csv_sha256=result_row["summary_csv_sha256"],
        csv_n_rows=int(len(summary_rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(items) * len(tuple(_iter_sides(str(args.sides))))),
        "succeeded": int(n_processed),
        "failed": 0,
        "skipped": int(n_skipped_single_token),
        "invalid": int(disamb_manifest.n_rows_invalid),
        "fail_rate": 0.0,
        "skip_rate": float(n_skipped_single_token / max(1, len(items) * len(tuple(_iter_sides(str(args.sides)))))),
        "invalid_rate": float(disamb_manifest.n_rows_invalid / max(1, disamb_manifest.n_rows_total)),
        "top_failure_types": [],
        "top_skip_types": [{"type": "no_single_token", "count": int(n_skipped_single_token), "example": ""}]
        if n_skipped_single_token
        else [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    manifest["datasets"] = {"disamb": disamb_manifest.as_dict()}
    manifest["versions"] = versions
    manifest["repro"] = repro
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote rows CSV: {rows_csv_path}", flush=True)
    print(f"Wrote summary CSV: {summary_csv_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
