from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.dataset_manifest import DatasetLoadError
from aom.data.loaders import load_disamb_pairs_with_manifest
from aom.mechanistic.logit_lens import SingleTokenSelectionError, compute_logit_lens_trace, select_single_token_continuation
from aom.models.loader import load_causal_lm
from aom.repro import ReproConfig, collect_versions, seed_everything
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.run_summary import ErrorPolicy, RunSummary, normalize_error_thresholds
from aom.utils import bootstrap_ci, get_best_device


def _sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2[:180] if len(s2) > 180 else s2


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _stamp_path(path: Path, *, model_tag: str, timestamp: str) -> Path:
    stem = path.stem if path.suffix else path.name
    suffix = path.suffix
    return path.with_name(f"{stem}_{model_tag}_{timestamp}{suffix}")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pick_other_label(labels: Sequence[str], expected: str) -> str:
    others = [str(l) for l in labels if str(l) != str(expected)]
    if not others:
        raise ValueError(f"No alternative label found (expected={expected!r}, labels={list(labels)!r})")
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
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
        help="Determinism mode: strict/best_effort/off (default: best_effort).",
    )
    p.add_argument(
        "--data_error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip"],
        help="How to handle invalid dataset rows (default: warn_skip).",
    )
    p.add_argument(
        "--strict_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any dataset rows are invalid (equivalent to --data_error_policy raise).",
    )

    p.add_argument("--disamb_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--max_pairs", type=int, default=0, help="0 means all pairs.")
    p.add_argument("--sides", type=str, default="ab", choices=["a", "b", "ab"])
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--lens", type=str, default="auto", choices=["auto", "raw", "final_norm"])

    p.add_argument("--bootstrap_n", type=int, default=500)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip", "skip_silent"],
        help="How to handle per-item exceptions (default: warn_skip).",
    )
    p.add_argument(
        "--strict_errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any failures/skips occurred (equivalent to --max_fail_rate 0 --max_skips 0).",
    )
    p.add_argument(
        "--max_fail_rate",
        type=float,
        default=1.0,
        help="Fail if failed/attempted exceeds this (default: 1.0 disables).",
    )
    p.add_argument("--max_skips", type=int, default=-1, help="Fail if skipped exceeds this (default: -1 disables).")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"), help="Directory for manifest output.")
    p.add_argument("--run_name", type=str, default="logit_lens_dataset", help="Base name for manifest output.")
    p.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Optional explicit manifest path (overrides --results_dir/--run_name).",
    )

    p.add_argument(
        "--table_csv_path",
        type=str,
        default=str(ROOT / "results" / "logit_lens_disamb_table.csv"),
        help="Aggregated per-layer table (mean + CI) over all datapoints.",
    )
    p.add_argument(
        "--png_path",
        type=str,
        default=str(ROOT / "results" / "logit_lens_disamb.png"),
        help="Single figure summarizing all datapoints (scatter + mean + CI).",
    )
    p.add_argument(
        "--raw_csv_path",
        type=str,
        default="",
        help="Optional: write the full per-datapoint per-layer trace CSV (can be large).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = _timestamp()
    model_tag = _sanitize_filename(str(args.model_name_or_path))
    error_policy: ErrorPolicy = str(getattr(args, "error_policy", "warn_skip"))  # type: ignore[assignment]
    strict_data = bool(getattr(args, "strict_data", False))
    data_error_policy = "raise" if strict_data else str(getattr(args, "data_error_policy", "warn_skip"))
    summary = RunSummary()
    repro: Dict[str, Any] | None = None
    versions: Dict[str, str] | None = None
    device_backend: str | None = None
    datasets: Dict[str, Any] | None = None
    dataset_warnings: List[str] = []

    raw_csv_written: Optional[Path] = None
    table_csv_written: Optional[Path] = None
    png_written: Optional[Path] = None

    n_skipped_no_single_token = 0
    n_failed_other = 0

    manifest_path_s = str(getattr(args, "manifest_path", "") or "").strip()
    if manifest_path_s:
        manifest_path = Path(manifest_path_s)
    else:
        results_dir = Path(str(getattr(args, "results_dir", ROOT / "results")))
        run_name = str(getattr(args, "run_name", "logit_lens_dataset") or "logit_lens_dataset").strip()
        manifest_path = results_dir / f"{run_name}.manifest.json"

    def _warn(msg: str) -> None:
        if error_policy != "skip_silent":
            print(str(msg), file=sys.stderr, flush=True)

    device_str = ""
    n_pairs = 0
    fatal_error: Exception | None = None
    status = "FAIL"

    try:
        if args.device == "auto":
            device = get_best_device()
        else:
            device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])
        device_str = str(device)
        device_backend = str(device.type)

        determinism = str(getattr(args, "determinism", "best_effort"))
        repro = seed_everything(ReproConfig(seed=int(args.seed), determinism=determinism), device=device)
        versions = collect_versions()
        if determinism == "strict" and not bool(repro.get("determinism_enforced", False)):
            raise ValueError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")

        try:
            disamb_items, disamb_manifest = load_disamb_pairs_with_manifest(
                str(args.disamb_path),
                role="disamb",
                error_policy=data_error_policy,
            )
            datasets = {"disamb": disamb_manifest.as_dict()}
        except DatasetLoadError as e:
            datasets = {"disamb": e.manifest.as_dict()}
            raise
        invalid = int(datasets["disamb"].get("n_rows_invalid", 0) or 0)
        total = int(datasets["disamb"].get("n_rows_total", 0) or 0)
        valid = int(datasets["disamb"].get("n_rows_valid", 0) or 0)
        if total > 0 and valid == 0:
            raise ValueError(f"Dataset disamb has {invalid}/{total} invalid rows and 0 valid rows")
        if invalid > 0:
            msg = f"dataset disamb: invalid_rows {invalid} of {total}"
            dataset_warnings.append(msg)
            _warn(f"[WARN] {msg}")

        if int(args.max_pairs) > 0:
            disamb_items = disamb_items[: int(args.max_pairs)]
        n_pairs = int(len(disamb_items))

        loaded = load_causal_lm(
            args.model_name_or_path,
            device=device,
            torch_dtype=args.torch_dtype,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            attn_implementation=args.attn_implementation,
            device_map=None,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer

        raw_rows: List[Dict[str, Any]] = []
        diffs_by_state: Dict[int, List[float]] = {}
        raw_csv_enabled = bool(str(args.raw_csv_path).strip())

        for it in disamb_items:
            labels = [str(k) for k in it.choices.keys()]
            for side_name in _iter_sides(str(args.sides)):
                side = it.a if side_name == "a" else it.b
                expected = str(side.expected_label)
                try:
                    other = _pick_other_label(labels, expected)

                    cont_expected, tok_expected = select_single_token_continuation(
                        tokenizer, list(it.choices[expected])
                    )
                    cont_other, tok_other = select_single_token_continuation(tokenizer, list(it.choices[other]))

                    trace = compute_logit_lens_trace(
                        model,
                        tokenizer,
                        side.prompt,
                        token_a_id=int(tok_expected),
                        token_b_id=int(tok_other),
                        device=device,
                        position=int(args.position),
                        lens=str(args.lens),
                        compute_logits=True,
                    )

                    local_diffs: List[Tuple[int, float]] = []
                    local_raw_rows: List[Dict[str, Any]] = []
                    for p in trace.points:
                        local_diffs.append((int(p.state_index), float(p.logit_diff)))
                        if raw_csv_enabled:
                            local_raw_rows.append(
                                {
                                    "model": str(args.model_name_or_path),
                                    "arch": str(trace.arch),
                                    "pair_id": str(it.pair_id),
                                    "side": str(side_name),
                                    "expected_label": str(expected),
                                    "other_label": str(other),
                                    "target_expected": str(cont_expected),
                                    "target_other": str(cont_other),
                                    "token_expected_id": int(tok_expected),
                                    "token_other_id": int(tok_other),
                                    "position": int(args.position),
                                    "lens": str(trace.lens),
                                    "apply_final_norm_intermediate": bool(trace.apply_final_norm_intermediate),
                                    "apply_final_norm_last": bool(trace.apply_final_norm_last),
                                    "final_logit_diff": float(trace.final_logit_diff),
                                    "state_index": int(p.state_index),
                                    "block_index": "" if p.block_index is None else int(p.block_index),
                                    "logit_expected": float(p.logit_a),
                                    "logit_other": float(p.logit_b),
                                    "logit_diff": float(p.logit_diff),
                                    "delta_from_prev": float(p.delta_from_prev),
                                }
                            )

                except SingleTokenSelectionError as e:
                    summary.record_skip(e)
                    n_skipped_no_single_token += 1
                    _warn(f"[skip] pair_id={it.pair_id} side={side_name}: {e}")
                    if error_policy == "raise":
                        raise
                except Exception as e:
                    summary.record_failure(e)
                    n_failed_other += 1
                    _warn(f"[fail] pair_id={it.pair_id} side={side_name}: {type(e).__name__}: {e}")
                    if error_policy == "raise":
                        raise
                else:
                    for state_index, v in local_diffs:
                        diffs_by_state.setdefault(int(state_index), []).append(float(v))
                    raw_rows.extend(local_raw_rows)
                    summary.record_success()

        raw_csv_path = str(args.raw_csv_path).strip()
        if raw_csv_path:
            pth = _stamp_path(Path(raw_csv_path), model_tag=model_tag, timestamp=ts)
            _write_csv(raw_rows, pth)
            raw_csv_written = pth
            print(f"Wrote raw trace CSV: {pth}", flush=True)

        table_rows: List[Dict[str, Any]] = []
        if diffs_by_state:
            for state_index in sorted(diffs_by_state.keys()):
                samples = diffs_by_state[state_index]
                mean, lo, hi = bootstrap_ci(
                    samples,
                    n_bootstrap=int(args.bootstrap_n),
                    ci=float(args.ci),
                    seed=int(args.bootstrap_seed),
                )
                frac_pos = float(sum(1 for v in samples if float(v) > 0.0) / max(1, len(samples)))
                table_rows.append(
                    {
                        "model": str(args.model_name_or_path),
                        "state_index": int(state_index),
                        "mean_logit_diff": float(mean),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "frac_positive": frac_pos,
                        "n_samples": int(len(samples)),
                    }
                )

        table_csv_path = str(args.table_csv_path).strip()
        if table_csv_path:
            pth = _stamp_path(Path(table_csv_path), model_tag=model_tag, timestamp=ts)
            _write_csv(table_rows, pth)
            table_csv_written = pth
            print(f"Wrote table CSV: {pth}", flush=True)

        png_path = str(args.png_path).strip()
        if png_path and table_rows and diffs_by_state:
            import matplotlib.pyplot as plt

            xs = [int(r["state_index"]) for r in table_rows]
            ys = [float(r["mean_logit_diff"]) for r in table_rows]
            lo = [float(r["ci_low"]) for r in table_rows]
            hi = [float(r["ci_high"]) for r in table_rows]

            rng = random.Random(int(args.seed))
            x_scatter: List[float] = []
            y_scatter: List[float] = []
            for state_index, samples in diffs_by_state.items():
                for v in samples:
                    x_scatter.append(float(state_index) + rng.uniform(-0.18, 0.18))
                    y_scatter.append(float(v))

            pth = _stamp_path(Path(png_path), model_tag=model_tag, timestamp=ts)
            pth.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.scatter(x_scatter, y_scatter, s=8, alpha=0.08, linewidths=0, label="datapoints")
            plt.plot(xs, ys, marker="o", linewidth=2, label="mean")
            plt.fill_between(xs, lo, hi, alpha=0.2, label=f"{float(args.ci):.2f} CI")
            plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
            plt.xlabel("Hidden-state index (0 = embeddings)")
            plt.ylabel("Logit diff (expected - other)")
            plt.title(
                f"Logit lens (DISAMB) | model={args.model_name_or_path} | sides={args.sides} | n={summary.succeeded}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(pth))
            png_written = pth
            print(f"Wrote PNG: {pth}", flush=True)

    except Exception as e:
        fatal_error = e

    n_sides = int(len(tuple(_iter_sides(str(args.sides)))))
    attempted_expected = int(n_pairs * n_sides)

    max_skips_raw = int(getattr(args, "max_skips", -1))
    strict_errors = bool(getattr(args, "strict_errors", False))
    max_fail_rate, max_skips = normalize_error_thresholds(
        max_fail_rate=float(getattr(args, "max_fail_rate", 1.0)),
        max_skips=max_skips_raw,
        strict_errors=strict_errors,
    )
    max_skips_manifest = int(max_skips_raw if max_skips is None else max_skips)
    if fatal_error is not None:
        status = "FAIL"
        status_reasons = [f"fatal: {type(fatal_error).__name__}"]
    else:
        status, status_reasons = summary.evaluate(max_fail_rate=max_fail_rate, max_skips=max_skips)
        if dataset_warnings:
            if status == "PASS":
                status = "WARN"
            for msg in dataset_warnings:
                if msg not in status_reasons:
                    status_reasons.append(str(msg))

    if fatal_error is None and int(summary.attempted) != int(attempted_expected):
        msg = f"attempted {int(summary.attempted)} != attempted_expected {int(attempted_expected)}"
        if status == "PASS":
            status = "WARN"
        if msg not in status_reasons:
            status_reasons.append(msg)

    results_row: Dict[str, Any] = {
        "model": str(args.model_name_or_path),
        "seed": int(args.seed),
        "device": str(device_str),
        "torch_dtype": str(getattr(args, "torch_dtype", "") or ""),
        "attn_implementation": str(getattr(args, "attn_implementation", "")),
        "disamb_path": str(args.disamb_path),
        "max_pairs": int(args.max_pairs),
        "pairs_observed": int(n_pairs),
        "sides": str(args.sides),
        "position": int(args.position),
        "lens": str(args.lens),
        "error_policy": str(error_policy),
        "strict_errors": bool(strict_errors),
        "max_fail_rate": float(max_fail_rate),
        "max_skips": int(max_skips_manifest),
        "raw_csv_path": "" if raw_csv_written is None else str(raw_csv_written),
        "table_csv_path": "" if table_csv_written is None else str(table_csv_written),
        "png_path": "" if png_written is None else str(png_written),
        "skipped_no_single_token": int(n_skipped_no_single_token),
        "failed_other": int(n_failed_other),
    }
    if fatal_error is not None:
        results_row["fatal_error_type"] = str(type(fatal_error).__name__)
        results_row["fatal_error"] = str(fatal_error)

    manifest = build_run_manifest(argv=sys.argv, results_row=results_row, dataset_manifest_path=None)
    manifest["error_policy"] = str(error_policy)
    manifest["strict_errors"] = bool(strict_errors)
    manifest["max_fail_rate"] = float(max_fail_rate)
    manifest["max_skips"] = int(max_skips_manifest)
    manifest["data_error_policy"] = str(data_error_policy)
    manifest["strict_data"] = bool(strict_data)
    manifest["attempt_unit"] = "pair_side"
    manifest["attempted_expected"] = int(attempted_expected)
    manifest["n_pairs"] = int(n_pairs)
    manifest["n_sides"] = int(n_sides)
    if repro is not None:
        manifest["repro"] = dict(repro)
    if versions is not None:
        manifest["versions"] = dict(versions)
    if device_backend is not None:
        manifest["device_backend"] = str(device_backend)
    if datasets is not None:
        manifest["datasets"] = dict(datasets)
    manifest["run_status"] = str(status)
    manifest["run_status_reasons"] = list(status_reasons)
    manifest["run_summary"] = summary.as_dict()
    write_run_manifest(manifest_path, manifest)

    print(
        "Done "
        f"pairs={n_pairs} sides={args.sides} "
        f"attempted={summary.attempted} ok={summary.succeeded} failed={summary.failed} skipped={summary.skipped} "
        f"status={status}",
        flush=True,
    )
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
