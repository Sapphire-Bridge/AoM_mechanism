from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from aom.data.dataset_manifest import DatasetLoadError
from aom.data.loaders import (
    attach_evidence_metadata_cf,
    attach_evidence_metadata_coh,
    attach_evidence_metadata_disamb,
    load_coherence_items_with_manifest,
    load_counterfactual_pairs_with_manifest,
    load_disamb_pairs_with_manifest,
    load_metadata_sidecar,
)
from aom.mechanistic.backends.transformer_lens import TransformerLensBackend, TransformerLensHookableBackend
from aom.mechanistic.logit_lens import SingleTokenSelectionError, select_single_token_continuation
from aom.mechanistic.why_fetch import WhyFetchExample, aggregate_top_heads, compute_qk_ov_effects
from aom.repro import ReproConfig, collect_versions, seed_everything
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.token_diff import divergence_span
from aom.token_spans import token_span_for_substring
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


def _parse_layers(raw: str) -> Optional[List[int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        p = str(part).strip()
        if not p:
            continue
        out.append(int(p))
    return out if out else None


def _extract_query_and_span(
    metadata: object,
    *,
    fallback_span: Tuple[int, int],
    fallback_query: int = -1,
) -> Tuple[int, Tuple[int, int]]:
    q = int(fallback_query)
    span = (int(fallback_span[0]), int(fallback_span[1]))
    if isinstance(metadata, dict):
        qv = metadata.get("query_pos", None)
        if isinstance(qv, int):
            q = int(qv)
        spans = metadata.get("evidence_spans", None)
        if isinstance(spans, list) and spans:
            sp0 = spans[0]
            if isinstance(sp0, dict):
                a, b = sp0.get("start", None), sp0.get("end", None)
                if isinstance(a, int) and isinstance(b, int) and b > a >= 0:
                    span = (int(a), int(b))
            elif isinstance(sp0, (list, tuple)) and len(sp0) == 2:
                a, b = sp0[0], sp0[1]
                if isinstance(a, int) and isinstance(b, int) and b > a >= 0:
                    span = (int(a), int(b))
    return q, span


def _iter_disamb_examples(
    *,
    items: Sequence[Any],
    tokenizer: Any,
    max_pairs: int,
    seed: int,
) -> List[WhyFetchExample]:
    rows = list(items)
    if int(max_pairs) > 0 and len(rows) > int(max_pairs):
        rng = random.Random(int(seed))
        rng.shuffle(rows)
        rows = rows[: int(max_pairs)]

    out: List[WhyFetchExample] = []
    for it in rows:
        labels = [str(k) for k in it.choices.keys()]
        for side_name, (clean_side, corrupt_side) in (
            ("a", (it.a, it.b)),
            ("b", (it.b, it.a)),
        ):
            expected = str(clean_side.expected_label)
            other = _pick_other_label(labels, expected)
            try:
                _cont_expected, tok_expected = select_single_token_continuation(tokenizer, list(it.choices[expected]))
                _cont_other, tok_other = select_single_token_continuation(tokenizer, list(it.choices[other]))
            except SingleTokenSelectionError:
                continue

            try:
                corr_span, _ = token_span_for_substring(tokenizer, corrupt_side.prompt, it.target, it.target_occurrence)
            except Exception:
                continue
            if not corr_span:
                continue
            q, ev = _extract_query_and_span(
                getattr(it, "metadata", None),
                fallback_span=(int(min(corr_span)), int(max(corr_span)) + 1),
                fallback_query=-1,
            )
            out.append(
                WhyFetchExample(
                    task="disamb",
                    item_id=f"{it.pair_id}__{side_name}",
                    prompt_clean=str(clean_side.prompt),
                    prompt_corrupt=str(corrupt_side.prompt),
                    query_pos=int(q),
                    evidence_span=(int(ev[0]), int(ev[1])),
                    token_expected_id=int(tok_expected),
                    token_other_id=int(tok_other),
                )
            )
    return out


def _iter_cf_examples(
    *,
    items: Sequence[Any],
    tokenizer: Any,
    max_items: int,
    seed: int,
) -> List[WhyFetchExample]:
    rows = list(items)
    if int(max_items) > 0 and len(rows) > int(max_items):
        rng = random.Random(int(seed))
        rng.shuffle(rows)
        rows = rows[: int(max_items)]

    out: List[WhyFetchExample] = []
    for it in rows:
        labels = [str(k) for k in it.choices.keys()]
        expected = str(it.cf.expected_label)
        other = str(it.base.expected_label) if str(it.base.expected_label) != expected else _pick_other_label(labels, expected)
        try:
            _cont_expected, tok_expected = select_single_token_continuation(tokenizer, list(it.choices[expected]))
            _cont_other, tok_other = select_single_token_continuation(tokenizer, list(it.choices[other]))
        except SingleTokenSelectionError:
            continue

        corr_ids = tokenizer(str(it.base.prompt), return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        clean_ids = tokenizer(str(it.cf.prompt), return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        div = divergence_span(corr_ids, clean_ids)
        if div is None or int(div.a_end) <= int(div.a_start):
            continue
        q, ev = _extract_query_and_span(
            getattr(it, "metadata", None),
            fallback_span=(int(div.a_start), int(div.a_end)),
            fallback_query=-1,
        )
        out.append(
            WhyFetchExample(
                task="cf",
                item_id=str(it.item_id),
                prompt_clean=str(it.cf.prompt),
                prompt_corrupt=str(it.base.prompt),
                query_pos=int(q),
                evidence_span=(int(ev[0]), int(ev[1])),
                token_expected_id=int(tok_expected),
                token_other_id=int(tok_other),
            )
        )
    return out


def _coh_base_id(item_id: str) -> str:
    s = str(item_id)
    return s.split("__", 1)[0] if "__" in s else s


def _iter_coh_examples(
    *,
    items: Sequence[Any],
    tokenizer: Any,
    max_items: int,
    seed: int,
) -> List[WhyFetchExample]:
    by_base: Dict[str, Dict[str, Any]] = {}
    for it in items:
        grp = str(getattr(it, "group", "main"))
        by_base.setdefault(_coh_base_id(str(it.item_id)), {})[grp] = it

    rows: List[Tuple[str, Any, Any]] = []
    for base, gmap in by_base.items():
        if "main" in gmap and "ablate_relevant" in gmap:
            rows.append((base, gmap["main"], gmap["ablate_relevant"]))
    if int(max_items) > 0 and len(rows) > int(max_items):
        rng = random.Random(int(seed))
        rng.shuffle(rows)
        rows = rows[: int(max_items)]

    out: List[WhyFetchExample] = []
    for base, clean, corrupt in rows:
        choices = {"valid": list(clean.valid_continuations), "invalid": list(clean.invalid_continuations)}
        try:
            _cont_expected, tok_expected = select_single_token_continuation(tokenizer, list(choices["valid"]))
            _cont_other, tok_other = select_single_token_continuation(tokenizer, list(choices["invalid"]))
        except SingleTokenSelectionError:
            continue

        corr_ids = tokenizer(str(corrupt.context), return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        clean_ids = tokenizer(str(clean.context), return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        div = divergence_span(corr_ids, clean_ids)
        if div is None or int(div.a_end) <= int(div.a_start):
            continue
        q, ev = _extract_query_and_span(
            getattr(clean, "metadata", None),
            fallback_span=(int(div.a_start), int(div.a_end)),
            fallback_query=-1,
        )
        out.append(
            WhyFetchExample(
                task="coh",
                item_id=str(base),
                prompt_clean=str(clean.context),
                prompt_corrupt=str(corrupt.context),
                query_pos=int(q),
                evidence_span=(int(ev[0]), int(ev[1])),
                token_expected_id=int(tok_expected),
                token_other_id=int(tok_other),
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Why-fetch analysis with QK/OV separation (TransformerLens backend).")
    p.add_argument("--task", type=str, default="disamb", choices=["disamb", "cf", "coh"])
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
    p.add_argument("--cf_path", type=str, default=str(root / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(root / "data" / "coherence.jsonl"))
    p.add_argument("--metadata_sidecar_path", type=str, default="")
    p.add_argument("--n_examples", type=int, default=100)

    p.add_argument("--layers", type=str, default="", help="Optional comma-separated layer ids.")
    p.add_argument("--heads_topk", type=int, default=4, help="Top-k heads per layer by fetch mass (0 = all heads).")

    p.add_argument("--bootstrap_n", type=int, default=500)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--data_error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip"],
    )

    p.add_argument("--rows_csv_path", type=str, default="")
    p.add_argument("--summary_csv_path", type=str, default="")
    p.add_argument("--manifest_path", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        args.n_examples = min(int(args.n_examples), 10)
        args.bootstrap_n = min(int(args.bootstrap_n), 100)
        args.heads_topk = min(int(args.heads_topk), 2)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])
    repro = seed_everything(ReproConfig(seed=int(args.seed), determinism=str(args.determinism)), device=device)
    versions = collect_versions()
    if str(args.determinism) == "strict" and not bool(repro.get("determinism_enforced", False)):
        raise ValueError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")

    task = str(args.task)
    if task == "disamb":
        try:
            items, ds_manifest = load_disamb_pairs_with_manifest(
                str(args.disamb_path), role="disamb", error_policy=str(args.data_error_policy)
            )
        except DatasetLoadError as e:
            raise ValueError(f"Failed loading DISAMB data: {e}") from e
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_disamb(items, by_id=md)
        examples: List[WhyFetchExample] = []
        dataset_info = {"disamb": ds_manifest.as_dict()}
    elif task == "cf":
        try:
            items, ds_manifest = load_counterfactual_pairs_with_manifest(
                str(args.cf_path), role="cf", error_policy=str(args.data_error_policy)
            )
        except DatasetLoadError as e:
            raise ValueError(f"Failed loading CF data: {e}") from e
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_cf(items, by_id=md)
        examples = []
        dataset_info = {"cf": ds_manifest.as_dict()}
    else:
        try:
            items, ds_manifest = load_coherence_items_with_manifest(
                str(args.coh_path), role="coh", error_policy=str(args.data_error_policy)
            )
        except DatasetLoadError as e:
            raise ValueError(f"Failed loading COH data: {e}") from e
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_coh(items, by_id=md)
        examples = []
        dataset_info = {"coh": ds_manifest.as_dict()}

    tl_backend = TransformerLensBackend()
    loaded = tl_backend.load(
        model_name_or_path=str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation="eager",
    )
    hook_backend = TransformerLensHookableBackend(model=loaded.model, tokenizer=loaded.tokenizer)

    # Build examples after tokenizer is available (single-token selection needs tokenizer).
    tokenizer = loaded.tokenizer
    if task == "disamb":
        items, _ = load_disamb_pairs_with_manifest(str(args.disamb_path), role="disamb", error_policy="warn_skip")
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_disamb(items, by_id=md)
        examples = _iter_disamb_examples(items=items, tokenizer=tokenizer, max_pairs=int(args.n_examples), seed=int(args.seed))
    elif task == "cf":
        items, _ = load_counterfactual_pairs_with_manifest(str(args.cf_path), role="cf", error_policy="warn_skip")
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_cf(items, by_id=md)
        examples = _iter_cf_examples(items=items, tokenizer=tokenizer, max_items=int(args.n_examples), seed=int(args.seed))
    else:
        items, _ = load_coherence_items_with_manifest(str(args.coh_path), role="coh", error_policy="warn_skip")
        if str(args.metadata_sidecar_path).strip():
            md = load_metadata_sidecar(str(args.metadata_sidecar_path), id_key="id")
            items = attach_evidence_metadata_coh(items, by_id=md)
        examples = _iter_coh_examples(items=items, tokenizer=tokenizer, max_items=int(args.n_examples), seed=int(args.seed))

    rows: List[Dict[str, Any]] = []
    layers = _parse_layers(str(args.layers))
    for ex in examples:
        clean_run = hook_backend.run_with_cache(
            prompt=str(ex.prompt_clean),
            capture=["pattern", "value"],
            prepend_bos=False,
        )
        corrupt_run = hook_backend.run_with_cache(
            prompt=str(ex.prompt_corrupt),
            capture=["pattern", "value"],
            prepend_bos=False,
        )
        clean_cache = dict(clean_run.cache)
        clean_cache["logits"] = clean_run.logits
        corrupt_cache = dict(corrupt_run.cache)
        corrupt_cache["logits"] = corrupt_run.logits
        tokens_corrupt = corrupt_run.meta.get("tokens")
        if not isinstance(tokens_corrupt, torch.Tensor):
            raise RuntimeError("Expected tokens in corrupt run metadata")

        per_ex = compute_qk_ov_effects(
            hook_backend=hook_backend,
            example=ex,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
            tokens_corrupt=tokens_corrupt,
            layers=layers,
            heads_topk=int(args.heads_topk),
        )
        rows.extend(per_ex)

    summary = aggregate_top_heads(
        rows,
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )

    root = Path(__file__).resolve().parent
    rows_path = Path(str(args.rows_csv_path)) if str(args.rows_csv_path).strip() else (root / "results" / f"why_fetch_{task}_rows.csv")
    summary_path = Path(str(args.summary_csv_path)) if str(args.summary_csv_path).strip() else (root / "results" / f"why_fetch_{task}_summary.csv")
    _write_csv(rows_path, rows)
    _write_csv(summary_path, summary)

    manifest_path = (
        Path(str(args.manifest_path))
        if str(args.manifest_path).strip()
        else summary_path.with_suffix(".manifest.json")
    )
    result_row = {
        "task": str(task),
        "model": str(args.model_name_or_path),
        "n_examples_requested": int(args.n_examples),
        "n_examples_used": int(len(examples)),
        "n_rows": int(len(rows)),
        "heads_topk": int(args.heads_topk),
        "layers": "" if layers is None else ",".join(str(x) for x in layers),
        "rows_csv_path": str(rows_path),
        "summary_csv_path": str(summary_path),
        "rows_csv_sha256": _sha256_file(rows_path) if rows_path.exists() else "",
        "summary_csv_sha256": _sha256_file(summary_path) if summary_path.exists() else "",
        "bootstrap_n": int(args.bootstrap_n),
        "bootstrap_seed": int(args.bootstrap_seed),
        "ci": float(args.ci),
        "device": str(device),
        "arch": str(loaded.architecture),
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path if task == "disamb" else (args.cf_path if task == "cf" else args.coh_path)),
        csv_path=str(summary_path),
        csv_sha256=result_row["summary_csv_sha256"],
        csv_n_rows=int(len(summary)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(examples)),
        "succeeded": int(len(examples)),
        "failed": 0,
        "skipped": 0,
        "invalid": 0,
        "fail_rate": 0.0,
        "skip_rate": 0.0,
        "invalid_rate": 0.0,
        "top_failure_types": [],
        "top_skip_types": [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    manifest["datasets"] = dataset_info
    manifest["versions"] = versions
    manifest["repro"] = repro
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote rows CSV: {rows_path}", flush=True)
    print(f"Wrote summary CSV: {summary_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
