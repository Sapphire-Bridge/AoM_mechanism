#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure repo root importability when invoked as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.metrics.disamb import score_labels_next_continuations
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, set_seed


def _mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def main() -> None:
    p = argparse.ArgumentParser(description="Dump per-side DISAMB label scores to JSONL/CSV for diagnostics.")
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model id or local snapshot path.")
    p.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid network access for model/tokenizer loading (default: --local_files_only).",
    )
    p.add_argument("--disamb_path", type=str, required=True, help="DisambPair JSONL path.")
    p.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL path for per-side rows.")
    p.add_argument("--out_csv", type=str, default="", help="Optional output CSV path (flat columns).")
    p.add_argument("--out_summary_json", type=str, default="", help="Optional output summary JSON path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for scoring (default: auto).",
    )
    p.add_argument("--torch_dtype", type=str, default="float32", help="torch dtype (default: float32).")
    p.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="If >0, score only the first N pairs (for quick checks).",
    )
    args = p.parse_args()

    set_seed(int(args.seed))
    if str(args.device) == "auto":
        device = get_best_device()
    else:
        device = torch.device(str(args.device))

    # Loading by HF id may still attempt hub calls in some transformer versions.
    # Prefer local snapshot paths when running offline.
    t0 = time.time()
    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=False,
        attn_implementation="eager",
        device_map=None,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    load_sec = float(time.time() - t0)

    items = load_disamb_pairs(str(args.disamb_path), validate=True)
    if int(args.max_pairs) > 0:
        items = items[: int(args.max_pairs)]

    rows: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        for side_key, side in (("a", it.a), ("b", it.b)):
            scores = score_labels_next_continuations(
                model,
                tokenizer,
                side.prompt,
                it.choices,
                device,
                normalize_by_length=True,
            )
            pred = scores.argmax_label()
            expected = str(side.expected_label)
            best_other = max(v for k, v in scores.by_label.items() if k != expected)
            margin = float(scores.by_label[expected] - best_other)
            row = {
                "pair_idx": int(idx),
                "pair_id": str(it.pair_id),
                "subset": str(it.pair_id).split("_", 1)[0] if "_" in str(it.pair_id) else "",
                "side": str(side_key),
                "expected": expected,
                "pred": str(pred),
                "correct": int(pred == expected),
                "margin": float(margin),
                "help_minus_refuse": float(scores.by_label.get("help", 0.0) - scores.by_label.get("refuse", 0.0)),
                "scores": dict(scores.by_label),
            }
            rows.append(row)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if str(args.out_csv).strip():
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Flatten score dict into columns.
        labels = sorted({k for r in rows for k in r["scores"].keys()})
        fieldnames = [
            "pair_idx",
            "pair_id",
            "subset",
            "side",
            "expected",
            "pred",
            "correct",
            "margin",
            "help_minus_refuse",
            *[f"score_{lab}" for lab in labels],
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                flat = {k: r.get(k) for k in fieldnames}
                for lab in labels:
                    flat[f"score_{lab}"] = r["scores"].get(lab)
                w.writerow(flat)

    summary: Dict[str, Any] = {
        "model_name_or_path": str(args.model_name_or_path),
        "arch": str(loaded.architecture),
        "device": str(device),
        "seed": int(args.seed),
        "load_sec": float(load_sec),
        "n_pairs": int(len(items)),
        "n_sides": int(len(rows)),
    }
    if rows:
        summary["overall"] = {
            "accuracy": _mean([float(r["correct"]) for r in rows]),
            "mean_margin": _mean([float(r["margin"]) for r in rows]),
        }

        by_expected: Dict[str, Dict[str, Any]] = {}
        for exp in sorted({str(r["expected"]) for r in rows}):
            rr = [r for r in rows if str(r["expected"]) == exp]
            by_expected[exp] = {
                "n": int(len(rr)),
                "accuracy": _mean([float(r["correct"]) for r in rr]),
                "mean_margin": _mean([float(r["margin"]) for r in rr]),
                "pred_counts": dict(Counter(str(r["pred"]) for r in rr)),
            }
        summary["by_expected"] = by_expected

        by_subset: Dict[str, Dict[str, Any]] = {}
        for subset in sorted({str(r["subset"]) for r in rows}):
            rr = [r for r in rows if str(r["subset"]) == subset]
            by_subset[subset] = {
                "n": int(len(rr)),
                "accuracy": _mean([float(r["correct"]) for r in rr]),
                "mean_margin": _mean([float(r["margin"]) for r in rr]),
            }
        summary["by_subset"] = by_subset

    if str(args.out_summary_json).strip():
        out_summary = Path(args.out_summary_json)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} side rows -> {out_jsonl}")
    if str(args.out_csv).strip():
        print(f"Wrote CSV -> {str(args.out_csv)}")
    if str(args.out_summary_json).strip():
        print(f"Wrote summary -> {str(args.out_summary_json)}")
    if "overall" in summary:
        print(f"overall acc={summary['overall']['accuracy']:.3f} mean_margin={summary['overall']['mean_margin']:.3f}")


if __name__ == "__main__":
    main()
