from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    row_key: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight prediction/control checks for AoM runs.")
    p.add_argument("--aom_eval_csv", type=str, default="")
    p.add_argument("--cf_patching_csv", type=str, default="")
    p.add_argument("--coh_patching_csv", type=str, default="")
    p.add_argument("--out_json", type=str, default="", help="Optional JSON output path for check results.")
    return p.parse_args()


def _as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(str(x))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in r]


def _row_key(row: Dict[str, str]) -> str:
    model = row.get("model", "")
    seed = row.get("seed", "")
    return f"model={model} seed={seed}"


def main() -> None:
    args = parse_args()
    checks: List[CheckResult] = []

    if str(args.aom_eval_csv).strip():
        p = Path(str(args.aom_eval_csv))
        for row in _read_rows(p):
            k = _row_key(row)
            # Cue leakage bounding: accuracy should not collapse on low-overlap items.
            acc_low = _as_float(row.get("disamb_accuracy_overlap_low"))
            acc_high = _as_float(row.get("disamb_accuracy_overlap_high"))
            if acc_low is not None and acc_high is not None:
                ok = (acc_low >= 0.5) and (acc_low + 1e-6 >= acc_high - 0.25)
                checks.append(
                    CheckResult(
                        name="disamb_overlap_strata_sanity",
                        ok=bool(ok),
                        detail=f"acc_low={acc_low:.3f} acc_high={acc_high:.3f}",
                        row_key=k,
                    )
                )

            kw = _as_float(row.get("baseline_disamb_keyword_accuracy"))
            if kw is not None:
                ok = kw < 0.95
                checks.append(
                    CheckResult(
                        name="disamb_keyword_baseline_not_trivial",
                        ok=bool(ok),
                        detail=f"keyword_acc={kw:.3f}",
                        row_key=k,
                    )
                )

    if str(args.cf_patching_csv).strip():
        p = Path(str(args.cf_patching_csv))
        for row in _read_rows(p):
            k = _row_key(row)
            mean_diff = _as_float(row.get("cf_patch_comparison_expected_effect_shift_vs_invariant_mean_diff"))
            d = _as_float(row.get("cf_patch_comparison_expected_effect_shift_vs_invariant_cohens_d"))
            if mean_diff is not None:
                checks.append(
                    CheckResult(
                        name="cf_patching_shift_gt_invariant_mean_diff",
                        ok=bool(mean_diff > 0.0),
                        detail=f"mean_diff={mean_diff:.4f}",
                        row_key=k,
                    )
                )
            if d is not None:
                checks.append(
                    CheckResult(
                        name="cf_patching_shift_gt_invariant_cohens_d",
                        ok=bool(d > 0.0),
                        detail=f"d={d:.3f}",
                        row_key=k,
                    )
                )

    if str(args.coh_patching_csv).strip():
        p = Path(str(args.coh_patching_csv))
        for row in _read_rows(p):
            k = _row_key(row)
            mean_diff = _as_float(row.get("coh_patch_comparison_condition_constraint_vs_irrelevant_mean_diff"))
            d = _as_float(row.get("coh_patch_comparison_condition_constraint_vs_irrelevant_cohens_d"))
            if mean_diff is not None:
                checks.append(
                    CheckResult(
                        name="coh_patching_constraint_gt_irrelevant_mean_diff",
                        ok=bool(mean_diff > 0.0),
                        detail=f"mean_diff={mean_diff:.4f}",
                        row_key=k,
                    )
                )
            if d is not None:
                checks.append(
                    CheckResult(
                        name="coh_patching_constraint_gt_irrelevant_cohens_d",
                        ok=bool(d > 0.0),
                        detail=f"d={d:.3f}",
                        row_key=k,
                    )
                )

    # Emit summary.
    n_ok = sum(1 for c in checks if c.ok)
    n_total = len(checks)
    n_bad = n_total - n_ok

    grouped: Dict[str, List[CheckResult]] = {}
    for c in checks:
        grouped.setdefault(c.row_key, []).append(c)

    for rk in sorted(grouped.keys()):
        bad = [c for c in grouped[rk] if not c.ok]
        if not bad:
            continue
        print(f"[WARN] {rk}:", flush=True)
        for c in bad:
            print(f"  - {c.name}: {c.detail}", flush=True)

    if str(args.out_json).strip():
        out = [asdict(c) for c in checks]
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote checks: {out_path}", flush=True)

    if n_bad > 0:
        print(f"Checks: {n_ok}/{n_total} OK ({n_bad} warnings).", flush=True)
    else:
        print(f"Checks: {n_ok}/{n_total} OK.", flush=True)


if __name__ == "__main__":
    main()

