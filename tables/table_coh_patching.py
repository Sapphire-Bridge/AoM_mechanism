from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a LaTeX table from a coh_patching CSV.")
    p.add_argument("--in_csv", type=str, required=True)
    p.add_argument("--out_tex", type=str, default="")
    return p.parse_args()


def _as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _mean(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _fmt(x: Optional[float], *, digits: int = 4) -> str:
    if x is None:
        return ""
    return f"{float(x):.{digits}f}"


def main() -> None:
    args = parse_args()
    in_csv = Path(str(args.in_csv))
    rows: List[Dict[str, str]] = []
    with open(in_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})

    by_model: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_model.setdefault(str(r.get("model", "unknown")), []).append(r)

    out_lines: List[str] = []
    out_lines.append("\\begin{tabular}{lrrrrr}")
    out_lines.append("\\toprule")
    out_lines.append("Model & mean max effect & sham & constraint & irrelevant & Cohen's d\\\\")
    out_lines.append("\\midrule")
    for model in sorted(by_model.keys()):
        rs = by_model[model]
        mean_eff = _mean([x for x in (_as_float(r.get("coh_patch_mean_max_effect")) for r in rs) if x is not None])
        sham = _mean([x for x in (_as_float(r.get("coh_patch_mean_sham_max_effect")) for r in rs) if x is not None])
        rel = _mean(
            [
                x
                for x in (_as_float(r.get("coh_patch_stratum_condition__constraint_span_mean_max_effect")) for r in rs)
                if x is not None
            ]
        )
        irr = _mean(
            [
                x
                for x in (_as_float(r.get("coh_patch_stratum_condition__irrelevant_span_mean_max_effect")) for r in rs)
                if x is not None
            ]
        )
        d = _mean(
            [
                x
                for x in (_as_float(r.get("coh_patch_comparison_condition_constraint_vs_irrelevant_cohens_d")) for r in rs)
                if x is not None
            ]
        )
        out_lines.append(
            " & ".join(
                [
                    model.replace("_", "\\_"),
                    _fmt(mean_eff),
                    _fmt(sham),
                    _fmt(rel),
                    _fmt(irr),
                    _fmt(d, digits=3),
                ]
            )
            + "\\\\"
        )
    out_lines.append("\\bottomrule")
    out_lines.append("\\end{tabular}")
    out_text = "\n".join(out_lines) + "\n"

    if str(args.out_tex).strip():
        out_path = Path(str(args.out_tex))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
    else:
        print(out_text, end="")


if __name__ == "__main__":
    main()

