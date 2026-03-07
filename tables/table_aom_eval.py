from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a LaTeX table from an aom_eval.csv.")
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


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
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
    out_lines.append("\\begin{tabular}{lrrrrrr}")
    out_lines.append("\\toprule")
    out_lines.append("Model & AoM & DISAMB & CF-dir & COH & KW-base & DISAMB(low-overlap)\\\\")
    out_lines.append("\\midrule")
    for model in sorted(by_model.keys()):
        rs = by_model[model]
        aom = _mean([x for x in (_as_float(r.get("aom_composite")) for r in rs) if x is not None])
        dis = _mean([x for x in (_as_float(r.get("disamb_accuracy")) for r in rs) if x is not None])
        cf = _mean([x for x in (_as_float(r.get("cf_shift_direction_accuracy")) for r in rs) if x is not None])
        coh = _mean([x for x in (_as_float(r.get("coh_constraint_accuracy")) for r in rs) if x is not None])
        kw = _mean([x for x in (_as_float(r.get("baseline_disamb_keyword_accuracy")) for r in rs) if x is not None])
        dis_low = _mean([x for x in (_as_float(r.get("disamb_accuracy_overlap_low")) for r in rs) if x is not None])
        out_lines.append(
            " & ".join(
                [
                    model.replace("_", "\\_"),
                    _fmt(aom),
                    _fmt(dis),
                    _fmt(cf),
                    _fmt(coh),
                    _fmt(kw),
                    _fmt(dis_low),
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

