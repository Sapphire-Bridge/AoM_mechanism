from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize CF/COH patching CSVs with heterogeneity-first tables.")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Metric prefix (e.g. 'cf_patch' or 'coh_patch'). If unset, inferred from CSV columns.",
    )
    p.add_argument("--out_path", type=str, default="", help="Optional path to write a Markdown report.")
    return p.parse_args()


def _as_float(s: object) -> Optional[float]:
    if s is None:
        return None
    try:
        v = float(str(s))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _infer_prefix(fieldnames: List[str]) -> str:
    if any(c.startswith("cf_patch_") for c in fieldnames):
        return "cf_patch"
    if any(c.startswith("coh_patch_") for c in fieldnames):
        return "coh_patch"
    return ""


def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out: List[str] = []
    out.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(_md_escape(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    csv_path = Path(str(args.csv_path))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        prefix = str(args.prefix).strip() or _infer_prefix(fieldnames)
        if not prefix:
            raise ValueError("Failed to infer prefix; pass --prefix explicitly.")

        rows = list(reader)

    key_cols = {
        "mean_max_effect": f"{prefix}_mean_max_effect",
        "mean_sham_max_effect": f"{prefix}_mean_sham_max_effect",
        "n_cases_used": f"{prefix}_n_cases_used",
        "cohens_d": f"{prefix}_comparison_expected_effect_shift_vs_invariant_cohens_d",
        "cohens_d_coh": f"{prefix}_comparison_condition_constraint_vs_irrelevant_cohens_d",
        "mean_diff": f"{prefix}_comparison_expected_effect_shift_vs_invariant_mean_diff",
        "mean_diff_coh": f"{prefix}_comparison_condition_constraint_vs_irrelevant_mean_diff",
    }

    extra_specs: List[tuple[str, str]] = []
    # CF primary strata
    shift_col = f"{prefix}_stratum_expected_effect__shift_mean_max_effect"
    inv_col = f"{prefix}_stratum_expected_effect__invariant_mean_max_effect"
    if shift_col in fieldnames and inv_col in fieldnames:
        extra_specs.extend([(shift_col, "shift_mean_max_effect"), (inv_col, "invariant_mean_max_effect")])
    # COH primary strata
    rel_col = f"{prefix}_stratum_condition__constraint_span_mean_max_effect"
    irr_col = f"{prefix}_stratum_condition__irrelevant_span_mean_max_effect"
    if rel_col in fieldnames and irr_col in fieldnames:
        extra_specs.extend([(rel_col, "constraint_mean_max_effect"), (irr_col, "irrelevant_mean_max_effect")])

    table_rows: List[List[str]] = []
    for r in rows:
        model = str(r.get("model", ""))
        seed = str(r.get("seed", ""))
        mean_eff = _as_float(r.get(key_cols["mean_max_effect"]))
        mean_sham = _as_float(r.get(key_cols["mean_sham_max_effect"]))
        n_used = r.get(key_cols["n_cases_used"], "")
        d = _as_float(r.get(key_cols["cohens_d"])) or _as_float(r.get(key_cols["cohens_d_coh"]))
        dd = _as_float(r.get(key_cols["mean_diff"])) or _as_float(r.get(key_cols["mean_diff_coh"]))

        base_row = [
            model,
            seed,
            "" if mean_eff is None else f"{mean_eff:.4f}",
            "" if mean_sham is None else f"{mean_sham:.4f}",
            str(n_used),
            "" if d is None else f"{d:.3f}",
            "" if dd is None else f"{dd:.4f}",
        ]
        for col, _hdr in extra_specs:
            v = _as_float(r.get(col))
            base_row.append("" if v is None else f"{v:.4f}")
        table_rows.append(base_row)

    table_rows.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0))

    md = []
    md.append(f"# Patching report: `{csv_path.name}`")
    md.append("")
    headers = ["model", "seed", "mean_max_effect", "mean_sham", "n_cases_used", "cohens_d", "mean_diff"]
    headers.extend([hdr for _col, hdr in extra_specs])
    md.append(_md_table(headers, table_rows))
    md.append("")

    out_text = "\n".join(md)
    if str(args.out_path).strip():
        out_path = Path(str(args.out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"Wrote report: {out_path}", flush=True)
    else:
        print(out_text, flush=True)


if __name__ == "__main__":
    main()
