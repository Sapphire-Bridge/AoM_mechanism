from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.metrics.cue_leakage import cue_stats_for_disamb_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit AoM-DISAMB for potential cue leakage via lexical overlap.")
    p.add_argument("--disamb_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--out_csv", type=str, default=str(ROOT / "results" / "disamb_cue_audit.csv"))
    p.add_argument(
        "--out_flagged_json",
        type=str,
        default="",
        help="Optional path to write a JSON list of flagged pair_ids (asymmetric >=2 content-word overlap).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_disamb_pairs(str(args.disamb_path), validate=True)

    rows = []
    flagged = []
    for it in items:
        s = cue_stats_for_disamb_pair(it)
        d = asdict(s)
        rows.append(d)
        if bool(s.asymmetric_flag_ge2):
            flagged.append(str(s.pair_id))

    rows.sort(
        key=lambda r: (
            0 if r.get("asymmetric_flag_ge2") else 1,
            -float(r.get("pair_jaccard_max", 0.0) or 0.0),
            -abs(int(r.get("a_shared_count", 0) or 0) - int(r.get("b_shared_count", 0) or 0)),
            str(r.get("pair_id", "")),
        )
    )

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if str(args.out_flagged_json).strip():
        out_flagged = Path(str(args.out_flagged_json))
        out_flagged.parent.mkdir(parents=True, exist_ok=True)
        out_flagged.write_text(json.dumps(flagged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {out_csv}", flush=True)
    print(f"Flagged {len(flagged)} pairs (asymmetric overlap >=2 content words).", flush=True)


if __name__ == "__main__":
    main()

