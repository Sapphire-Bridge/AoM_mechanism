from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.reporting import generate_results_report, write_results_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan `results/` and write a Markdown summary report.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--out_path", type=str, default=str(ROOT / "results" / "results_report.md"))
    p.add_argument(
        "--max_rows_per_csv",
        type=int,
        default=0,
        help="0 means read all rows; otherwise cap rows per CSV for faster scanning.",
    )
    p.add_argument("--no_file_inventory", action="store_true", help="Skip the per-CSV inventory table.")
    p.add_argument("--stdout", action="store_true", help="Print the report to stdout as well.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = None if int(args.max_rows_per_csv) <= 0 else int(args.max_rows_per_csv)
    include_inventory = not bool(args.no_file_inventory)

    if str(args.out_path).strip():
        out_path = write_results_report(
            args.out_path,
            results_dir=args.results_dir,
            max_rows_per_csv=max_rows,
            include_file_inventory=include_inventory,
        )
        if args.stdout:
            print(out_path.read_text(encoding="utf-8"))
        else:
            print(f"Wrote report: {out_path}")
        return

    report = generate_results_report(
        args.results_dir,
        max_rows_per_csv=max_rows,
        include_file_inventory=include_inventory,
    )
    print(report)


if __name__ == "__main__":
    main()

