from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PROVENANCE_COLUMNS: tuple[str, ...] = ("dataset_bundle_id", "git_commit", "argv_sha256")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper tables (LaTeX) from `results/` CSV artifacts.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "tables" / "out"))
    p.add_argument("--aom_eval_csv", type=str, default="")
    p.add_argument("--cf_patching_csv", type=str, default="")
    p.add_argument("--coh_patching_csv", type=str, default="")
    p.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip tables whose input CSV is missing (or missing required provenance columns) instead of failing.",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _resolve_default_csv(path: str, *, results_dir: Path, default_name: str) -> Path:
    p = Path(str(path)) if str(path).strip() else (results_dir / default_name)
    return p


def _read_csv_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
    if header is None:
        raise ValueError(f"CSV is empty: {str(path)}")
    return [str(x).strip() for x in header]


def _read_unique_column_value(path: Path, *, column: str, max_rows: int = 1000) -> str:
    values: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= max_rows:
                break
            v = "" if row.get(column) is None else str(row.get(column)).strip()
            if not v:
                continue
            values.add(v)
            if len(values) > 1:
                break
    if not values:
        raise ValueError(f"CSV {str(path)} has no non-empty values for required column {column!r}")
    if len(values) > 1:
        raise ValueError(f"CSV {str(path)} has multiple values for column {column!r}: {sorted(values)!r}")
    return next(iter(values))


def _run(cmd: List[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    args = parse_args()
    results_dir = Path(str(args.results_dir))
    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, Path, Path, Path]] = [
        (
            "aom_eval",
            ROOT / "tables" / "table_aom_eval.py",
            _resolve_default_csv(args.aom_eval_csv, results_dir=results_dir, default_name="aom_eval.csv"),
            out_dir / "aom_eval.tex",
        ),
        (
            "cf_patching",
            ROOT / "tables" / "table_cf_patching.py",
            _resolve_default_csv(args.cf_patching_csv, results_dir=results_dir, default_name="cf_patching.csv"),
            out_dir / "cf_patching.tex",
        ),
        (
            "coh_patching",
            ROOT / "tables" / "table_coh_patching.py",
            _resolve_default_csv(args.coh_patching_csv, results_dir=results_dir, default_name="coh_patching.csv"),
            out_dir / "coh_patching.tex",
        ),
    ]

    selected: List[Tuple[str, Path, Path, Path]] = []
    bundle_id_by_job: dict[str, str] = {}
    for name, script, in_csv, out_tex in jobs:
        if not script.exists():
            raise FileNotFoundError(f"Missing table script: {str(script)}")
        if not in_csv.exists():
            msg = f"[missing] {name}: {str(in_csv)}"
            if bool(args.skip_missing):
                print(msg, file=sys.stderr, flush=True)
                continue
            raise FileNotFoundError(msg)
        header = _read_csv_header(in_csv)
        missing_cols = [c for c in REQUIRED_PROVENANCE_COLUMNS if c not in set(header)]
        if missing_cols:
            msg = f"[missing] {name}: {str(in_csv)} (missing required columns: {missing_cols!r})"
            if bool(args.skip_missing):
                print(msg, file=sys.stderr, flush=True)
                continue
            raise ValueError(msg)
        try:
            for col in REQUIRED_PROVENANCE_COLUMNS:
                _read_unique_column_value(in_csv, column=col)
            bundle_id_by_job[name] = _read_unique_column_value(in_csv, column="dataset_bundle_id")
        except ValueError as e:
            msg = f"[missing] {name}: {str(in_csv)} ({e})"
            if bool(args.skip_missing):
                print(msg, file=sys.stderr, flush=True)
                continue
            raise ValueError(msg) from e
        selected.append((name, script, in_csv, out_tex))

    used_bundle_ids = {v for v in bundle_id_by_job.values() if str(v).strip()}
    if len(used_bundle_ids) > 1:
        raise ValueError(f"Mismatched dataset_bundle_id across table inputs: {bundle_id_by_job!r}")

    for name, script, in_csv, out_tex in selected:
        _run(
            [
                sys.executable,
                str(script),
                "--in_csv",
                str(in_csv),
                "--out_tex",
                str(out_tex),
            ],
            dry_run=bool(args.dry_run),
        )

    print(f"Wrote tables to {str(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
