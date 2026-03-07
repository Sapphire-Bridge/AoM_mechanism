from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


METRICS_PATTERNS: tuple[str, ...] = (
    "aom_eval.csv",
    "cf_patching.csv",
    "coh_patching.csv",
    "cpt_layer_sweep_disamb_only.csv",
    "cpt_specificity_disamb_only.csv",
    "cpt_specificity_seed*_disamb_only.csv",
)

MECHANISTIC_PATTERNS: tuple[str, ...] = (
    "disamb_path_decomp_summary*.csv",
    "why_fetch_*_summary*.csv",
    "feature_families_*_feature_families_summary.csv",
    "feature_families_*_feature_family_validation.csv",
)

COMPLETENESS_PATTERNS: tuple[str, ...] = (
    "completeness.csv",
    "completeness_*.csv",
)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k): v for k, v in row.items()} for row in reader]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, base_columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(base_columns)
    seen = set(keys)
    for r in rows:
        for k in r.keys():
            kk = str(k)
            if kk not in seen:
                seen.add(kk)
                keys.append(kk)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out = {str(k): row.get(k, "") for k in keys}
            w.writerow(out)


def _iter_matching_files(run_dir: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for patt in patterns:
        for p in sorted(run_dir.glob(str(patt))):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _tag_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    study_name: str,
    model_label: str,
    model_name_or_path: str,
    run_dir: Path,
    source_file: str,
) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for r in rows:
        out = dict(r)
        out.update(
            {
                "study_name": str(study_name),
                "model_label": str(model_label),
                "model_name_or_path": str(model_name_or_path),
                "run_dir": str(run_dir),
                "source_file": str(source_file),
            }
        )
        tagged.append(out)
    return tagged


def _collect_group(
    *,
    run_dir: Path,
    patterns: Sequence[str],
    study_name: str,
    model_label: str,
    model_name_or_path: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in _iter_matching_files(run_dir, patterns):
        rows = _read_csv_rows(path)
        out.extend(
            _tag_rows(
                rows=rows,
                study_name=study_name,
                model_label=model_label,
                model_name_or_path=model_name_or_path,
                run_dir=run_dir,
                source_file=str(path.name),
            )
        )
    return out


def aggregate_scaling_outputs(
    *,
    study_name: str,
    model_runs: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> Dict[str, Path]:
    metrics_rows: List[Dict[str, Any]] = []
    mechanistic_rows: List[Dict[str, Any]] = []
    completeness_rows: List[Dict[str, Any]] = []

    for mr in model_runs:
        run_dir = Path(str(mr["run_dir"]))
        model_label = str(mr["model_label"])
        model_name_or_path = str(mr["model_name_or_path"])

        metrics_rows.extend(
            _collect_group(
                run_dir=run_dir,
                patterns=METRICS_PATTERNS,
                study_name=str(study_name),
                model_label=model_label,
                model_name_or_path=model_name_or_path,
            )
        )
        mechanistic_rows.extend(
            _collect_group(
                run_dir=run_dir,
                patterns=MECHANISTIC_PATTERNS,
                study_name=str(study_name),
                model_label=model_label,
                model_name_or_path=model_name_or_path,
            )
        )
        completeness_rows.extend(
            _collect_group(
                run_dir=run_dir,
                patterns=COMPLETENESS_PATTERNS,
                study_name=str(study_name),
                model_label=model_label,
                model_name_or_path=model_name_or_path,
            )
        )

    metrics_path = out_dir / "scaling_summary_metrics.csv"
    mechanistic_path = out_dir / "scaling_summary_mechanistic.csv"
    completeness_path = out_dir / "scaling_summary_completeness.csv"

    base_cols = ["study_name", "model_label", "model_name_or_path", "run_dir", "source_file"]
    _write_csv(metrics_path, metrics_rows, base_columns=base_cols)
    _write_csv(mechanistic_path, mechanistic_rows, base_columns=base_cols)
    _write_csv(completeness_path, completeness_rows, base_columns=base_cols)

    return {
        "metrics": metrics_path,
        "mechanistic": mechanistic_path,
        "completeness": completeness_path,
    }
