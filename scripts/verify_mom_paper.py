#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_readme_reproduction import (
    CheckResult,
    MissingArtifact,
    _is_failure_status,
    _status_label,
    verify_run as verify_core_run,
)


SUPPORT_EXPECTED_CSV_OUTPUTS = (
    "paper_support/gemma2b_raw_6layer_full_seed42.csv",
    "paper_support/gemma2b_clt_6layer_full_seed42.csv",
    "paper_support/gemma2b_sae.csv",
)

SIX_LAYER_DIR = ROOT / "results" / "overnight_mech_20260225T135850Z"
FIXED_LAYER_SAE_DIR = ROOT / "results" / "mom_overnight_gemma2b_sae_20260224T165425Z"
NUMERIC_EQUIVALENT_FIELDS = {
    "fixed-layer sae sae_cpt_flip_rate_at_best_layer",
}
SUPPORT_CONTROL_NEAR_ZERO_ATOL = 2e-6


@dataclass(frozen=True)
class LoadedResultRow:
    row: dict[str, Any]
    source_kind: str
    source_path: Path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_reference_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _csv_single_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise KeyError(f"CSV has no data rows: {path}")
    if len(rows) != 1:
        raise KeyError(f"Expected exactly one data row in {path}; found {len(rows)}")
    return dict(rows[0])


def _load_result_row(path: Path) -> LoadedResultRow:
    if path.exists():
        row = _load_json(path).get("results_row")
        if not isinstance(row, dict):
            raise KeyError(f"Missing results_row in {path}")
        return LoadedResultRow(row=dict(row), source_kind="manifest", source_path=path)
    if str(path.name).endswith(".manifest.json"):
        csv_name = str(path.name).replace(".manifest.json", ".csv")
        csv_path = path.with_name(csv_name)
        if csv_path.exists():
            return LoadedResultRow(row=_csv_single_row(csv_path), source_kind="csv_fallback", source_path=csv_path)
    raise FileNotFoundError(path)


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _compare_exact(
    name: str,
    observed: Any,
    expected: Any,
    reference_path: str,
    *,
    artifact_kind: str = "core",
) -> CheckResult:
    ok = observed == expected
    if name in NUMERIC_EQUIVALENT_FIELDS:
        observed_num = _coerce_numeric(observed)
        expected_num = _coerce_numeric(expected)
        if observed_num is not None and expected_num is not None:
            ok = math.isclose(observed_num, expected_num, abs_tol=0.0, rel_tol=0.0)
    return CheckResult(
        name=name,
        status="pass" if ok else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
        artifact_kind=artifact_kind,
    )


def _compare_close(
    name: str,
    observed: float,
    expected: float,
    reference_path: str,
    *,
    atol: float,
    artifact_kind: str = "core",
) -> CheckResult:
    ok = math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=0.0)
    return CheckResult(
        name=name,
        status="pass" if ok else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
        note=f"abs_tol={atol}",
        artifact_kind=artifact_kind,
    )


def _compat_source_check(label: str, loaded: LoadedResultRow, *, artifact_kind: str) -> CheckResult | None:
    if loaded.source_kind == "manifest":
        return None
    return CheckResult(
        name=f"compat {label} row_source",
        status="warn",
        observed=loaded.source_kind,
        expected="manifest",
        reference_path=_display_reference_path(loaded.source_path),
        note=f"compat_fallback_used={loaded.source_path.name}",
        artifact_kind=artifact_kind,
    )


def verify_run(run_root: Path) -> tuple[list[MissingArtifact], list[CheckResult]]:
    missing, checks = verify_core_run(run_root)

    support_missing = [
        MissingArtifact(run_root / rel, artifact_kind="support")
        for rel in SUPPORT_EXPECTED_CSV_OUTPUTS
        if not (run_root / rel).exists()
    ]
    if support_missing:
        return [*missing, *support_missing], checks

    raw_run_loaded = _load_result_row(run_root / "paper_support" / "gemma2b_raw_6layer_full_seed42.manifest.json")
    raw_ref_path = SIX_LAYER_DIR / "gemma2b_raw_6layer_full_seed42.manifest.json"
    raw_ref_loaded = _load_result_row(raw_ref_path)

    clt_run_loaded = _load_result_row(run_root / "paper_support" / "gemma2b_clt_6layer_full_seed42.manifest.json")
    clt_ref_path = SIX_LAYER_DIR / "gemma2b_clt_6layer_full_seed42.manifest.json"
    clt_ref_loaded = _load_result_row(clt_ref_path)

    sae_run_loaded = _load_result_row(run_root / "paper_support" / "gemma2b_sae.manifest.json")
    sae_ref_path = FIXED_LAYER_SAE_DIR / "gemma2b_sae.manifest.json"
    sae_ref_loaded = _load_result_row(sae_ref_path)

    for label, loaded in (
        ("six-layer raw run", raw_run_loaded),
        ("six-layer raw reference", raw_ref_loaded),
        ("six-layer clt run", clt_run_loaded),
        ("six-layer clt reference", clt_ref_loaded),
        ("fixed-layer sae run", sae_run_loaded),
        ("fixed-layer sae reference", sae_ref_loaded),
    ):
        compat_check = _compat_source_check(label, loaded, artifact_kind="support")
        if compat_check is not None:
            checks.append(compat_check)

    raw_run = raw_run_loaded.row
    raw_ref = raw_ref_loaded.row
    clt_run = clt_run_loaded.row
    clt_ref = clt_ref_loaded.row
    sae_run = sae_run_loaded.row
    sae_ref = sae_ref_loaded.row

    raw_ref_rel = _display_reference_path(raw_ref_path)
    clt_ref_rel = _display_reference_path(clt_ref_path)
    sae_ref_rel = _display_reference_path(sae_ref_path)

    for layer in (4, 8, 12, 16, 20, 24):
        key = f"cpt_effect_layer_{layer}"
        checks.append(
            _compare_close(
                f"six-layer raw {key}",
                float(raw_run[key]),
                float(raw_ref[key]),
                raw_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            )
        )

    checks.extend(
        [
            _compare_close(
                "six-layer raw cpt_mean_max_effect",
                float(raw_run["cpt_mean_max_effect"]),
                float(raw_ref["cpt_mean_max_effect"]),
                raw_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "six-layer raw cpt_mean_sham_max_effect",
                float(raw_run["cpt_mean_sham_max_effect"]),
                float(raw_ref["cpt_mean_sham_max_effect"]),
                raw_ref_rel,
                atol=SUPPORT_CONTROL_NEAR_ZERO_ATOL,
                artifact_kind="support",
            ),
            _compare_close(
                "six-layer raw cpt_flip_rate_at_best_layer",
                float(raw_run["cpt_flip_rate_at_best_layer"]),
                float(raw_ref["cpt_flip_rate_at_best_layer"]),
                raw_ref_rel,
                atol=1e-6,
                artifact_kind="support",
            ),
        ]
    )

    for layer in (4, 8, 12, 16, 20, 24):
        key = f"clt_cpt_effect_layer_{layer}"
        checks.append(
            _compare_close(
                f"six-layer clt {key}",
                float(clt_run[key]),
                float(clt_ref[key]),
                clt_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            )
        )

    checks.extend(
        [
            _compare_close(
                "six-layer clt clt_cpt_mean_max_effect",
                float(clt_run["clt_cpt_mean_max_effect"]),
                float(clt_ref["clt_cpt_mean_max_effect"]),
                clt_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "six-layer clt clt_cpt_mean_sham_max_effect",
                float(clt_run["clt_cpt_mean_sham_max_effect"]),
                float(clt_ref["clt_cpt_mean_sham_max_effect"]),
                clt_ref_rel,
                atol=1e-6,
                artifact_kind="support",
            ),
            _compare_close(
                "six-layer clt clt_cpt_mean_identity_max_abs_effect",
                float(clt_run["clt_cpt_mean_identity_max_abs_effect"]),
                float(clt_ref["clt_cpt_mean_identity_max_abs_effect"]),
                clt_ref_rel,
                atol=SUPPORT_CONTROL_NEAR_ZERO_ATOL,
                artifact_kind="support",
            ),
            _compare_close(
                "six-layer clt clt_cpt_flip_rate_at_best_layer",
                float(clt_run["clt_cpt_flip_rate_at_best_layer"]),
                float(clt_ref["clt_cpt_flip_rate_at_best_layer"]),
                clt_ref_rel,
                atol=1e-6,
                artifact_kind="support",
            ),
        ]
    )

    checks.extend(
        [
            _compare_close(
                "fixed-layer sae sae_cpt_effect_layer_24",
                float(sae_run["sae_cpt_effect_layer_24"]),
                float(sae_ref["sae_cpt_effect_layer_24"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer sae sae_cpt_mean_max_effect",
                float(sae_run["sae_cpt_mean_max_effect"]),
                float(sae_ref["sae_cpt_mean_max_effect"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer sae sae_cpt_mean_sham_max_effect",
                float(sae_run["sae_cpt_mean_sham_max_effect"]),
                float(sae_ref["sae_cpt_mean_sham_max_effect"]),
                sae_ref_rel,
                atol=1e-6,
                artifact_kind="support",
            ),
            _compare_exact(
                "fixed-layer sae sae_cpt_flip_rate_at_best_layer",
                sae_run["sae_cpt_flip_rate_at_best_layer"],
                sae_ref["sae_cpt_flip_rate_at_best_layer"],
                sae_ref_rel,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_target_effect",
                float(sae_run["cpt_spec_mean_signed_target_effect"]),
                float(sae_ref["cpt_spec_mean_signed_target_effect"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_ctrl_effect",
                float(sae_run["cpt_spec_mean_signed_ctrl_effect"]),
                float(sae_ref["cpt_spec_mean_signed_ctrl_effect"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_delta",
                float(sae_run["cpt_spec_mean_signed_delta"]),
                float(sae_ref["cpt_spec_mean_signed_delta"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_win_rate_signed",
                float(sae_run["cpt_spec_win_rate_signed"]),
                float(sae_ref["cpt_spec_win_rate_signed"]),
                sae_ref_rel,
                atol=1e-4,
                artifact_kind="support",
            ),
        ]
    )

    return missing, checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a MoM paper reproduction run against checked-in reference artifacts.")
    parser.add_argument("--run_root", required=True, help="Run directory produced by scripts/run_mom_paper.py.")
    args = parser.parse_args(argv)

    missing, checks = verify_run(Path(args.run_root).expanduser().resolve())
    if missing:
        for artifact in missing:
            print(f"[missing] {artifact.path}")
        return 2

    failed = [check for check in checks if _is_failure_status(check.status)]
    for check in checks:
        label = _status_label(check.status)
        prefix = f"[{label.lower()}]"
        note = f" ({check.note})" if check.note else ""
        print(f"{prefix} {check.name}: observed={check.observed!r} expected={check.expected!r} ref={check.reference_path}{note}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
