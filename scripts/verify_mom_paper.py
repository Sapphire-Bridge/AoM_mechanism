#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_readme_reproduction import CheckResult, verify_run as verify_core_run


SUPPORT_EXPECTED_OUTPUTS = (
    "paper_support/gemma2b_raw_6layer_full_seed42.csv",
    "paper_support/gemma2b_raw_6layer_full_seed42.manifest.json",
    "paper_support/gemma2b_clt_6layer_full_seed42.csv",
    "paper_support/gemma2b_clt_6layer_full_seed42.manifest.json",
    "paper_support/gemma2b_sae.csv",
    "paper_support/gemma2b_sae.manifest.json",
)

SIX_LAYER_DIR = ROOT / "results" / "overnight_mech_20260225T135850Z"
FIXED_LAYER_SAE_DIR = ROOT / "results" / "mom_overnight_gemma2b_sae_20260224T165425Z"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_row(path: Path) -> dict[str, Any]:
    row = _load_json(path).get("results_row")
    if not isinstance(row, dict):
        raise KeyError(f"Missing results_row in {path}")
    return row


def _compare_exact(name: str, observed: Any, expected: Any, reference_path: str) -> CheckResult:
    return CheckResult(
        name=name,
        status="pass" if observed == expected else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
    )


def _compare_close(
    name: str,
    observed: float,
    expected: float,
    reference_path: str,
    *,
    atol: float,
) -> CheckResult:
    ok = math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=0.0)
    return CheckResult(
        name=name,
        status="pass" if ok else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
        note=f"abs_tol={atol}",
    )


def verify_run(run_root: Path) -> tuple[list[Path], list[CheckResult]]:
    missing, checks = verify_core_run(run_root)

    support_missing = [run_root / rel for rel in SUPPORT_EXPECTED_OUTPUTS if not (run_root / rel).exists()]
    if support_missing:
        return [*missing, *support_missing], checks

    raw_run = _manifest_row(run_root / "paper_support" / "gemma2b_raw_6layer_full_seed42.manifest.json")
    raw_ref_path = SIX_LAYER_DIR / "gemma2b_raw_6layer_full_seed42.manifest.json"
    raw_ref = _manifest_row(raw_ref_path)

    clt_run = _manifest_row(run_root / "paper_support" / "gemma2b_clt_6layer_full_seed42.manifest.json")
    clt_ref_path = SIX_LAYER_DIR / "gemma2b_clt_6layer_full_seed42.manifest.json"
    clt_ref = _manifest_row(clt_ref_path)

    sae_run = _manifest_row(run_root / "paper_support" / "gemma2b_sae.manifest.json")
    sae_ref_path = FIXED_LAYER_SAE_DIR / "gemma2b_sae.manifest.json"
    sae_ref = _manifest_row(sae_ref_path)

    raw_ref_rel = raw_ref_path.relative_to(ROOT).as_posix()
    clt_ref_rel = clt_ref_path.relative_to(ROOT).as_posix()
    sae_ref_rel = sae_ref_path.relative_to(ROOT).as_posix()

    for layer in (4, 8, 12, 16, 20, 24):
        key = f"cpt_effect_layer_{layer}"
        checks.append(
            _compare_close(
                f"six-layer raw {key}",
                float(raw_run[key]),
                float(raw_ref[key]),
                raw_ref_rel,
                atol=1e-4,
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
            ),
            _compare_close(
                "six-layer raw cpt_mean_sham_max_effect",
                float(raw_run["cpt_mean_sham_max_effect"]),
                float(raw_ref["cpt_mean_sham_max_effect"]),
                raw_ref_rel,
                atol=1e-6,
            ),
            _compare_close(
                "six-layer raw cpt_flip_rate_at_best_layer",
                float(raw_run["cpt_flip_rate_at_best_layer"]),
                float(raw_ref["cpt_flip_rate_at_best_layer"]),
                raw_ref_rel,
                atol=1e-6,
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
            ),
            _compare_close(
                "six-layer clt clt_cpt_mean_sham_max_effect",
                float(clt_run["clt_cpt_mean_sham_max_effect"]),
                float(clt_ref["clt_cpt_mean_sham_max_effect"]),
                clt_ref_rel,
                atol=1e-6,
            ),
            _compare_exact(
                "six-layer clt clt_cpt_mean_identity_max_abs_effect",
                clt_run["clt_cpt_mean_identity_max_abs_effect"],
                clt_ref["clt_cpt_mean_identity_max_abs_effect"],
                clt_ref_rel,
            ),
            _compare_close(
                "six-layer clt clt_cpt_flip_rate_at_best_layer",
                float(clt_run["clt_cpt_flip_rate_at_best_layer"]),
                float(clt_ref["clt_cpt_flip_rate_at_best_layer"]),
                clt_ref_rel,
                atol=1e-6,
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
            ),
            _compare_close(
                "fixed-layer sae sae_cpt_mean_max_effect",
                float(sae_run["sae_cpt_mean_max_effect"]),
                float(sae_ref["sae_cpt_mean_max_effect"]),
                sae_ref_rel,
                atol=1e-4,
            ),
            _compare_close(
                "fixed-layer sae sae_cpt_mean_sham_max_effect",
                float(sae_run["sae_cpt_mean_sham_max_effect"]),
                float(sae_ref["sae_cpt_mean_sham_max_effect"]),
                sae_ref_rel,
                atol=1e-6,
            ),
            _compare_exact(
                "fixed-layer sae sae_cpt_flip_rate_at_best_layer",
                sae_run["sae_cpt_flip_rate_at_best_layer"],
                sae_ref["sae_cpt_flip_rate_at_best_layer"],
                sae_ref_rel,
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_target_effect",
                float(sae_run["cpt_spec_mean_signed_target_effect"]),
                float(sae_ref["cpt_spec_mean_signed_target_effect"]),
                sae_ref_rel,
                atol=1e-4,
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_ctrl_effect",
                float(sae_run["cpt_spec_mean_signed_ctrl_effect"]),
                float(sae_ref["cpt_spec_mean_signed_ctrl_effect"]),
                sae_ref_rel,
                atol=1e-4,
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_mean_signed_delta",
                float(sae_run["cpt_spec_mean_signed_delta"]),
                float(sae_ref["cpt_spec_mean_signed_delta"]),
                sae_ref_rel,
                atol=1e-4,
            ),
            _compare_close(
                "fixed-layer specificity cpt_spec_win_rate_signed",
                float(sae_run["cpt_spec_win_rate_signed"]),
                float(sae_ref["cpt_spec_win_rate_signed"]),
                sae_ref_rel,
                atol=1e-4,
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
        for path in missing:
            print(f"[missing] {path}")
        return 2

    failed = [check for check in checks if check.status != "pass"]
    for check in checks:
        prefix = "[pass]" if check.status == "pass" else "[fail]"
        note = f" ({check.note})" if check.note else ""
        print(f"{prefix} {check.name}: observed={check.observed!r} expected={check.expected!r} ref={check.reference_path}{note}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
