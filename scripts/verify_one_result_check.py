#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_readme_reproduction import (
    CheckResult,
    MissingArtifact,
    _compare_close,
    _compare_exact,
    _control_metric_atol,
    _is_failure_status,
    _load_json,
    _per_layer_entry,
    _status_label,
)


EXPECTED_OUTPUTS = (
    "one_result_controls_l4.csv",
    "one_result_controls_l4.summary.json",
)


def _run_device(run_root: Path) -> str:
    log_path = run_root / "one_result_check_log.json"
    if not log_path.exists():
        return "cpu"
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            return "cpu"
        argv = records[0].get("argv", [])
        if "--device" not in argv:
            return "cpu"
        device = str(argv[argv.index("--device") + 1]).strip().lower()
        return device or "cpu"
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "cpu"


def verify_run(run_root: Path) -> tuple[list[MissingArtifact], list[CheckResult]]:
    missing = [MissingArtifact(run_root / rel, artifact_kind="core") for rel in EXPECTED_OUTPUTS if not (run_root / rel).exists()]
    if missing:
        return missing, []

    run_device = _run_device(run_root)
    run_summary = _load_json(run_root / "one_result_controls_l4.summary.json")
    ref_path = ROOT / "results" / "clt_raw_comparability_l4_l8_l12_controls_full.summary.json"
    ref_summary = _load_json(ref_path)
    ref_rel = ref_path.relative_to(ROOT).as_posix()

    run_layer_4 = _per_layer_entry(run_summary, 4)
    ref_layer_4 = _per_layer_entry(ref_summary, 4)

    checks: list[CheckResult] = [
        _compare_exact(
            "one-result counts.n_pairs_analysis_included",
            run_summary["counts"]["n_pairs_analysis_included"],
            ref_summary["counts"]["n_pairs_analysis_included"],
            ref_rel,
        ),
        _compare_exact(
            "one-result counts.n_invariant_fail_rows",
            run_summary["counts"]["n_invariant_fail_rows"],
            0,
            ref_rel,
        ),
        _compare_exact(
            "one-result arm_fail_counts",
            run_summary.get("arm_fail_counts", {}),
            {},
            ref_rel,
        ),
    ]

    for field in (
        "effect_STRESS_RECON_mean",
        "effect_A_mean",
        "effect_PRJ_PCA_mean",
        "effect_PRJ_RAND_mean_mean",
        "effect_STRESS_RESID_mean",
    ):
        check = _compare_close(
            f"one-result layer 4 {field}",
            float(run_layer_4[field]),
            float(ref_layer_4[field]),
            ref_rel,
            atol=_control_metric_atol(field),
        )
        if field == "effect_PRJ_PCA_mean" and run_device != "cpu" and check.status == "fail":
            check = replace(check, status="warn", note=f"accelerator run on {run_device}; PCA baseline drift is advisory only")
        checks.append(check)

    ordering = (
        run_layer_4["effect_STRESS_RECON_mean"] > run_layer_4["effect_A_mean"] > run_layer_4["effect_PRJ_PCA_mean"] > run_layer_4["effect_PRJ_RAND_mean_mean"]
        and run_layer_4["effect_STRESS_RESID_mean"] < 0
    )
    checks.append(
        CheckResult(
            name="one-result layer 4 ordering",
            status="pass" if ordering else "fail",
            observed={
                "recon": run_layer_4["effect_STRESS_RECON_mean"],
                "raw_a": run_layer_4["effect_A_mean"],
                "pca": run_layer_4["effect_PRJ_PCA_mean"],
                "rand": run_layer_4["effect_PRJ_RAND_mean_mean"],
                "resid": run_layer_4["effect_STRESS_RESID_mean"],
            },
            expected="RECON > Raw A > PCA > Random and RESID < 0",
            reference_path=ref_rel,
        )
    )
    return missing, checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the single-result reviewer check against tracked layer-4 controls.")
    parser.add_argument("--run_root", required=True, help="Directory containing the one-result-check outputs.")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    missing, checks = verify_run(run_root)
    for artifact in missing:
        print(f"[missing] {artifact}")
    for check in checks:
        note = f"; {check.note}" if check.note else ""
        print(
            f"[{_status_label(check.status).lower()}] {check.name}: observed={check.observed!r}; "
            f"expected={check.expected!r}{note}; reference={check.reference_path}"
        )

    if missing or any(_is_failure_status(check.status) for check in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
