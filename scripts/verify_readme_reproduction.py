#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_OUTPUTS = (
    "clt_raw_comparability_l4_l8_l12_controls_full.csv",
    "clt_raw_comparability_l4_l8_l12_controls_full.summary.json",
    "r1_full_cpu_f32.csv",
    "r1_full_cpu_f32.summary.json",
    "r1_full_cpu_f64.csv",
    "r1_full_cpu_f64.summary.json",
    "r1_full_cpu_f64.endpoint_pair_aggregates_v2.csv",
    "r1_full_cpu_f64.endpoint_decomp_summary_v2.json",
    "r1_full_cpu_f64.endpoint_decomp_summary_v2.md",
    "clt_raw_comparability_cf_l4_l8_l12_final_f32.csv",
    "clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json",
    "clt_raw_comparability_coh_l4_l8_l12_final_f32.csv",
    "clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    observed: Any
    expected: Any
    reference_path: str
    note: str = ""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_get(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            raise KeyError(f"cannot traverse list with key {part!r} in {dotted!r}")
        cur = cur[part]
    return cur


def _per_layer_entry(summary: dict[str, Any], layer: int) -> dict[str, Any]:
    per_layer = summary["per_layer"]
    if isinstance(per_layer, dict):
        return per_layer[str(layer)]
    for row in per_layer:
        if int(row["layer"]) == int(layer):
            return row
    raise KeyError(f"Layer {layer} missing")


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
    missing = [run_root / rel for rel in EXPECTED_OUTPUTS if not (run_root / rel).exists()]
    if missing:
        return missing, []

    run_f64_summary = _load_json(run_root / "r1_full_cpu_f64.summary.json")
    ref_f64_summary_path = ROOT / "results" / "mom_endpoint_plan" / "r1_full_cpu_f64.summary.json"
    ref_f64_summary = _load_json(ref_f64_summary_path)

    run_endpoint = _load_json(run_root / "r1_full_cpu_f64.endpoint_decomp_summary_v2.json")
    ref_endpoint_path = ROOT / "results" / "mom_endpoint_plan" / "r1_full_cpu_f64.endpoint_decomp_summary_v2.json"
    ref_endpoint = _load_json(ref_endpoint_path)

    run_controls = _load_json(run_root / "clt_raw_comparability_l4_l8_l12_controls_full.summary.json")
    ref_controls_path = ROOT / "results" / "clt_raw_comparability_l4_l8_l12_controls_full.summary.json"
    ref_controls = _load_json(ref_controls_path)

    run_cf = _load_json(run_root / "clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json")
    ref_cf_path = ROOT / "results" / "clt_raw_comparability_cf_l4_l8_l12_final_f32.summary.json"
    ref_cf = _load_json(ref_cf_path)

    run_coh = _load_json(run_root / "clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json")
    ref_coh_path = ROOT / "results" / "clt_raw_comparability_coh_l4_l8_l12_final_f32.summary.json"
    ref_coh = _load_json(ref_coh_path)

    results: list[CheckResult] = [
        _compare_exact(
            "f64 counts.n_rows_analysis_included",
            _nested_get(run_f64_summary, "counts.n_rows_analysis_included"),
            _nested_get(ref_f64_summary, "counts.n_rows_analysis_included"),
            ref_f64_summary_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "f64 counts.n_invariant_fail_rows",
            _nested_get(run_f64_summary, "counts.n_invariant_fail_rows"),
            _nested_get(ref_f64_summary, "counts.n_invariant_fail_rows"),
            ref_f64_summary_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "f64 run_config.bootstrap_n",
            _nested_get(run_f64_summary, "run_config.bootstrap_n"),
            _nested_get(ref_f64_summary, "run_config.bootstrap_n"),
            ref_f64_summary_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "endpoint counts.n_rows_primary_logodds_applicable",
            _nested_get(run_endpoint, "counts.n_rows_primary_logodds_applicable"),
            _nested_get(ref_endpoint, "counts.n_rows_primary_logodds_applicable"),
            ref_endpoint_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "controls counts.n_rows_analysis_included",
            _nested_get(run_controls, "counts.n_rows_analysis_included"),
            _nested_get(ref_controls, "counts.n_rows_analysis_included"),
            ref_controls_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "controls counts.n_pairs_analysis_included",
            _nested_get(run_controls, "counts.n_pairs_analysis_included"),
            _nested_get(ref_controls, "counts.n_pairs_analysis_included"),
            ref_controls_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "controls counts.n_invariant_fail_rows",
            _nested_get(run_controls, "counts.n_invariant_fail_rows"),
            _nested_get(ref_controls, "counts.n_invariant_fail_rows"),
            ref_controls_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "controls arm_fail_counts",
            run_controls.get("arm_fail_counts", {}),
            ref_controls.get("arm_fail_counts", {}),
            ref_controls_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "cf counts.n_rows_analysis_included",
            _nested_get(run_cf, "counts.n_rows_analysis_included"),
            _nested_get(ref_cf, "counts.n_rows_analysis_included"),
            ref_cf_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "cf counts.n_invariant_fail_rows",
            _nested_get(run_cf, "counts.n_invariant_fail_rows"),
            _nested_get(ref_cf, "counts.n_invariant_fail_rows"),
            ref_cf_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "coh counts.n_rows_analysis_included",
            _nested_get(run_coh, "counts.n_rows_analysis_included"),
            _nested_get(ref_coh, "counts.n_rows_analysis_included"),
            ref_coh_path.relative_to(ROOT).as_posix(),
        ),
        _compare_exact(
            "coh counts.n_invariant_fail_rows",
            _nested_get(run_coh, "counts.n_invariant_fail_rows"),
            _nested_get(ref_coh, "counts.n_invariant_fail_rows"),
            ref_coh_path.relative_to(ROOT).as_posix(),
        ),
    ]

    for layer in (4, 8, 12):
        run_primary = run_endpoint["results_by_layer"][str(layer)]["primary_applicable_pairs"]
        ref_primary = ref_endpoint["results_by_layer"][str(layer)]["primary_applicable_pairs"]
        results.append(
            _compare_close(
                f"endpoint layer {layer} primary_applicable_pairs.ddm.mean",
                run_primary["ddm"]["mean"],
                ref_primary["ddm"]["mean"],
                ref_endpoint_path.relative_to(ROOT).as_posix(),
                atol=1e-4,
            )
        )
        results.append(
            _compare_close(
                f"endpoint layer {layer} primary_applicable_pairs.d_ca_diag_logz.mean",
                run_primary["d_ca_diag_logz"]["mean"],
                ref_primary["d_ca_diag_logz"]["mean"],
                ref_endpoint_path.relative_to(ROOT).as_posix(),
                atol=1e-9,
            )
        )

    control_layer_4_run = _per_layer_entry(run_controls, 4)
    control_layer_4_ref = _per_layer_entry(ref_controls, 4)
    for field in (
        "effect_STRESS_RECON_mean",
        "effect_A_mean",
        "effect_PRJ_PCA_mean",
        "effect_PRJ_RAND_mean_mean",
        "effect_STRESS_RESID_mean",
    ):
        results.append(
            _compare_close(
                f"controls layer 4 {field}",
                float(control_layer_4_run[field]),
                float(control_layer_4_ref[field]),
                ref_controls_path.relative_to(ROOT).as_posix(),
                atol=1e-4,
            )
        )

    ordering = (
        control_layer_4_run["effect_STRESS_RECON_mean"] > control_layer_4_run["effect_A_mean"] > control_layer_4_run["effect_PRJ_PCA_mean"] > control_layer_4_run["effect_PRJ_RAND_mean_mean"]
        and control_layer_4_run["effect_STRESS_RESID_mean"] < 0
    )
    results.append(
        CheckResult(
            name="controls layer 4 ordering",
            status="pass" if ordering else "fail",
            observed={
                "recon": control_layer_4_run["effect_STRESS_RECON_mean"],
                "raw_a": control_layer_4_run["effect_A_mean"],
                "pca": control_layer_4_run["effect_PRJ_PCA_mean"],
                "rand": control_layer_4_run["effect_PRJ_RAND_mean_mean"],
                "resid": control_layer_4_run["effect_STRESS_RESID_mean"],
            },
            expected="RECON > Raw A > PCA > Random and RESID < 0",
            reference_path=ref_controls_path.relative_to(ROOT).as_posix(),
        )
    )

    for layer in (4, 8, 12):
        run_cf_layer = _per_layer_entry(run_cf, layer)
        ref_cf_layer = _per_layer_entry(ref_cf, layer)
        run_coh_layer = _per_layer_entry(run_coh, layer)
        ref_coh_layer = _per_layer_entry(ref_coh, layer)
        results.append(
            _compare_close(
                f"cf layer {layer} crr_C_over_A_mean",
                float(run_cf_layer["crr_C_over_A_mean"]),
                float(ref_cf_layer["crr_C_over_A_mean"]),
                ref_cf_path.relative_to(ROOT).as_posix(),
                atol=1e-4,
            )
        )
        results.append(
            _compare_close(
                f"coh layer {layer} crr_C_over_A_mean",
                float(run_coh_layer["crr_C_over_A_mean"]),
                float(ref_coh_layer["crr_C_over_A_mean"]),
                ref_coh_path.relative_to(ROOT).as_posix(),
                atol=1e-4,
            )
        )

    return missing, results


def _render_report(run_root: Path, missing: list[Path], results: list[CheckResult]) -> str:
    lines = [
        "# README Reproduction Verification",
        "",
        f"- run root: `{run_root}`",
    ]
    if missing:
        lines.extend(["", "## Missing Outputs", ""])
        lines.extend(f"- `{path.name}`" for path in missing)
        return "\n".join(lines) + "\n"

    lines.extend(["", "## Output Files", ""])
    lines.extend(f"- `{rel}`" for rel in EXPECTED_OUTPUTS)

    lines.extend(["", "## Numeric Checks", ""])
    for result in results:
        status = "PASS" if result.status == "pass" else "FAIL"
        detail = f"observed={result.observed!r}; expected={result.expected!r}"
        if result.note:
            detail += f"; {result.note}"
        lines.append(f"- {status}: `{result.name}` ({detail}; reference=`{result.reference_path}`)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify README core-claim reproduction outputs against tracked reference artifacts.")
    parser.add_argument("--run_root", required=True, help="Directory containing the README reproduction outputs.")
    parser.add_argument("--report_path", default="", help="Optional markdown report path.")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    missing, results = verify_run(run_root)
    report = _render_report(run_root, missing, results)
    if args.report_path:
        Path(args.report_path).expanduser().resolve().write_text(report, encoding="utf-8")

    if missing:
        print(report, file=sys.stderr)
        return 2
    failures = [result for result in results if result.status != "pass"]
    if failures:
        print(report, file=sys.stderr)
        return 2

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
