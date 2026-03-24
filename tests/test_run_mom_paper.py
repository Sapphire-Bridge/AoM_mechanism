from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import run_mom_paper
from scripts.run_readme_reproduction import CommandRecord, _readme_command_specs
from scripts.verify_readme_reproduction import CheckResult, MissingArtifact


def test_paper_support_command_specs_are_support_kind_and_cpu(tmp_path: Path) -> None:
    specs = run_mom_paper._paper_support_command_specs(tmp_path, local_files_only=True)
    assert {spec.artifact_kind for spec in specs} == {"support"}
    for spec in specs:
        idx = spec.argv.index("--device")
        assert spec.argv[idx + 1] == "cpu"
        assert "--device auto" not in spec.argv


def test_status_summary_uses_artifact_kind_not_names(tmp_path: Path) -> None:
    support_record = CommandRecord(
        name="renamed-stage",
        argv=(),
        display_command="python something.py",
        started_at_utc="",
        ended_at_utc="",
        exit_code=1,
        generated_files=(),
        artifact_kind="support",
    )
    core_check = CheckResult(
        name="renamed-check",
        status="fail",
        observed=1,
        expected=2,
        reference_path="ref.json",
        artifact_kind="core",
    )
    missing = [MissingArtifact(tmp_path / "paper_support" / "missing.csv", artifact_kind="support")]

    core_status, support_status, overall_status = run_mom_paper._status_summary(
        records=[support_record],
        missing=missing,
        checks=[core_check],
    )

    assert core_status == "FAIL"
    assert support_status == "FAIL"
    assert overall_status == "FAIL"


def test_status_summary_fails_closed_on_invalid_status_and_kind() -> None:
    invalid_check = CheckResult(
        name="unknown-state",
        status="error",
        observed=1,
        expected=1,
        reference_path="ref.json",
        artifact_kind="mystery",
    )

    core_status, support_status, overall_status = run_mom_paper._status_summary(
        records=[],
        missing=[],
        checks=[invalid_check],
    )

    assert core_status == "FAIL"
    assert support_status == "FAIL"
    assert overall_status == "FAIL"


def test_render_report_surfaces_warn_checks(tmp_path: Path) -> None:
    run_root = tmp_path / "mom-paper"
    run_root.mkdir()

    report = run_mom_paper._render_report(
        run_root=run_root,
        created_run_root=False,
        records=[],
        missing=[],
        checks=[
            CheckResult(
                name="compat fallback",
                status="warn",
                observed="csv_fallback",
                expected="manifest",
                reference_path="ref.csv",
                note="compat_fallback_used=ref.csv",
                artifact_kind="support",
            )
        ],
        dry_run=False,
    )

    assert "WARN" in report
    assert "compat fallback" in report


def test_support_provenance_uses_loader_and_reports_both_seeds(tmp_path: Path) -> None:
    run_root = tmp_path / "mom-paper"
    support_root = run_root / "paper_support"
    support_root.mkdir(parents=True)
    (support_root / "gemma2b_sae.csv").write_text(
        (
            "requested_device,device,model_param_dtype,logprobs_dtype,seed,bootstrap_seed\n"
            "cpu,cpu,float32,float32,20260224,42\n"
        ),
        encoding="utf-8",
    )

    lines = run_mom_paper._support_artifact_provenance_lines(run_root)

    assert len(lines) == 1
    assert "row_source=`csv_fallback`" in lines[0]
    assert "row_path=`$RUN_ROOT/paper_support/gemma2b_sae.csv`" in lines[0]
    assert "seed=`20260224`" in lines[0]
    assert "bootstrap_seed=`42`" in lines[0]


def test_readme_command_specs_use_expected_cf_and_coh_datasets(tmp_path: Path) -> None:
    specs = {spec.name: spec for spec in _readme_command_specs(tmp_path, local_files_only=True)}

    cf_spec = specs["CF task-axis comparability"]
    coh_spec = specs["COH task-axis comparability"]

    cf_cf_path_idx = cf_spec.argv.index("--cf_path")
    cf_coh_path_idx = cf_spec.argv.index("--coh_path")
    coh_cf_path_idx = coh_spec.argv.index("--cf_path")
    coh_coh_path_idx = coh_spec.argv.index("--coh_path")

    assert cf_spec.argv[cf_cf_path_idx + 1] == "data/counterfactual.jsonl"
    assert cf_spec.argv[cf_coh_path_idx + 1] == "data/coherence.jsonl"
    assert coh_spec.argv[coh_cf_path_idx + 1] == "data/counterfactual.jsonl"
    assert coh_spec.argv[coh_coh_path_idx + 1] == "data_paper_hardened_v2/coherence.jsonl"


def test_run_mom_paper_dry_run_emits_core_and_sae_support_commands(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "mom-paper"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_mom_paper.py"),
        "--dry_run",
        "--run_root",
        str(run_root),
        "--local_files_only",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    out = proc.stdout
    assert "scripts/clt_raw_comparability.py" in out
    assert "--device cpu" in out
    assert "--torch_dtype float32" in out
    assert "--run_clt_patching" in out
    assert "--run_sae_patching" in out
    assert "--run_patching_specificity" in out
    assert "--patch_layers 4,8,12,16,20,24" in out
    assert "gemma2b_sae.csv" in out
    assert "- core_claims_status: `NOT_EXECUTED`" in out
    assert "- support_artifacts_status: `NOT_EXECUTED`" in out
    assert "- overall_status: `PLAN_ONLY`" in out
    assert "- status: `PLAN_ONLY`" in out
    assert "--device auto" not in out
