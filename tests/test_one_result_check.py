from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from scripts import run_one_result_check, verify_one_result_check
from scripts.verify_readme_reproduction import CheckResult


def _reference_summary() -> dict:
    path = Path(__file__).resolve().parents[1] / "results" / "clt_raw_comparability_l4_l8_l12_controls_full.summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_one_result_command_spec_targets_layer_4_controls(tmp_path: Path) -> None:
    spec = run_one_result_check._one_result_command_spec(tmp_path, local_files_only=True, device="cpu")
    layer_idx = spec.argv.index("--layers")
    device_idx = spec.argv.index("--device")
    out_json_idx = spec.argv.index("--out_json")

    assert spec.argv[layer_idx + 1] == "4"
    assert spec.argv[device_idx + 1] == "cpu"
    assert "--local_files_only" in spec.argv
    assert spec.argv[out_json_idx + 1].endswith("one_result_controls_l4.summary.json")


def test_run_one_result_check_dry_run_emits_plan_only(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "one-result"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_one_result_check.py"),
        "--dry_run",
        "--run_root",
        str(run_root),
        "--local_files_only",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    out = proc.stdout

    assert "scripts/clt_raw_comparability.py" in out
    assert "--layers 4" in out
    assert "--device cpu" in out
    assert "one_result_controls_l4.summary.json" in out
    assert "- overall_status: `PLAN_ONLY`" in out


def test_run_one_result_check_gpu_dry_run_uses_resolved_accelerator(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "one-result-gpu"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_one_result_check.py"),
        "--dry_run",
        "--run_root",
        str(run_root),
        "--local_files_only",
        "--device",
        "auto",
        "--require_accelerator",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    out = proc.stdout

    assert "--device mps" in out or "--device cuda" in out


def test_run_one_result_check_requires_accelerator_if_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_one_result_check, "_resolve_requested_device", lambda device: "cpu")
    monkeypatch.setattr(run_one_result_check, "_device_available", lambda device: True)

    code = run_one_result_check.main(
        [
            "--dry_run",
            "--run_root",
            str(tmp_path / "one-result"),
            "--device",
            "auto",
            "--require_accelerator",
        ]
    )

    assert code == 2


def test_verify_one_result_check_passes_on_reference_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "one-result"
    run_root.mkdir()
    (run_root / "one_result_controls_l4.csv").write_text("placeholder\n", encoding="utf-8")
    (run_root / "one_result_controls_l4.summary.json").write_text(
        json.dumps(_reference_summary(), indent=2) + "\n",
        encoding="utf-8",
    )

    missing, checks = verify_one_result_check.verify_run(run_root)

    assert missing == []
    assert all(check.status == "pass" for check in checks)


def test_verify_one_result_check_fails_on_bad_ordering(tmp_path: Path) -> None:
    summary = deepcopy(_reference_summary())
    layer_4 = next(row for row in summary["per_layer"] if int(row["layer"]) == 4)
    layer_4["effect_STRESS_RECON_mean"] = float(layer_4["effect_A_mean"]) - 0.1

    run_root = tmp_path / "one-result"
    run_root.mkdir()
    (run_root / "one_result_controls_l4.csv").write_text("placeholder\n", encoding="utf-8")
    (run_root / "one_result_controls_l4.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _missing, checks = verify_one_result_check.verify_run(run_root)

    assert any(check.name == "one-result layer 4 ordering" and check.status == "fail" for check in checks)


def test_verify_one_result_check_fails_on_invariant_fail_rows(tmp_path: Path) -> None:
    summary = deepcopy(_reference_summary())
    summary["counts"]["n_invariant_fail_rows"] = 1

    run_root = tmp_path / "one-result"
    run_root.mkdir()
    (run_root / "one_result_controls_l4.csv").write_text("placeholder\n", encoding="utf-8")
    (run_root / "one_result_controls_l4.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _missing, checks = verify_one_result_check.verify_run(run_root)

    assert any(check.name == "one-result counts.n_invariant_fail_rows" and check.status == "fail" for check in checks)


def test_verify_one_result_check_warns_on_accelerator_pca_drift(tmp_path: Path) -> None:
    summary = deepcopy(_reference_summary())
    layer_4 = next(row for row in summary["per_layer"] if int(row["layer"]) == 4)
    layer_4["effect_PRJ_PCA_mean"] = float(layer_4["effect_PRJ_PCA_mean"]) + 0.01

    run_root = tmp_path / "one-result"
    run_root.mkdir()
    (run_root / "one_result_controls_l4.csv").write_text("placeholder\n", encoding="utf-8")
    (run_root / "one_result_controls_l4.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_root / "one_result_check_log.json").write_text(
        json.dumps({"records": [{"argv": ["python", "scripts/clt_raw_comparability.py", "--device", "mps"]}]}, indent=2) + "\n",
        encoding="utf-8",
    )

    _missing, checks = verify_one_result_check.verify_run(run_root)

    assert any(
        check.name == "one-result layer 4 effect_PRJ_PCA_mean"
        and check.status == "warn"
        and "accelerator run on mps" in (check.note or "")
        for check in checks
    )


def test_run_one_result_check_writes_preverify_log_for_accelerator(monkeypatch, tmp_path: Path) -> None:
    run_root = tmp_path / "one-result"
    report_path = run_root / "one_result_check_report.md"
    json_log_path = run_root / "one_result_check_log.json"
    run_root.mkdir()

    monkeypatch.setattr(run_one_result_check, "_resolve_requested_device", lambda device: "mps")
    monkeypatch.setattr(run_one_result_check, "_device_available", lambda device: True)

    def fake_run_one(spec, *, cwd, run_root, dry_run):
        (run_root / "one_result_controls_l4.csv").write_text("placeholder\n", encoding="utf-8")
        (run_root / "one_result_controls_l4.summary.json").write_text(
            json.dumps(_reference_summary(), indent=2) + "\n",
            encoding="utf-8",
        )
        return run_one_result_check.CommandRecord(
            name=spec.name,
            argv=tuple(spec.argv),
            display_command=" ".join(spec.argv),
            started_at_utc="start",
            ended_at_utc="end",
            exit_code=0,
            generated_files=("one_result_controls_l4.csv", "one_result_controls_l4.summary.json"),
            artifact_kind=spec.artifact_kind,
        )

    monkeypatch.setattr(run_one_result_check, "_run_one", fake_run_one)

    def fake_verify(run_root_arg: Path):
        payload = json.loads((run_root_arg / "one_result_check_log.json").read_text(encoding="utf-8"))
        argv = payload["records"][0]["argv"]
        assert "--device" in argv
        assert argv[argv.index("--device") + 1] == "mps"
        return [], [
            CheckResult(
                name="one-result layer 4 effect_PRJ_PCA_mean",
                status="warn",
                observed=0.0,
                expected=0.0,
                reference_path="ref.json",
                note="accelerator run on mps; PCA baseline drift is advisory only",
            )
        ]

    monkeypatch.setattr(run_one_result_check, "verify_run", fake_verify)

    code = run_one_result_check.main(
        [
            "--run_root",
            str(run_root),
            "--report_path",
            str(report_path),
            "--json_log_path",
            str(json_log_path),
            "--device",
            "auto",
            "--require_accelerator",
            "--local_files_only",
        ]
    )

    assert code == 0
