from __future__ import annotations

import csv

import pytest

from scripts import verify_mom_paper, verify_readme_reproduction


def test_compare_exact_only_coerces_whitelisted_numeric_field() -> None:
    result = verify_mom_paper._compare_exact(
        "fixed-layer sae sae_cpt_flip_rate_at_best_layer",
        0.0,
        "0.0",
        "ref.json",
        artifact_kind="support",
    )
    assert result.status == "pass"

    other = verify_mom_paper._compare_exact("flip-rate", 0.0, "0.0", "ref.json")
    assert other.status == "fail"


def test_load_result_row_falls_back_to_csv_when_manifest_missing(tmp_path) -> None:
    csv_path = tmp_path / "gemma2b_sae.csv"
    csv_path.write_text("sae_cpt_flip_rate_at_best_layer,other\n0.0,value\n", encoding="utf-8")

    loaded = verify_mom_paper._load_result_row(tmp_path / "gemma2b_sae.manifest.json")
    assert loaded.source_kind == "csv_fallback"
    assert loaded.row["sae_cpt_flip_rate_at_best_layer"] == "0.0"


def test_csv_fallback_requires_exactly_one_data_row(tmp_path) -> None:
    csv_path = tmp_path / "gemma2b_sae.csv"
    csv_path.write_text("sae_cpt_flip_rate_at_best_layer\n0.0\n1.0\n", encoding="utf-8")

    with pytest.raises(KeyError, match="exactly one data row"):
        verify_mom_paper._load_result_row(tmp_path / "gemma2b_sae.manifest.json")


def test_compat_source_check_reports_warning_for_fallback(tmp_path) -> None:
    csv_path = tmp_path / "gemma2b_sae.csv"
    csv_path.write_text("sae_cpt_flip_rate_at_best_layer\n0.0\n", encoding="utf-8")

    loaded = verify_mom_paper._load_result_row(tmp_path / "gemma2b_sae.manifest.json")
    check = verify_mom_paper._compat_source_check("fixed-layer sae reference", loaded, artifact_kind="support")
    assert check is not None
    assert check.status == "warn"
    assert check.artifact_kind == "support"


def test_status_helpers_fail_closed_on_unknown_status() -> None:
    assert verify_readme_reproduction._status_label("error") == "INVALID(error)"
    assert verify_readme_reproduction._is_failure_status("error") is True
    assert verify_readme_reproduction._is_failure_status("warn") is False


def test_control_metric_atol_relaxes_only_pca_field() -> None:
    assert verify_readme_reproduction._control_metric_atol("effect_PRJ_PCA_mean") == 2e-3
    assert verify_readme_reproduction._control_metric_atol("effect_A_mean") == 1e-4


def test_verify_run_accepts_support_csv_fallback_when_support_manifests_missing(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    support_dir = run_root / "paper_support"
    support_dir.mkdir(parents=True)

    six_layer_dir = tmp_path / "six_layer_ref"
    sae_dir = tmp_path / "sae_ref"
    six_layer_dir.mkdir()
    sae_dir.mkdir()

    monkeypatch.setattr(verify_mom_paper, "verify_core_run", lambda _run_root: ([], []))
    monkeypatch.setattr(verify_mom_paper, "SIX_LAYER_DIR", six_layer_dir)
    monkeypatch.setattr(verify_mom_paper, "FIXED_LAYER_SAE_DIR", sae_dir)

    def write_row(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    raw_row = {
        "cpt_effect_layer_4": "0.0",
        "cpt_effect_layer_8": "0.0",
        "cpt_effect_layer_12": "0.0",
        "cpt_effect_layer_16": "0.0",
        "cpt_effect_layer_20": "0.0",
        "cpt_effect_layer_24": "0.0",
        "cpt_mean_max_effect": "0.0",
        "cpt_mean_sham_max_effect": "0.0",
        "cpt_flip_rate_at_best_layer": "0.0",
    }
    clt_row = {
        "clt_cpt_effect_layer_4": "0.0",
        "clt_cpt_effect_layer_8": "0.0",
        "clt_cpt_effect_layer_12": "0.0",
        "clt_cpt_effect_layer_16": "0.0",
        "clt_cpt_effect_layer_20": "0.0",
        "clt_cpt_effect_layer_24": "0.0",
        "clt_cpt_mean_max_effect": "0.0",
        "clt_cpt_mean_sham_max_effect": "0.0",
        "clt_cpt_mean_identity_max_abs_effect": "0.0",
        "clt_cpt_flip_rate_at_best_layer": "0.0",
    }
    sae_row = {
        "sae_cpt_effect_layer_24": "0.0",
        "sae_cpt_mean_max_effect": "0.0",
        "sae_cpt_mean_sham_max_effect": "0.0",
        "sae_cpt_flip_rate_at_best_layer": "0.0",
        "cpt_spec_mean_signed_target_effect": "0.0",
        "cpt_spec_mean_signed_ctrl_effect": "0.0",
        "cpt_spec_mean_signed_delta": "0.0",
        "cpt_spec_win_rate_signed": "0.0",
    }

    write_row(support_dir / "gemma2b_raw_6layer_full_seed42.csv", raw_row)
    write_row(six_layer_dir / "gemma2b_raw_6layer_full_seed42.csv", raw_row)
    write_row(support_dir / "gemma2b_clt_6layer_full_seed42.csv", clt_row)
    write_row(six_layer_dir / "gemma2b_clt_6layer_full_seed42.csv", clt_row)
    write_row(support_dir / "gemma2b_sae.csv", sae_row)
    write_row(sae_dir / "gemma2b_sae.csv", sae_row)

    missing, checks = verify_mom_paper.verify_run(run_root)

    assert missing == []
    compat_checks = [check for check in checks if check.name.startswith("compat ")]
    assert len(compat_checks) == 6
    assert {check.status for check in compat_checks} == {"warn"}


def test_verify_run_accepts_near_zero_support_control_drift(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    support_dir = run_root / "paper_support"
    support_dir.mkdir(parents=True)

    six_layer_dir = tmp_path / "six_layer_ref"
    sae_dir = tmp_path / "sae_ref"
    six_layer_dir.mkdir()
    sae_dir.mkdir()

    monkeypatch.setattr(verify_mom_paper, "verify_core_run", lambda _run_root: ([], []))
    monkeypatch.setattr(verify_mom_paper, "SIX_LAYER_DIR", six_layer_dir)
    monkeypatch.setattr(verify_mom_paper, "FIXED_LAYER_SAE_DIR", sae_dir)

    def write_row(path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    raw_run_row = {
        "cpt_effect_layer_4": "0.0",
        "cpt_effect_layer_8": "0.0",
        "cpt_effect_layer_12": "0.0",
        "cpt_effect_layer_16": "0.0",
        "cpt_effect_layer_20": "0.0",
        "cpt_effect_layer_24": "0.0",
        "cpt_mean_max_effect": "0.0",
        "cpt_mean_sham_max_effect": "0.0",
        "cpt_flip_rate_at_best_layer": "0.0",
    }
    raw_ref_row = dict(raw_run_row)
    raw_ref_row["cpt_mean_sham_max_effect"] = "1.1072093676516122e-06"

    clt_run_row = {
        "clt_cpt_effect_layer_4": "0.0",
        "clt_cpt_effect_layer_8": "0.0",
        "clt_cpt_effect_layer_12": "0.0",
        "clt_cpt_effect_layer_16": "0.0",
        "clt_cpt_effect_layer_20": "0.0",
        "clt_cpt_effect_layer_24": "0.0",
        "clt_cpt_mean_max_effect": "0.0",
        "clt_cpt_mean_sham_max_effect": "0.0",
        "clt_cpt_mean_identity_max_abs_effect": "1.1758978499040941e-06",
        "clt_cpt_flip_rate_at_best_layer": "0.0",
    }
    clt_ref_row = dict(clt_run_row)
    clt_ref_row["clt_cpt_mean_identity_max_abs_effect"] = "0.0"

    sae_row = {
        "sae_cpt_effect_layer_24": "0.0",
        "sae_cpt_mean_max_effect": "0.0",
        "sae_cpt_mean_sham_max_effect": "0.0",
        "sae_cpt_flip_rate_at_best_layer": "0.0",
        "cpt_spec_mean_signed_target_effect": "0.0",
        "cpt_spec_mean_signed_ctrl_effect": "0.0",
        "cpt_spec_mean_signed_delta": "0.0",
        "cpt_spec_win_rate_signed": "0.0",
    }

    write_row(support_dir / "gemma2b_raw_6layer_full_seed42.csv", raw_run_row)
    write_row(six_layer_dir / "gemma2b_raw_6layer_full_seed42.csv", raw_ref_row)
    write_row(support_dir / "gemma2b_clt_6layer_full_seed42.csv", clt_run_row)
    write_row(six_layer_dir / "gemma2b_clt_6layer_full_seed42.csv", clt_ref_row)
    write_row(support_dir / "gemma2b_sae.csv", sae_row)
    write_row(sae_dir / "gemma2b_sae.csv", sae_row)

    missing, checks = verify_mom_paper.verify_run(run_root)

    assert missing == []
    assert all(check.status != "fail" for check in checks)


def test_main_prints_missing_artifact_paths(tmp_path, monkeypatch, capsys) -> None:
    missing_path = tmp_path / "paper_support" / "missing.csv"
    monkeypatch.setattr(
        verify_mom_paper,
        "verify_run",
        lambda _run_root: ([verify_readme_reproduction.MissingArtifact(missing_path, artifact_kind="support")], []),
    )

    exit_code = verify_mom_paper.main(["--run_root", str(tmp_path)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out.strip() == f"[missing] {missing_path}"
    assert "MissingArtifact(" not in captured.out
