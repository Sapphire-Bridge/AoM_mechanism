from __future__ import annotations

from scripts import prepare_reviewer_assets, reviewer_assets


def test_prepare_reviewer_assets_main_success(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(prepare_reviewer_assets, "prepare_model_snapshot", lambda: tmp_path / "model")
    monkeypatch.setattr(prepare_reviewer_assets, "materialize_reviewer_clt_bundle", lambda: tmp_path / "bundle")
    monkeypatch.setattr(prepare_reviewer_assets, "prepare_fixed_layer_sae_cache", lambda: tmp_path / "sae" / "params.npz")
    monkeypatch.setattr(
        prepare_reviewer_assets,
        "local_asset_results",
        lambda: [reviewer_assets.AssetCheckResult(name="local_model_cache", ok=True, detail="ready")],
    )
    monkeypatch.setattr(prepare_reviewer_assets, "_best_accelerator", lambda: "")

    rc = prepare_reviewer_assets.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "# MoM Reviewer Assets" in out
    assert "make reviewer-check" in out


def test_prepare_reviewer_assets_main_returns_failure_when_verification_fails(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(prepare_reviewer_assets, "prepare_model_snapshot", lambda: tmp_path / "model")
    monkeypatch.setattr(prepare_reviewer_assets, "materialize_reviewer_clt_bundle", lambda: tmp_path / "bundle")
    monkeypatch.setattr(prepare_reviewer_assets, "prepare_fixed_layer_sae_cache", lambda: tmp_path / "sae" / "params.npz")
    monkeypatch.setattr(
        prepare_reviewer_assets,
        "local_asset_results",
        lambda: [reviewer_assets.AssetCheckResult(name="local_model_cache", ok=False, detail="missing")],
    )
    monkeypatch.setattr(prepare_reviewer_assets, "_best_accelerator", lambda: "")

    rc = prepare_reviewer_assets.main()
    out = capsys.readouterr().out

    assert rc == 2
    assert "ready_for_offline_paper_run: `FAIL`" in out
