from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts import paper_requirements, reviewer_assets


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, probe=np.zeros((1,), dtype=np.float32))


def test_model_snapshot_ready_accepts_complete_snapshot(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00003.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(reviewer_assets, "probe_tokenizer", lambda snapshot: None)

    ok, detail = reviewer_assets.model_snapshot_ready(tmp_path)
    assert ok is True
    assert "verified config/tokenizer/weights" in detail


def test_probe_weight_file_requires_all_indexed_shards(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00003.safetensors",
                    "b": "model-00002-of-00003.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00003.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="missing weight shards"):
        reviewer_assets.probe_weight_file(tmp_path)


def test_scope_snapshot_bundle_source_ready_requires_expected_runs(tmp_path: Path) -> None:
    _write_npz(tmp_path / paper_requirements.representative_scope_params_path())

    ok, detail = reviewer_assets.scope_snapshot_bundle_source_ready(tmp_path)
    assert ok is False
    assert "missing required CLT params.npz files" in detail


def test_required_params_tree_ready_opens_all_required_files(tmp_path: Path) -> None:
    required = paper_requirements.required_scope_params_paths()
    for rel in required:
        _write_npz(tmp_path / rel)
    corrupt = tmp_path / required[-1]
    corrupt.write_bytes(b"not-an-npz")

    with pytest.raises(Exception):
        reviewer_assets.required_params_tree_ready(tmp_path, label="CLT bundle")


def test_bundle_or_cache_ready_prefers_existing_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "clt_bundle"
    for rel in paper_requirements.required_scope_params_paths():
        _write_npz(bundle / rel)

    ok, detail = reviewer_assets.bundle_or_cache_ready(bundle, tmp_path / "unused_snapshot")
    assert ok is True
    assert "CLT bundle ready" in detail


def test_scope_snapshot_fixed_layer_sae_ready_accepts_required_file(tmp_path: Path) -> None:
    _write_npz(tmp_path / paper_requirements.fixed_layer_sae_params_path())

    ok, detail = reviewer_assets.scope_snapshot_fixed_layer_sae_ready(tmp_path)
    assert ok is True
    assert "readable fixed-layer SAE support" in detail


def test_prepare_model_snapshot_downloads_pinned_model(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, str] = {}
    resolved = tmp_path / "resolved_snapshot"
    resolved.mkdir()

    def _fake_snapshot_download(*, repo_id: str, revision: str) -> str:
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        downloaded = tmp_path / "downloaded_snapshot"
        downloaded.mkdir(exist_ok=True)
        return str(downloaded)

    monkeypatch.setattr(reviewer_assets, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(reviewer_assets, "resolve_cached_snapshot", lambda repo_id, revision: resolved)

    snapshot = reviewer_assets.prepare_model_snapshot()
    assert snapshot == resolved
    assert calls == {
        "repo_id": paper_requirements.PAPER_MODEL_REPO_ID,
        "revision": paper_requirements.PAPER_MODEL_REVISION,
    }


def test_prepare_fixed_layer_sae_cache_downloads_required_files(tmp_path: Path, monkeypatch) -> None:
    filenames: list[str] = []

    def _fake_hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        filenames.append(filename)
        out = tmp_path / Path(filename).name
        out.write_bytes(b"weights")
        return str(out)

    monkeypatch.setattr(reviewer_assets, "hf_hub_download", _fake_hf_hub_download)

    params_path = reviewer_assets.prepare_fixed_layer_sae_cache()
    assert params_path.name == "params.npz"
    assert str(paper_requirements.fixed_layer_sae_params_path()) in filenames
    assert any(name.endswith("/cfg.json") for name in filenames)


def test_build_clt_bundle_materialization_command_uses_pinned_args() -> None:
    cmd = reviewer_assets.build_clt_bundle_materialization_command(python_executable="python")
    assert cmd[:2] == ("python", str(reviewer_assets.ROOT / "scripts" / "gemma_scope_to_clt.py"))
    assert "--preset" in cmd and "readme_core_bundle" in cmd
    assert "--revision" in cmd and paper_requirements.PAPER_SCOPE_REVISION in cmd
    assert "--out_dir" in cmd and str(reviewer_assets.ROOT / paper_requirements.PAPER_CLT_BUNDLE_PATH) in cmd


def test_materialize_reviewer_clt_bundle_raises_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        reviewer_assets.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="failed to materialize reviewer CLT bundle"):
        reviewer_assets.materialize_reviewer_clt_bundle(python_executable="python")
