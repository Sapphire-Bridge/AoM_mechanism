from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aom.interventions.sae_loader import list_gemma_scope_runs, load_gemma_scope_sae


def _write_scope_fixture(
    tmp_path, *, layer: int, width: str, run_name: str, d_in: int, d_sae: int, include_cfg: bool = True
) -> str:
    run_dir = tmp_path / f"layer_{layer}" / f"width_{width}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if include_cfg:
        cfg = {"d_in": int(d_in), "d_sae": int(d_sae)}
        (run_dir / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Intentionally use "ambiguous" orientations to test canonicalization:
    # W_enc: (d_sae, d_in) and W_dec: (d_in, d_sae) should be auto-transposed.
    W_enc = np.random.randn(d_sae, d_in).astype(np.float32)
    W_dec = np.random.randn(d_in, d_sae).astype(np.float32)
    b_enc = np.random.randn(d_sae).astype(np.float32)
    b_dec = np.random.randn(d_in).astype(np.float32)
    np.savez(run_dir / "params.npz", W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec)

    return str(tmp_path)


def test_load_gemma_scope_sae_local_fixture_transposes_weights(tmp_path):
    root = _write_scope_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_sae=16)
    sae, meta = load_gemma_scope_sae(root, layer=0, width="16k", device="cpu", dtype="float32")

    assert meta.d_in == 8
    assert meta.d_sae == 16
    assert sae.W_enc.shape == (8, 16)
    assert sae.W_dec.shape == (16, 8)

    x = torch.randn(2, 5, 8)
    f = sae.encode(x)
    x_hat = sae.decode(f)
    assert f.shape == (2, 5, 16)
    assert x_hat.shape == (2, 5, 8)


def test_load_gemma_scope_sae_requires_run_name_when_ambiguous(tmp_path):
    root = _write_scope_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_sae=16)
    _ = _write_scope_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_105", d_in=8, d_sae=16)

    with pytest.raises(ValueError, match="Multiple Gemma Scope runs"):
        _ = load_gemma_scope_sae(root, layer=0, width="16k", device="cpu", dtype="float32")

    sae, meta = load_gemma_scope_sae(root, layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")
    assert meta.run_name == "average_l0_71"
    assert hasattr(sae, "encode") and hasattr(sae, "decode")


def test_list_gemma_scope_runs_local_fixture(tmp_path):
    root = _write_scope_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_sae=16)
    _ = _write_scope_fixture(tmp_path, layer=0, width="16k", run_name="average_l0_105", d_in=8, d_sae=16)
    runs = list_gemma_scope_runs(root, layer=0, width="16k")
    assert runs == sorted(runs)
    assert set(runs) == {"average_l0_71", "average_l0_105"}


def test_load_gemma_scope_sae_infers_cfg_when_cfg_json_missing(tmp_path):
    root = _write_scope_fixture(
        tmp_path, layer=0, width="16k", run_name="average_l0_71", d_in=8, d_sae=16, include_cfg=False
    )
    sae, meta = load_gemma_scope_sae(root, layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32")

    assert meta.d_in == 8
    assert meta.d_sae == 16
    assert meta.inferred_from_weights is True
    assert meta.cfg_path is None
    assert sae.cfg.get("inferred_from_weights") is True
    assert sae.cfg.get("dtype") == "float32"


def test_load_gemma_scope_sae_hf_run_name_skips_directory_discovery(monkeypatch, tmp_path):
    try:
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        pytest.skip("huggingface_hub not installed")

    params_path = tmp_path / "params.npz"
    d_in, d_sae = 8, 16
    W_enc = np.random.randn(d_sae, d_in).astype(np.float32)
    W_dec = np.random.randn(d_in, d_sae).astype(np.float32)
    b_enc = np.random.randn(d_sae).astype(np.float32)
    b_dec = np.random.randn(d_in).astype(np.float32)
    np.savez(params_path, W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec)

    class _NoListApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("HfApi() should not be constructed when run_name is provided")

    def _fake_hf_hub_download(repo_id_or_path: str, *, filename: str, **kwargs) -> str:
        assert repo_id_or_path == "dummy/repo"
        assert filename == "layer_0/width_16k/average_l0_71/params.npz" or filename == "layer_0/width_16k/average_l0_71/cfg.json"
        if filename.endswith("params.npz"):
            return str(params_path)
        raise EntryNotFoundError("cfg.json not found")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoListApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_hub_download)

    sae, meta = load_gemma_scope_sae(
        "dummy/repo", layer=0, width="16k", run_name="average_l0_71", device="cpu", dtype="float32"
    )
    assert meta.run_name == "average_l0_71"
    assert meta.inferred_from_weights is True
    assert meta.cfg_path is None
    assert sae.W_enc.shape == (d_in, d_sae)
