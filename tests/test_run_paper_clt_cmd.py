from __future__ import annotations

import pytest

from scripts.run_paper import _aom_eval_cmd


def _base_kwargs() -> dict[str, object]:
    return {
        "models": ["gpt2"],
        "device": "cpu",
        "torch_dtype": "float32",
        "attn_implementation": "eager",
        "local_files_only": True,
        "trust_remote_code": False,
        "disamb_path": "data/disamb_pairs.jsonl",
        "cf_path": "data/counterfactual.jsonl",
        "coh_path": "data/coherence.jsonl",
        "dataset_manifest_path": None,
        "bootstrap_n": 10,
        "bootstrap_seed": 0,
        "ci": 0.95,
        "csv_path": "results/out.csv",
    }


def test_aom_eval_cmd_includes_clt_flags() -> None:
    cmd = _aom_eval_cmd(
        **_base_kwargs(),
        run_clt_patching=True,
        clt_repo="/tmp/clt_bundle",
        clt_width="16k",
        clt_run_name="average_l0_71",
        clt_l0_target=71,
        clt_layers="12,13",
        clt_scale=1.25,
        clt_dtype="float32",
        clt_decode_strategy="delta_1decode",
        clt_dtype_policy="clt",
        clt_eps_active=1e-5,
    )

    assert "--run_clt_patching" in cmd
    assert "--clt_repo" in cmd and cmd[cmd.index("--clt_repo") + 1] == "/tmp/clt_bundle"
    assert "--clt_layers" in cmd and cmd[cmd.index("--clt_layers") + 1] == "12,13"
    assert "--clt_decode_strategy" in cmd and cmd[cmd.index("--clt_decode_strategy") + 1] == "delta_1decode"


def test_aom_eval_cmd_clt_requires_repo() -> None:
    with pytest.raises(ValueError, match="clt_repo"):
        _aom_eval_cmd(
            **_base_kwargs(),
            run_clt_patching=True,
            clt_repo="",
            clt_layers="12",
        )


def test_aom_eval_cmd_clt_requires_layers() -> None:
    with pytest.raises(ValueError, match="clt_layers"):
        _aom_eval_cmd(
            **_base_kwargs(),
            run_clt_patching=True,
            clt_repo="/tmp/clt_bundle",
            clt_layers="",
        )
