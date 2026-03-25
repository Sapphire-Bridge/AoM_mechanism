from __future__ import annotations

import sys
from argparse import Namespace
from types import SimpleNamespace

import torch

import aom_eval
from aom import baselines
from aom.interventions import sae_loader
from aom.metrics import coherence, composite, counterfactual, disamb, sae_patching


class _DummyModel:
    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)


def test_parse_args_accepts_sae_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aom_eval.py",
            "--model_name_or_path",
            "dummy-model",
            "--sae_revision",
            "test-sae-rev",
        ],
    )

    args = aom_eval.parse_args()

    assert args.sae_revision == "test-sae-rev"


def test_run_eval_with_loaded_forwards_sae_revision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(disamb, "compute_aom_disamb", lambda *args, **kwargs: {})
    monkeypatch.setattr(counterfactual, "compute_aom_cf", lambda *args, **kwargs: {})
    monkeypatch.setattr(coherence, "compute_aom_coh", lambda *args, **kwargs: {})
    monkeypatch.setattr(baselines, "compute_disamb_keyword_baseline", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        composite,
        "compute_composite_metric",
        lambda *args, **kwargs: SimpleNamespace(value=0.0, valid=True, n=0, reason=None),
    )
    monkeypatch.setattr(
        sae_patching,
        "compute_sae_cpt_context_swap_patching",
        lambda *args, **kwargs: {"mean_max_effect": 0.0},
    )

    def _fake_load_gemma_scope_sae(repo_id_or_path: str, **kwargs):
        captured["repo_id_or_path"] = repo_id_or_path
        captured.update(kwargs)
        return object(), SimpleNamespace()

    monkeypatch.setattr(sae_loader, "load_gemma_scope_sae", _fake_load_gemma_scope_sae)

    args = Namespace(
        no_length_norm=False,
        ci=0.95,
        bootstrap_n=10,
        bootstrap_seed=42,
        strict_metrics=False,
        composite_missing_policy="nan",
        config_path="",
        config_sha256="",
        protocol_path="",
        protocol_sha256="",
        protocol_sha256_verified=False,
        protocol_sha256_source="",
        protocol_name="",
        protocol_version="",
        protocol_prereg_tag="",
        dataset_manifest_path="",
        dataset_bundle_manifest_sha256="",
        dataset_bundle_manifest_name="",
        dataset_bundle_id="",
        disamb_path="",
        cf_path="",
        coh_path="",
        device="cpu",
        device_map=None,
        attn_implementation="eager",
        torch_dtype="float32",
        logprobs_dtype="float32",
        score_batch_size=1,
        use_prefix_cache=False,
        strict_finite=True,
        trust_remote_code=False,
        local_files_only=True,
        prompt_input="full_prompt",
        prompt_mode="raw",
        add_generation_prompt=True,
        revision="model-rev",
        tokenizer_revision=None,
        run_patching=False,
        patch_layers="",
        patch_allow_token_id_mismatch=False,
        run_sae_patching=True,
        sae_repo="google/gemma-scope-2b-pt-res",
        sae_revision="test-sae-rev",
        sae_width="16k",
        sae_run_name="average_l0_457",
        sae_l0_target=None,
        sae_layers="24",
        sae_scale=1.0,
        sae_dtype="float32",
        sae_decode_strategy="delta_1decode",
        sae_dtype_policy="sae",
        sae_eps_active=1e-6,
        run_clt_patching=False,
        run_patching_specificity=False,
    )

    out = aom_eval.run_eval_with_loaded(
        model_name="dummy-model",
        model=_DummyModel(),
        tokenizer=object(),
        architecture="dummy-arch",
        hf_model_commit_hash=None,
        hf_tokenizer_revision_effective=None,
        system_prompt_sha256="",
        chat_template_sha256="",
        args=args,
        seed=7,
        git_commit_hash="deadbeef",
        argv_redacted_json="[]",
        argv_sha256="argv-sha",
        tensor_device="cpu",
        disamb_items=[{"prompt": "prompt", "target_new": "A", "target_old": "B"}],
        cf_items=[],
        coh_items=[],
    )

    assert captured["repo_id_or_path"] == "google/gemma-scope-2b-pt-res"
    assert captured["revision"] == "test-sae-rev"
    assert out["sae_repo"] == "google/gemma-scope-2b-pt-res"
