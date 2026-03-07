from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from aom.interventions.activation_patching import PatchSite, PatchSpanSite, get_hidden_states
from aom.interventions.patching.scoring import score_labels_next_continuations_patched_ids
from aom.interventions.sae_adapter import IdentityFeaturePolicy, SAEInputTransform
from aom.metrics.disamb import score_labels_next_continuations_patched
from aom.metrics.sae_patching import score_labels_next_continuations_sae_patched
from aom.utils import configure_scoring_performance


@pytest.fixture(autouse=True)
def _reset_scoring_performance():
    configure_scoring_performance(score_batch_size=1, use_prefix_cache=False)
    yield
    configure_scoring_performance(score_batch_size=1, use_prefix_cache=False)


@dataclass
class CharTokenizer:
    vocab_size: int = 128
    pad_token_id: int = 0
    eos_token_id: int = 1

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,  # noqa: ARG002
    ) -> Dict[str, torch.Tensor]:
        _ = return_tensors
        s = str(text)
        if not s:
            s = " "
        ids = [ord(ch) % int(self.vocab_size) for ch in s]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class IdentitySAE(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self._d_in = int(d_in)
        self._d_sae = int(d_in)
        self.W_dec = nn.Parameter(torch.eye(self._d_sae))

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_sae(self) -> int:
        return self._d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features


def _tiny_model(vocab_size: int = 128) -> GPT2LMHeadModel:
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=vocab_size, n_positions=64)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _assert_dict_scores_close(a: Dict[str, float], b: Dict[str, float], atol: float = 1e-5) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        assert abs(float(a[k]) - float(b[k])) <= float(atol), f"score mismatch for {k}: {a[k]} vs {b[k]}"


def test_disamb_patched_scoring_falls_back_when_prefill_fails(monkeypatch) -> None:
    model = _tiny_model()
    tokenizer = CharTokenizer(vocab_size=128)
    device = torch.device("cpu")
    prompt = "abcde"
    choices = {"x": [" f", " g"], "y": [" h", " i"]}

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    hs = get_hidden_states(model, prompt_ids)
    replacement = hs[1][0, 2, :].detach()
    patch_site = PatchSite(layer=0, token_index=2)

    configure_scoring_performance(score_batch_size=2, use_prefix_cache=False)
    expected = score_labels_next_continuations_patched(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt=prompt,
        choices=choices,
        device=device,
        patch_site=patch_site,
        replacement=replacement,
        normalize_by_length=True,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr("aom.metrics.disamb.prefill_with_patched_block_output_span", _boom)
    configure_scoring_performance(score_batch_size=2, use_prefix_cache=True)
    got = score_labels_next_continuations_patched(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt=prompt,
        choices=choices,
        device=device,
        patch_site=patch_site,
        replacement=replacement,
        normalize_by_length=True,
    )
    _assert_dict_scores_close(expected.by_label, got.by_label)


def test_sae_patched_scoring_falls_back_when_prefill_fails(monkeypatch) -> None:
    model = _tiny_model()
    tokenizer = CharTokenizer(vocab_size=128)
    device = torch.device("cpu")
    prompt = "abcde"
    choices = {"x": [" f", " g"], "y": [" h", " i"]}

    sae = IdentitySAE(d_in=32)
    transform = SAEInputTransform(scale=1.0)
    policy = IdentityFeaturePolicy()

    configure_scoring_performance(score_batch_size=2, use_prefix_cache=False)
    expected = score_labels_next_continuations_sae_patched(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt=prompt,
        choices=choices,
        device=device,
        layer=0,
        token_indices=[2],
        sae=sae,
        transform=transform,
        policy=policy,  # type: ignore[arg-type]
        config=None,
        normalize_by_length=True,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr("aom.metrics.sae_patching.prefill_with_sae_feature_patching_span", _boom)
    configure_scoring_performance(score_batch_size=2, use_prefix_cache=True)
    got = score_labels_next_continuations_sae_patched(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt=prompt,
        choices=choices,
        device=device,
        layer=0,
        token_indices=[2],
        sae=sae,
        transform=transform,
        policy=policy,  # type: ignore[arg-type]
        config=None,
        normalize_by_length=True,
    )
    _assert_dict_scores_close(expected.by_label, got.by_label)


def test_protocol_patched_scoring_falls_back_when_prefill_fails(monkeypatch) -> None:
    model = _tiny_model()
    tokenizer = CharTokenizer(vocab_size=128)
    device = torch.device("cpu")
    prompt_ids = tokenizer("abcde", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    choices = {"x": [" f", " g"], "y": [" h", " i"]}

    hs = get_hidden_states(model, prompt_ids)
    replacement = hs[1][0, [2], :].detach()
    patch_site = PatchSpanSite(layer=0, token_indices=(2,))

    configure_scoring_performance(score_batch_size=2, use_prefix_cache=False)
    expected = score_labels_next_continuations_patched_ids(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt_ids=prompt_ids,
        choices=choices,
        device=device,
        patch_site=patch_site,
        replacement=replacement,
        normalize_by_length=True,
        label_aggregation="logmeanexp",
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr("aom.interventions.patching.scoring.prefill_with_patched_block_output_span", _boom)
    configure_scoring_performance(score_batch_size=2, use_prefix_cache=True)
    got = score_labels_next_continuations_patched_ids(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        prompt_ids=prompt_ids,
        choices=choices,
        device=device,
        patch_site=patch_site,
        replacement=replacement,
        normalize_by_length=True,
        label_aggregation="logmeanexp",
    )
    _assert_dict_scores_close(expected, got)
