from __future__ import annotations

from typing import Dict

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from aom.data.loaders import load_authority_pairs
from aom.data.schemas import AuthorityPair, AuthoritySite, PromptSide, SubstringSpan
from aom.interventions.patching import run_activation_patching
from aom.interventions.patching.authority_protocol import AuthorityLanguageGameProtocol, AuthorityPatchingConfig
from aom.interventions.sae_adapter import SAEInputTransform, SAEPatchConfig
from aom.metrics.authority_game import compute_authority_sae_subset_copy_effect


def _make_wordlevel_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


class IdentitySAE(nn.Module):
    def __init__(self, d_in: int):
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


def _toy_authority_pair() -> AuthorityPair:
    prompt_a = "X <BLOCK MODE=0> ACTION: Output the word EXECUTE. ADMIN_OVERRIDE: Answer:"
    prompt_b = "X <BLOCK MODE=1> ACTION: Output the word EXECUTE. ADMIN_OVERRIDE: Answer:"
    return AuthorityPair(
        pair_id="auth_test_0001",
        a=PromptSide(prompt=prompt_a, expected_label="IGNORE"),
        b=PromptSide(prompt=prompt_b, expected_label="EXECUTE"),
        choices={"EXECUTE": [" EXECUTE"], "IGNORE": [" IGNORE"]},
        sites={
            "marker": AuthoritySite(
                a=SubstringSpan(substr="<BLOCK MODE=0>", occurrence=0),
                b=SubstringSpan(substr="<BLOCK MODE=1>", occurrence=0),
            ),
            "imperative": AuthoritySite(
                a=SubstringSpan(substr="ACTION: Output the word EXECUTE.", occurrence=0),
                b=SubstringSpan(substr="ACTION: Output the word EXECUTE.", occurrence=0),
            ),
            "distractor": AuthoritySite(
                a=SubstringSpan(substr="ADMIN_OVERRIDE:", occurrence=0),
                b=SubstringSpan(substr="ADMIN_OVERRIDE:", occurrence=0),
            ),
        },
    )


def _toy_vocab() -> Dict[str, int]:
    toks = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "X",
        "<BLOCK",
        "MODE=0>",
        "MODE=1>",
        "ACTION:",
        "Output",
        "the",
        "word",
        "EXECUTE.",
        "ADMIN_OVERRIDE:",
        "Answer:",
        "EXECUTE",
        "IGNORE",
    ]
    return {t: i for i, t in enumerate(toks)}


def _toy_authority_pair_mismatched_site_lengths() -> AuthorityPair:
    prompt_a = "X ALPHA BETA Answer:"
    prompt_b = "X GAMMA Answer:"
    return AuthorityPair(
        pair_id="auth_test_mismatch_0001",
        a=PromptSide(prompt=prompt_a, expected_label="IGNORE"),
        b=PromptSide(prompt=prompt_b, expected_label="EXECUTE"),
        choices={"EXECUTE": [" EXECUTE"], "IGNORE": [" IGNORE"]},
        sites={
            "marker": AuthoritySite(
                a=SubstringSpan(substr="ALPHA BETA", occurrence=0),
                b=SubstringSpan(substr="GAMMA", occurrence=0),
            ),
        },
    )


def _toy_vocab_mismatch() -> Dict[str, int]:
    toks = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "X",
        "ALPHA",
        "BETA",
        "GAMMA",
        "Answer:",
        "EXECUTE",
        "IGNORE",
    ]
    return {t: i for i, t in enumerate(toks)}


def test_load_authority_pairs_example_smoke():
    items = load_authority_pairs("data/authority_pairs.jsonl", validate=True)
    assert len(items) >= 1
    it = items[0]
    assert "marker" in it.sites
    assert "distractor" in it.sites
    assert "imperative" in it.sites


def test_authority_protocol_build_cases_counts():
    tok = _make_wordlevel_tokenizer(_toy_vocab())
    it = _toy_authority_pair()
    proto = AuthorityLanguageGameProtocol()
    cases, skips = proto.build_cases(tokenizer=tok, items=[it], device=torch.device("cpu"))
    assert len(skips) == 0
    assert len(cases) == 2 * 3  # two directions * three sites
    assert {c.strata["site"] for c in cases} == {"marker", "imperative", "distractor"}
    assert all(isinstance(c.strata.get("patched_text", ""), str) for c in cases)


def test_authority_protocol_reduces_before_span_len_check():
    tok = _make_wordlevel_tokenizer(_toy_vocab_mismatch())
    it = _toy_authority_pair_mismatched_site_lengths()
    proto = AuthorityLanguageGameProtocol(
        config=AuthorityPatchingConfig(include_sites=("marker",), max_tokens_per_site=1, span_take="last")
    )
    cases, skips = proto.build_cases(tokenizer=tok, items=[it], device=torch.device("cpu"))
    assert len(skips) == 0
    assert len(cases) == 2  # two directions at one site
    assert all(len(c.receiver_span) == 1 and len(c.donor_span) == 1 for c in cases)


def test_authority_protocol_rejects_nonpositive_max_tokens():
    tok = _make_wordlevel_tokenizer(_toy_vocab())
    it = _toy_authority_pair()
    proto = AuthorityLanguageGameProtocol(
        config=AuthorityPatchingConfig(include_sites=("marker",), max_tokens_per_site=0, span_take="last")
    )
    with pytest.raises(ValueError, match="max_tokens_per_site must be positive"):
        _cases, _skips = proto.build_cases(tokenizer=tok, items=[it], device=torch.device("cpu"))


def test_authority_sae_subset_copy_empty_features_is_noop():
    tok = _make_wordlevel_tokenizer(_toy_vocab())
    config = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(_toy_vocab()), n_positions=64)
    model = GPT2LMHeadModel(config)
    model.eval()

    sae = IdentitySAE(d_in=16)
    it = _toy_authority_pair()
    res = compute_authority_sae_subset_copy_effect(
        model=model,
        tokenizer=tok,
        items=[it],
        device=torch.device("cpu"),
        layer=0,
        site="marker",
        sae=sae,
        transform=SAEInputTransform(scale=1.0),
        feature_ids=[],
        config=SAEPatchConfig(decode_strategy="delta_1decode"),
        normalize_by_length=True,
        bootstrap_n=50,
        bootstrap_seed=0,
    )
    assert res.n_cases_used == 2
    assert abs(float(res.mean_effect)) < 1e-6


def test_authority_sae_subset_copy_all_features_matches_raw_identity_sae():
    tok = _make_wordlevel_tokenizer(_toy_vocab())
    config = GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(_toy_vocab()), n_positions=64)
    model = GPT2LMHeadModel(config)
    model.eval()

    sae = IdentitySAE(d_in=16)
    it = _toy_authority_pair()

    proto = AuthorityLanguageGameProtocol(config=AuthorityPatchingConfig(include_sites=("marker",)))
    raw = run_activation_patching(
        model=model,
        tokenizer=tok,
        protocol=proto,
        items=[it],
        device=torch.device("cpu"),
        layers=[0],
        normalize_by_length=True,
        bootstrap_n=50,
        bootstrap_seed=0,
    )
    e_raw = float(raw["mean_max_effect"])

    sae_res = compute_authority_sae_subset_copy_effect(
        model=model,
        tokenizer=tok,
        items=[it],
        device=torch.device("cpu"),
        layer=0,
        site="marker",
        sae=sae,
        transform=SAEInputTransform(scale=1.0),
        feature_ids=list(range(16)),
        config=SAEPatchConfig(decode_strategy="delta_1decode"),
        normalize_by_length=True,
        bootstrap_n=50,
        bootstrap_seed=0,
    )
    assert abs(float(sae_res.mean_effect) - e_raw) < 1e-5
