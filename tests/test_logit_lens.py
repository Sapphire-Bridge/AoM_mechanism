import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aom.mechanistic.logit_lens import (
    compute_logit_lens_trace,
    encode_single_token_id,
    select_single_token_continuation,
)


class _DummyTokenizer:
    def __init__(self, vocab_size: int):
        self._vocab_size = int(vocab_size)

    def encode(self, text: str, add_special_tokens: bool = False):
        _ = add_special_tokens
        return [ord(ch) % self._vocab_size for ch in text]

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        _ = add_special_tokens
        ids = self.encode(text, add_special_tokens=False)
        if return_tensors != "pt":
            raise ValueError("dummy tokenizer only supports return_tensors='pt'")
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_encode_single_token_id_and_selection():
    tok = _DummyTokenizer(vocab_size=100)
    assert encode_single_token_id(tok, "A") == tok.encode("A")[0]
    with pytest.raises(ValueError):
        _ = encode_single_token_id(tok, "AB")

    c, tok_id = select_single_token_continuation(tok, ["AB", "C"])
    assert c == "C"
    assert tok_id == tok.encode("C")[0]


def test_logit_lens_trace_matches_final_logits():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    tok = _DummyTokenizer(vocab_size=100)
    prompt = "hello world"
    trace = compute_logit_lens_trace(
        model,
        tok,
        prompt,
        token_a_id=1,
        token_b_id=2,
        device=torch.device("cpu"),
        position=-1,
        lens="auto",
        compute_logits=True,
    )
    assert len(trace.points) >= 2
    assert pytest.approx(trace.final_logit_diff, abs=1e-3) == trace.points[-1].logit_diff
    assert pytest.approx(trace.points[0].logit_diff, abs=1e-6) == trace.points[0].delta_from_prev

    trace2 = compute_logit_lens_trace(
        model,
        tok,
        prompt,
        token_a_id=1,
        token_b_id=2,
        device=torch.device("cpu"),
        position=-1,
        lens="auto",
        compute_logits=True,
    )
    assert [p.logit_diff for p in trace.points] == [p.logit_diff for p in trace2.points]

