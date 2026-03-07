from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
from aom.interventions.clt_loader import calibrate_clt_scale
from aom_clt_check import compute_identity_logit_drift, compute_reconstruction_telemetry


class _IdentityCLT(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self._d_in = int(d_in)
        self._d_latent = int(d_in)
        self._d_out = int(d_in)
        self.W_dec = nn.Parameter(torch.eye(self._d_out))

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_latent(self) -> int:
        return self._d_latent

    @property
    def d_out(self) -> int:
        return self._d_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


class _DummyTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = int(vocab_size)

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):  # noqa: ARG002
        toks = [ord(ch) % self._vocab_size for ch in str(text)]
        if not toks:
            toks = [0]
        ids = torch.tensor([toks], dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def _tiny_model() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=97, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_clt_check_identity_and_recon_smoke():
    model = _tiny_model()
    tok = _DummyTokenizer(vocab_size=97)
    clt = _IdentityCLT(d_in=32)
    prompts = ["alpha beta", "gamma delta"]
    transform = CLTInputTransform(scale=1.0)
    cfg = CLTPatchConfig(decode_strategy="safe_2decode", dtype_policy="clt")

    ident = compute_identity_logit_drift(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        clt=clt,
        layer=0,
        transform=transform,
        config=cfg,
        device=torch.device("cpu"),
        tolerance=1e-6,
    )
    recon = compute_reconstruction_telemetry(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        clt=clt,
        layer=0,
        transform=transform,
        device=torch.device("cpu"),
    )

    assert ident.passed is True
    assert ident.max_abs_logit_delta <= 1e-6
    assert recon.recon_mse_mean <= 1e-8
    assert recon.recon_cos_mean >= 0.999999


def test_calibrate_clt_scale_smoke():
    model = _tiny_model()
    tok = _DummyTokenizer(vocab_size=97)
    clt = _IdentityCLT(d_in=32)
    enc = tok("calibrate me")
    best_scale, mse_by_scale = calibrate_clt_scale(
        model=model,
        clt=clt,
        layer=0,
        scales=(0.5, 1.0, 2.0),
        calibration_input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        device=torch.device("cpu"),
    )
    assert float(best_scale) in {0.5, 1.0, 2.0}
    assert set(mse_by_scale.keys()) == {0.5, 1.0, 2.0}
    assert all(v >= 0.0 for v in mse_by_scale.values())

