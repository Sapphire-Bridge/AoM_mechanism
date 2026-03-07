from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..interventions.activation_patching import detect_architecture


LensMode = Literal["auto", "raw", "final_norm"]


class SingleTokenSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class LogitLensPoint:
    state_index: int
    block_index: Optional[int]
    logit_a: float
    logit_b: float
    logit_diff: float
    delta_from_prev: float


@dataclass(frozen=True)
class LogitLensTrace:
    arch: str
    lens: LensMode
    position: int
    token_a_id: int
    token_b_id: int
    apply_final_norm_intermediate: bool
    apply_final_norm_last: bool
    final_logit_diff: float
    points: Tuple[LogitLensPoint, ...]


def encode_single_token_id(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected a single token for {text!r}, got {len(ids)} tokens: {ids}")
    return int(ids[0])


def select_single_token_continuation(
    tokenizer: PreTrainedTokenizerBase, continuations: Sequence[str]
) -> Tuple[str, int]:
    for c in continuations:
        try:
            tok_id = encode_single_token_id(tokenizer, c)
        except ValueError:
            continue
        return str(c), int(tok_id)
    raise SingleTokenSelectionError(
        "No single-token continuation found; provide an explicit --target_a/--target_b."
    )


def _get_output_embedding(model: PreTrainedModel) -> torch.nn.Module:
    out = None
    try:
        out = model.get_output_embeddings()
    except Exception:
        out = None
    if out is None and hasattr(model, "lm_head"):
        out = getattr(model, "lm_head")
    if out is None or not hasattr(out, "weight"):
        raise ValueError("Could not access output embedding weights (need model.get_output_embeddings().weight).")
    return out  # type: ignore[return-value]


def _get_final_norm(model: PreTrainedModel, arch: str) -> Optional[torch.nn.Module]:
    if arch == "gpt2":
        return model.transformer.ln_f  # type: ignore[attr-defined]
    if arch in ("llama", "mistral", "qwen2", "qwen3"):
        return model.model.norm  # type: ignore[attr-defined]
    if arch == "gpt_neox":
        return model.gpt_neox.final_layer_norm  # type: ignore[attr-defined]
    return None


def _apply_norm(norm: torch.nn.Module, vec: torch.Tensor) -> torch.Tensor:
    # Most norms accept (..., hidden); use a minimal batch dim for safety.
    if vec.ndim == 1:
        v2 = vec.unsqueeze(0)
        out = norm(v2)
        return out.squeeze(0)
    return norm(vec)


def _logit_diff_from_vec(
    vec: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    token_a_id: int,
    token_b_id: int,
    compute_logits: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # weight: (vocab, hidden)
    wa = weight[token_a_id]
    wb = weight[token_b_id]
    va = torch.dot(vec, wa)
    vb = torch.dot(vec, wb)
    if bias is not None:
        va = va + bias[token_a_id]
        vb = vb + bias[token_b_id]
    diff = va - vb
    if not compute_logits:
        return diff, None, None
    return diff, va, vb


@torch.no_grad()
def compute_logit_lens_trace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    token_a_id: int,
    token_b_id: int,
    device: torch.device,
    position: int = -1,
    lens: LensMode = "auto",
    compute_logits: bool = True,
) -> LogitLensTrace:
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if int(token_a_id) == int(token_b_id):
        raise ValueError("token_a_id and token_b_id must be different")

    arch = detect_architecture(model)
    out_emb = _get_output_embedding(model)
    weight = out_emb.weight  # type: ignore[attr-defined]
    bias = getattr(out_emb, "bias", None)
    if bias is not None and not torch.is_tensor(bias):
        bias = None

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("Expected batch size 1 input_ids")
    seq_len = int(input_ids.size(1))
    if seq_len < 1:
        raise ValueError("Prompt tokenization produced empty input_ids")

    pos = int(position)
    if pos < 0:
        pos = seq_len + pos
    if pos < 0 or pos >= seq_len:
        raise ValueError(f"position {position} out of range for seq_len={seq_len}")

    outputs = model(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; ensure output_hidden_states=True is supported.")

    final_logits = outputs.logits
    if final_logits.ndim != 3:
        raise RuntimeError(f"Unexpected logits shape: {tuple(final_logits.shape)}")
    final_diff = (final_logits[0, pos, int(token_a_id)] - final_logits[0, pos, int(token_b_id)]).detach()
    final_diff = final_diff.to(dtype=torch.float32, device=weight.device)

    final_norm = _get_final_norm(model, arch)
    apply_intermediate = bool(final_norm is not None and lens in ("auto", "final_norm"))
    apply_last = bool(final_norm is not None and lens == "final_norm")

    if final_norm is not None and lens == "auto":
        last_vec = hidden_states[-1][0, pos, :].detach()
        last_vec = last_vec.to(dtype=torch.float32, device=weight.device)
        d_raw, _, _ = _logit_diff_from_vec(
            last_vec,
            weight=weight.to(dtype=torch.float32),
            bias=bias.to(dtype=torch.float32) if bias is not None else None,
            token_a_id=int(token_a_id),
            token_b_id=int(token_b_id),
            compute_logits=False,
        )
        d_norm, _, _ = _logit_diff_from_vec(
            _apply_norm(final_norm, last_vec),
            weight=weight.to(dtype=torch.float32),
            bias=bias.to(dtype=torch.float32) if bias is not None else None,
            token_a_id=int(token_a_id),
            token_b_id=int(token_b_id),
            compute_logits=False,
        )
        tol = 5e-3
        if torch.isfinite(d_raw) and torch.isfinite(final_diff) and float((d_raw - final_diff).abs().item()) <= tol:
            apply_last = False
        elif torch.isfinite(d_norm) and torch.isfinite(final_diff) and float((d_norm - final_diff).abs().item()) <= tol:
            apply_last = True
        else:
            apply_last = False

    points = []
    prev = None
    weight_f32 = weight.to(dtype=torch.float32)
    bias_f32 = bias.to(dtype=torch.float32) if bias is not None else None

    for i, hs in enumerate(hidden_states):
        vec = hs[0, pos, :].detach().to(dtype=torch.float32, device=weight_f32.device)
        if final_norm is not None:
            if i == len(hidden_states) - 1:
                if apply_last:
                    vec = _apply_norm(final_norm, vec)
            else:
                if apply_intermediate:
                    vec = _apply_norm(final_norm, vec)

        diff, la, lb = _logit_diff_from_vec(
            vec,
            weight=weight_f32,
            bias=bias_f32,
            token_a_id=int(token_a_id),
            token_b_id=int(token_b_id),
            compute_logits=bool(compute_logits),
        )
        if prev is None:
            delta = diff
        else:
            delta = diff - prev
        prev = diff
        points.append(
            LogitLensPoint(
                state_index=int(i),
                block_index=int(i - 1) if i > 0 else None,
                logit_a=float(la.item()) if la is not None else float("nan"),
                logit_b=float(lb.item()) if lb is not None else float("nan"),
                logit_diff=float(diff.item()),
                delta_from_prev=float(delta.item()),
            )
        )

    return LogitLensTrace(
        arch=str(arch),
        lens=str(lens),
        position=int(position),
        token_a_id=int(token_a_id),
        token_b_id=int(token_b_id),
        apply_final_norm_intermediate=bool(apply_intermediate),
        apply_final_norm_last=bool(apply_last),
        final_logit_diff=float(final_diff.item()),
        points=tuple(points),
    )
