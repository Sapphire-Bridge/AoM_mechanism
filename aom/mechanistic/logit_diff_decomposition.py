from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class DecompositionComponent:
    name: str
    kind: str
    layer: Optional[int]
    head: Optional[int]
    contribution: float


@dataclass(frozen=True)
class LogitDiffDecomposition:
    components: tuple[DecompositionComponent, ...]
    sum_components: float
    constant: float
    predicted_logit_diff: float


def logit_diff_direction(model: Any, token_a_id: int, token_b_id: int) -> tuple[torch.Tensor, float]:
    """
    Return (direction, bias_diff) for logits(token_a) - logits(token_b).
    """
    a = int(token_a_id)
    b = int(token_b_id)
    if a == b:
        raise ValueError("token_a_id and token_b_id must differ")

    if hasattr(model, "W_U"):
        W_U = getattr(model, "W_U")
        if not torch.is_tensor(W_U) or W_U.ndim != 2:
            raise ValueError("model.W_U must be rank-2 tensor [d_model, vocab]")
        if max(a, b) >= int(W_U.shape[1]):
            raise ValueError(f"token id out of range for W_U vocab={int(W_U.shape[1])}")
        direction = W_U[:, a] - W_U[:, b]
        direction = direction.detach().to(dtype=torch.float32)
        b_u = getattr(model, "b_U", None)
        bias_diff = 0.0
        if torch.is_tensor(b_u) and b_u.ndim == 1 and max(a, b) < int(b_u.shape[0]):
            bias_diff = float((b_u[a] - b_u[b]).detach().to(dtype=torch.float32).item())
        return direction, bias_diff

    out_emb = None
    try:
        out_emb = model.get_output_embeddings()
    except Exception:
        out_emb = None
    if out_emb is None and hasattr(model, "lm_head"):
        out_emb = getattr(model, "lm_head")
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise ValueError("Could not resolve output embedding weights for logit-diff direction.")
    W = out_emb.weight
    if not torch.is_tensor(W) or W.ndim != 2:
        raise ValueError("output embedding weight must be rank-2 [vocab, d_model]")
    if max(a, b) >= int(W.shape[0]):
        raise ValueError(f"token id out of range for output embedding vocab={int(W.shape[0])}")
    direction = (W[a] - W[b]).detach().to(dtype=torch.float32)
    bias = getattr(out_emb, "bias", None)
    bias_diff = 0.0
    if torch.is_tensor(bias) and bias.ndim == 1 and max(a, b) < int(bias.shape[0]):
        bias_diff = float((bias[a] - bias[b]).detach().to(dtype=torch.float32).item())
    return direction, bias_diff


def _to_vec_at_pos(x: torch.Tensor, *, pos: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 [batch, pos, d], got shape={tuple(x.shape)}")
    if int(x.size(0)) < 1:
        raise ValueError("expected batch size >= 1")
    p = int(pos)
    if p < 0:
        p = int(x.size(1)) + p
    if p < 0 or p >= int(x.size(1)):
        raise ValueError(f"position out of range: {int(pos)} for seq_len={int(x.size(1))}")
    return x[0, p, :].detach().to(dtype=torch.float32)


def _to_head_mat_at_pos(x: torch.Tensor, *, pos: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 [batch, pos, head, d], got shape={tuple(x.shape)}")
    if int(x.size(0)) < 1:
        raise ValueError("expected batch size >= 1")
    p = int(pos)
    if p < 0:
        p = int(x.size(1)) + p
    if p < 0 or p >= int(x.size(1)):
        raise ValueError(f"position out of range: {int(pos)} for seq_len={int(x.size(1))}")
    return x[0, p, :, :].detach().to(dtype=torch.float32)


def _get_norm_weight_bias(norm: Any, d_model: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = getattr(norm, "w", None)
    if not torch.is_tensor(w):
        w = getattr(norm, "weight", None)
    if torch.is_tensor(w):
        weight = w.detach().to(dtype=torch.float32)
    else:
        weight = torch.ones(d_model, dtype=torch.float32)

    b = getattr(norm, "b", None)
    if not torch.is_tensor(b):
        b = getattr(norm, "bias", None)
    if torch.is_tensor(b):
        bias = b.detach().to(dtype=torch.float32)
    else:
        bias = torch.zeros(d_model, dtype=torch.float32)

    if weight.ndim != 1 or int(weight.numel()) != int(d_model):
        raise ValueError(f"final norm weight shape mismatch: expected ({d_model},), got {tuple(weight.shape)}")
    if bias.ndim != 1 or int(bias.numel()) != int(d_model):
        raise ValueError(f"final norm bias shape mismatch: expected ({d_model},), got {tuple(bias.shape)}")
    return weight, bias


def _final_norm_affine_params(
    *,
    final_norm: Optional[Any],
    total_resid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (scale, const) such that y_fixed = scale * x + const.

    This uses total_resid statistics (mu/sigma or rms) as fixed constants.
    """
    x = total_resid.detach().to(dtype=torch.float32)
    d_model = int(x.numel())
    if final_norm is None:
        return torch.ones_like(x), torch.zeros_like(x)

    weight, bias = _get_norm_weight_bias(final_norm, d_model)
    weight = weight.to(device=x.device, dtype=x.dtype)
    bias = bias.to(device=x.device, dtype=x.dtype)
    eps = float(getattr(final_norm, "eps", 1e-5))
    name = type(final_norm).__name__.lower()
    is_rms = "rms" in name

    if is_rms:
        rms = torch.sqrt(torch.mean(x.pow(2)) + float(eps))
        scale = weight / rms
        const = bias
        return scale, const

    mu = torch.mean(x)
    var = torch.mean((x - mu).pow(2))
    std = torch.sqrt(var + float(eps))
    scale = weight / std
    const = bias - scale * mu
    return scale, const


def _sorted_layer_keys(cache: Mapping[str, Any], *, prefix: str) -> List[tuple[int, str]]:
    out: List[tuple[int, str]] = []
    for key in cache.keys():
        k = str(key)
        if not k.startswith(prefix):
            continue
        try:
            idx = int(k.rsplit(".", 1)[1])
        except Exception:
            continue
        out.append((idx, k))
    out.sort(key=lambda t: t[0])
    return out


def decompose_logit_diff(
    cache: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    pos: int = -1,
    mode: str = "layer",
    final_norm: Optional[Any] = None,
    bias_diff: float = 0.0,
) -> LogitDiffDecomposition:
    if mode not in {"layer", "head"}:
        raise ValueError("mode must be one of: layer, head")
    d = direction.detach().to(dtype=torch.float32)
    d_model = int(d.numel())

    resid_final_t = cache.get("resid_final", None)
    if not torch.is_tensor(resid_final_t):
        raise ValueError("cache missing tensor key 'resid_final'")
    total_resid = _to_vec_at_pos(resid_final_t, pos=pos)
    if int(total_resid.numel()) != d_model:
        raise ValueError(
            f"direction dim mismatch with resid_final: direction={d_model} resid_final={int(total_resid.numel())}"
        )
    d = d.to(device=total_resid.device, dtype=total_resid.dtype)
    scale, const = _final_norm_affine_params(final_norm=final_norm, total_resid=total_resid)

    comps: List[DecompositionComponent] = []

    def _add_component(name: str, vec: torch.Tensor, *, kind: str, layer: Optional[int], head: Optional[int]) -> None:
        if int(vec.numel()) != d_model:
            return
        vec = vec.to(device=scale.device, dtype=torch.float32)
        contrib = float(torch.dot(vec * scale, d).item())
        comps.append(
            DecompositionComponent(
                name=str(name),
                kind=str(kind),
                layer=None if layer is None else int(layer),
                head=None if head is None else int(head),
                contribution=float(contrib),
            )
        )

    if torch.is_tensor(cache.get("embed", None)):
        _add_component("embed", _to_vec_at_pos(cache["embed"], pos=pos), kind="embed", layer=None, head=None)
    if torch.is_tensor(cache.get("pos_embed", None)):
        _add_component("pos_embed", _to_vec_at_pos(cache["pos_embed"], pos=pos), kind="pos_embed", layer=None, head=None)

    if mode == "layer":
        for layer, key in _sorted_layer_keys(cache, prefix="attn_out."):
            x = cache.get(key, None)
            if torch.is_tensor(x):
                _add_component(f"attn_out.{int(layer)}", _to_vec_at_pos(x, pos=pos), kind="attn_out", layer=layer, head=None)
        for layer, key in _sorted_layer_keys(cache, prefix="mlp_out."):
            x = cache.get(key, None)
            if torch.is_tensor(x):
                _add_component(f"mlp_out.{int(layer)}", _to_vec_at_pos(x, pos=pos), kind="mlp_out", layer=layer, head=None)
    else:
        for layer, key in _sorted_layer_keys(cache, prefix="head_result."):
            x = cache.get(key, None)
            if not torch.is_tensor(x):
                continue
            try:
                mat = _to_head_mat_at_pos(x, pos=pos)  # [head, d]
            except Exception:
                continue
            if mat.ndim != 2 or int(mat.size(1)) != d_model:
                continue
            for h in range(int(mat.size(0))):
                _add_component(
                    f"head_result.{int(layer)}.{int(h)}",
                    mat[h, :],
                    kind="head_result",
                    layer=int(layer),
                    head=int(h),
                )

    sum_components = float(sum(float(c.contribution) for c in comps))
    constant = float(torch.dot(const.to(device=d.device, dtype=torch.float32), d).item() + float(bias_diff))
    predicted = float(sum_components + constant)
    return LogitDiffDecomposition(
        components=tuple(comps),
        sum_components=float(sum_components),
        constant=float(constant),
        predicted_logit_diff=float(predicted),
    )


def validate_decomposition(*, predicted_logit_diff: float, actual_logit_diff: float, tol: float = 1e-3) -> Dict[str, Any]:
    pred = float(predicted_logit_diff)
    act = float(actual_logit_diff)
    abs_err = float(abs(pred - act))
    return {
        "predicted_logit_diff": pred,
        "actual_logit_diff": act,
        "abs_error": abs_err,
        "within_tol": bool(abs_err <= float(tol)),
        "tol": float(tol),
    }
