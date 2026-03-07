from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def validate_attention_weights(attn_weights: torch.Tensor, seq_len: int) -> None:
    if attn_weights.ndim != 4:
        raise ValueError(f"attn_weights must be rank-4 (got shape={tuple(attn_weights.shape)})")
    if attn_weights.shape[-2] != seq_len or attn_weights.shape[-1] != seq_len:
        raise ValueError(
            "attn_weights must be square with seq_len keys/queries "
            f"(got shape={tuple(attn_weights.shape)} seq_len={seq_len})"
        )
    if not torch.isfinite(attn_weights).all():
        raise ValueError("attn_weights contain non-finite values")
    min_val = float(attn_weights.min().item())
    if min_val < -1e-6:
        raise ValueError(f"attn_weights contain negative probabilities (min={min_val})")
    sums = attn_weights.sum(dim=-1)
    ones = torch.ones_like(sums)
    if not torch.allclose(sums, ones, atol=1e-2, rtol=1e-2):
        raise ValueError("attn_weights rows do not sum to 1 within tolerance")


def calculate_induction_score(attn_weights: torch.Tensor, base_len: int) -> torch.Tensor:
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    seq_len = int(attn_weights.shape[-1])
    expected = int(base_len) * 2
    if seq_len != expected:
        raise ValueError(f"expected seq_len={expected} for repeats=2 (got seq_len={seq_len})")

    attn_slice = attn_weights[..., base_len:, :base_len]
    induction_diag = attn_slice.diagonal(offset=1, dim1=-2, dim2=-1)
    return induction_diag.mean(dim=-1)


def calculate_induction_control(attn_weights: torch.Tensor, base_len: int, mode: str) -> torch.Tensor:
    if mode != "offset0":
        raise ValueError(f"unsupported control mode: {mode!r}")
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    seq_len = int(attn_weights.shape[-1])
    expected = int(base_len) * 2
    if seq_len != expected:
        raise ValueError(f"expected seq_len={expected} for repeats=2 (got seq_len={seq_len})")

    attn_slice = attn_weights[..., base_len:, :base_len]
    baseline_diag = attn_slice.diagonal(offset=0, dim1=-2, dim2=-1)[..., :-1]
    return baseline_diag.mean(dim=-1)


def calculate_first_half_mass(attn_weights: torch.Tensor, base_len: int) -> torch.Tensor:
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    seq_len = int(attn_weights.shape[-1])
    expected = int(base_len) * 2
    if seq_len != expected:
        raise ValueError(f"expected seq_len={expected} for repeats=2 (got seq_len={seq_len})")

    attn_slice = attn_weights[..., base_len:, :base_len]
    mass = attn_slice.sum(dim=-1)[..., :-1]
    return mass.mean(dim=-1)


def calculate_induction_diag_fraction(attn_weights: torch.Tensor, base_len: int, eps: float = 1e-8) -> torch.Tensor:
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    seq_len = int(attn_weights.shape[-1])
    expected = int(base_len) * 2
    if seq_len != expected:
        raise ValueError(f"expected seq_len={expected} for repeats=2 (got seq_len={seq_len})")

    attn_slice = attn_weights[..., base_len:, :base_len]
    induction_diag = attn_slice.diagonal(offset=1, dim1=-2, dim2=-1)
    mass = attn_slice.sum(dim=-1)[..., :-1]
    frac = induction_diag / (mass + float(eps))
    return frac.mean(dim=-1)


def expected_uniform(base_len: int, repeats: int = 2) -> float:
    if base_len < 1:
        raise ValueError("base_len must be >= 1")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    return float(1.0 / float(base_len * repeats))


def bootstrap_ci_matrix(
    samples: np.ndarray, *, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap CI for per-head means.

    samples: [n_samples, n_heads]
    Returns (mean, ci_low, ci_high) arrays of shape [n_heads].
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")

    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("samples must be rank-2 [n_samples, n_heads]")
    n = arr.shape[0]
    if n == 0:
        nan = np.full(arr.shape[1], np.nan, dtype=float)
        return nan, nan, nan
    if not np.isfinite(arr).all():
        nan = np.full(arr.shape[1], np.nan, dtype=float)
        return nan, nan, nan

    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, n, size=(int(n_bootstrap), n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - float(ci)
    lo = np.percentile(means, 100.0 * (alpha / 2.0), axis=0)
    hi = np.percentile(means, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return arr.mean(axis=0), lo, hi
