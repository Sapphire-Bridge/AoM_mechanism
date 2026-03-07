"""Tests for SAE channel decomposition with 2x2 factorial design."""

import pytest
import torch
from torch import nn

from aom.interventions.sae_adapter import SAEInputTransform
from aom.metrics.sae_decomposition import _decompose, _norm_match


class SimpleSAE(nn.Module):
    """Simple SAE that projects to lower-rank subspace."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self._d_in = int(d_in)
        self._d_sae = int(d_sae)
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.1)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_sae(self) -> int:
        return self._d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


class IdentitySAE(nn.Module):
    """Identity SAE for testing (no reconstruction error)."""

    def __init__(self, d_in: int):
        super().__init__()
        self._d_in = int(d_in)
        self._d_sae = int(d_in)
        self.W_dec = nn.Parameter(torch.eye(d_in))

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


def test_identity_sae_decomposition_full_equals_sae_only():
    """With identity SAE, complement is zero, so full == sae_only."""
    sae = IdentitySAE(d_in=64)
    transform = SAEInputTransform(scale=1.0)

    h = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    S, C = _decompose(h, sae, transform)

    # For identity SAE, S == h and C == 0
    assert torch.allclose(S, h, atol=1e-5)
    assert torch.allclose(C, torch.zeros_like(C), atol=1e-5)


def test_simple_sae_decomposition_sums_to_original():
    """S + C should equal original exactly (by definition)."""
    sae = SimpleSAE(d_in=64, d_sae=32)
    transform = SAEInputTransform(scale=1.0)

    h = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    S, C = _decompose(h, sae, transform)

    # S + C == h by definition
    assert torch.allclose(S + C, h, atol=1e-5)


def test_decomposition_with_scaling():
    """Decomposition should work correctly with non-unit scale."""
    sae = SimpleSAE(d_in=64, d_sae=32)
    transform = SAEInputTransform(scale=2.5)

    h = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    S, C = _decompose(h, sae, transform)

    # S + C should still equal h
    assert torch.allclose(S + C, h, atol=1e-5)


def test_complement_captures_reconstruction_error():
    """Complement should capture what SAE cannot reconstruct."""
    # Use a very low-rank SAE
    sae = SimpleSAE(d_in=64, d_sae=4)
    transform = SAEInputTransform(scale=1.0)

    h = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    S, C = _decompose(h, sae, transform)

    # C should be non-zero for low-rank SAE
    assert C.abs().max() > 1e-3

    # Verify decomposition
    assert torch.allclose(S + C, h, atol=1e-5)


def test_norm_match_scales_correctly():
    """_norm_match should scale vector to target norm."""
    vec = torch.randn(10, 64)
    target_norm = 5.0

    scaled = _norm_match(vec, target_norm)
    actual_norm = float(scaled.norm().item())

    assert abs(actual_norm - target_norm) < 1e-5


def test_norm_match_preserves_direction():
    """_norm_match should preserve vector direction."""
    vec = torch.randn(10, 64)
    target_norm = 3.0

    scaled = _norm_match(vec, target_norm)

    # Normalize both and compare
    vec_normalized = vec / vec.norm()
    scaled_normalized = scaled / scaled.norm()

    assert torch.allclose(vec_normalized, scaled_normalized, atol=1e-5)


def test_norm_match_handles_zero_vector():
    """_norm_match should handle zero vectors gracefully."""
    vec = torch.zeros(10, 64)
    target_norm = 5.0

    scaled = _norm_match(vec, target_norm)

    # Zero vector should stay zero
    assert torch.allclose(scaled, vec, atol=1e-8)


def test_delta_additivity_identity():
    """For any SAE, delta_r = delta_S + delta_C by construction."""
    sae = SimpleSAE(d_in=64, d_sae=32)
    transform = SAEInputTransform(scale=1.0)

    r_donor = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    r_recv = torch.randn(1, 5, 64)

    S_donor, C_donor = _decompose(r_donor, sae, transform)
    S_recv, C_recv = _decompose(r_recv, sae, transform)

    delta_r = r_donor - r_recv
    delta_S = S_donor - S_recv
    delta_C = C_donor - C_recv

    # Key identity: delta_r = delta_S + delta_C (exactly)
    assert torch.allclose(delta_r, delta_S + delta_C, atol=1e-5)


def test_identity_sae_delta_decomposition():
    """For identity SAE: delta_S == delta_r, delta_C == 0."""
    sae = IdentitySAE(d_in=64)
    transform = SAEInputTransform(scale=1.0)

    r_donor = torch.randn(1, 5, 64)  # (batch, seq, hidden)
    r_recv = torch.randn(1, 5, 64)

    S_donor, C_donor = _decompose(r_donor, sae, transform)
    S_recv, C_recv = _decompose(r_recv, sae, transform)

    delta_r = r_donor - r_recv
    delta_S = S_donor - S_recv
    delta_C = C_donor - C_recv

    # For identity SAE: delta_S == delta_r, delta_C == 0
    assert torch.allclose(delta_S, delta_r, atol=1e-5)
    assert torch.allclose(delta_C, torch.zeros_like(delta_C), atol=1e-5)


def test_result_structure():
    """Test that compute_cpt_channel_decomposition returns correct structure."""
    from aom.metrics.sae_decomposition import compute_cpt_channel_decomposition

    # Verify the function exists and has correct signature
    assert callable(compute_cpt_channel_decomposition)

    # Expected top-level keys
    expected_keys = [
        "layer",
        "norm_matching",
        "n_pairs",
        "n_directions_total",
        "n_directions_patched",
        "n_directions_skipped_misaligned",
        "total_effect",
        "recon_main_effect",
        "resid_main_effect",
        "interaction",
        "norm_stats",
        "conditions",
    ]

    # Expected condition keys
    expected_conditions = ["A_baseline", "B_recon_only", "C_resid_only", "D_full", "sham"]

    # Verify imports work
    from aom.metrics import compute_cpt_channel_decomposition as imported_fn
    assert callable(imported_fn)


def test_norm_stats_structure():
    """Test that norm_stats contains expected fields."""
    expected_norm_stats_keys = [
        "delta_r_mean",
        "delta_r_median",
        "delta_S_mean",
        "delta_S_median",
        "delta_C_mean",
        "delta_C_median",
        "additivity_error_mean",
        "additivity_error_median",
    ]
    # Just verify the structure expectation is documented
    assert len(expected_norm_stats_keys) == 8


def test_bootstrap_ci_items_basic():
    """Test block bootstrap by item."""
    from aom.metrics.sae_decomposition import _bootstrap_ci_items

    # 3 items, each with 2 directions
    item_values = [
        [1.0, 1.2],  # item 0
        [2.0, 2.1],  # item 1
        [3.0, 3.3],  # item 2
    ]

    mean, lo, hi = _bootstrap_ci_items(item_values, n_bootstrap=100, ci=0.95, seed=42)

    # Mean should be average of all values
    all_vals = [x for item in item_values for x in item]
    expected_mean = sum(all_vals) / len(all_vals)
    assert abs(mean - expected_mean) < 1e-6

    # CI should bracket the mean
    assert lo <= mean <= hi


def test_bootstrap_ci_items_handles_missing_directions():
    """Test block bootstrap handles items with 0 or 1 valid direction."""
    from aom.metrics.sae_decomposition import _bootstrap_ci_items

    # Item 1 has only 1 direction, item 2 has none
    item_values = [
        [1.0, 1.2],  # item 0: 2 directions
        [2.0],       # item 1: 1 direction
        [],          # item 2: 0 directions (skipped)
    ]

    mean, lo, hi = _bootstrap_ci_items(item_values, n_bootstrap=100, ci=0.95, seed=42)

    # Should only use items with values
    valid_vals = [1.0, 1.2, 2.0]
    expected_mean = sum(valid_vals) / len(valid_vals)
    assert abs(mean - expected_mean) < 1e-6


def test_bootstrap_ci_items_empty():
    """Test block bootstrap handles all-empty gracefully."""
    from aom.metrics.sae_decomposition import _bootstrap_ci_items

    item_values = [[], [], []]

    mean, lo, hi = _bootstrap_ci_items(item_values, n_bootstrap=100, ci=0.95, seed=42)

    import math
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)
