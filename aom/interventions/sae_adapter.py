"""
SAE Adapter for AoM patching system.

This module provides a clean interface for SAE-based interventions that integrates
with the existing AoM activation patching infrastructure. It addresses common pitfalls:

1. Correctness-first delta injection (preserves SAE residual / reconstruction error)
2. Explicit input transforms (prevents silent invalidation from mis-scaled streams)
3. Stateful normalization guards (warns if normalization is stateful)
4. Clean, HF-compatible hookpoints for feature manipulation
5. Random mask controls at matched sparsity

Design Philosophy:
- Prioritize correctness and defensibility for research use
- Support both encode/decode and direct feature manipulation
- Compatible with both TransformerLens and HuggingFace models
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Tuple

import torch
from torch import nn


class SAEProtocol(Protocol):
    """Protocol for SAE compatibility."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to features. Returns features (batch, seq, n_features)."""
        ...

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to activations. Returns reconstructed (batch, seq, hidden_dim)."""
        ...

    @property
    def d_in(self) -> int:
        """Input dimension (model hidden_dim)."""
        ...

    @property
    def d_sae(self) -> int:
        """SAE feature dimension."""
        ...


@dataclass(frozen=True)
class SAEInputTransform:
    """
    Map HF activations into the distribution the SAE was trained on.

    This is intentionally minimal (scale + optional shift). If your SAE uses a more complex
    preprocessing scheme, encapsulate it here to avoid silent invalidation.
    """

    scale: float = 1.0
    shift: float | None = None
 
    def __post_init__(self) -> None:
        if float(self.scale) == 0.0:
            raise ValueError("scale must be non-zero")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * float(self.scale)
        if self.shift is not None:
            y = y + float(self.shift)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Invert the forward transform.

        If forward is: y = x * scale + shift
        then inverse is: x = (y - shift) / scale
        """
        x = y
        if self.shift is not None:
            x = x - float(self.shift)
        return x / float(self.scale)

    def inverse_delta(self, delta: torch.Tensor) -> torch.Tensor:
        # Only invert multiplicative scaling for deltas; additive shifts cancel in deltas.
        return delta / float(self.scale)


@dataclass
class SAEHookState:
    total_active: float = 0.0
    total_preserved: float = 0.0

    def reset(self) -> None:
        self.total_active = 0.0
        self.total_preserved = 0.0

    def sparsity_gain(self) -> float:
        if self.total_active <= 0.0:
            return 0.0
        return 1.0 - (self.total_preserved / self.total_active)


class FeaturePolicy(Protocol):
    """
    Policy that edits SAE features in-place, restricted to a site mask.

    features: (batch, seq, d_sae)
    site_mask: (batch, seq, 1) boolean mask marking where intervention applies
    token_mask: optional (batch, seq) boolean mask for valid tokens (padding exclusion)
    """

    def apply(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class IdentityFeaturePolicy:
    def apply(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = site_mask
        _ = token_mask
        return features


@dataclass(frozen=True)
class RelativeThresholdPolicy:
    """
    Zero out low-salience features relative to a context-dependent scale.

    Typical usage: compute a per-feature scale across the sequence (max or quantile),
    and set features with |f| <= t * scale to zero at the intervention site.
    """

    threshold: float
    scale_mode: Literal["max", "quantile", "global"] = "quantile"
    quantile: float = 0.95
    global_scale: Optional[torch.Tensor] = None  # (d_sae,)
    min_scale: float = 1e-6
    use_abs: bool = True
    keep: Literal["high", "low"] = "high"  # anti-control: keep="low" removes top activations

    def apply(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, seq, d_sae)")
        if site_mask.ndim != 3 or site_mask.size(-1) != 1:
            raise ValueError("site_mask must have shape (batch, seq, 1)")
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        if self.min_scale <= 0.0:
            raise ValueError("min_scale must be > 0")
        if self.keep not in {"high", "low"}:
            raise ValueError("keep must be 'high' or 'low'")

        feats = features.abs() if self.use_abs else features

        valid_tokens: Optional[torch.Tensor] = None
        if token_mask is not None:
            if token_mask.ndim == 3 and token_mask.size(-1) == 1:
                token_mask = token_mask.squeeze(-1)
            if token_mask.ndim != 2:
                raise ValueError("token_mask must have shape (batch, seq)")
            valid_tokens = token_mask.to(device=features.device, dtype=torch.bool)

        if self.scale_mode == "global":
            if self.global_scale is None:
                raise ValueError("global_scale must be provided when scale_mode='global'")
            scale = self.global_scale.to(device=features.device, dtype=features.dtype)
            if scale.ndim != 1 or scale.numel() != features.size(-1):
                raise ValueError("global_scale must have shape (d_sae,)")
            scale = scale.view(1, 1, -1)
        elif self.scale_mode == "max":
            if valid_tokens is None:
                scale = feats.max(dim=1, keepdim=True).values
            else:
                masked = feats.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
                scale = masked.max(dim=1, keepdim=True).values
        elif self.scale_mode == "quantile":
            if not (0.0 < self.quantile < 1.0):
                raise ValueError("quantile must be in (0, 1)")
            # Batch loop keeps this robust under variable-length masks.
            batch_size, _seq_len, d_sae = feats.shape
            scale_list = []
            for b in range(batch_size):
                if valid_tokens is None:
                    vals = feats[b]  # (seq, d_sae)
                else:
                    idx = valid_tokens[b].nonzero(as_tuple=False).squeeze(-1)
                    if idx.numel() < 1:
                        scale_list.append(torch.ones((d_sae,), device=feats.device, dtype=feats.dtype))
                        continue
                    vals = feats[b, idx, :]
                q = torch.quantile(vals.to(dtype=torch.float32), float(self.quantile), dim=0)
                scale_list.append(q.to(dtype=feats.dtype))
            scale = torch.stack(scale_list, dim=0).view(batch_size, 1, d_sae)
        else:
            raise ValueError(f"unknown scale_mode={self.scale_mode!r}")

        scale = torch.clamp(scale, min=float(self.min_scale))
        thresh = scale * float(self.threshold)
        keep_mask = feats > thresh
        if self.keep == "low":
            keep_mask = ~keep_mask

        # Only apply policy within site_mask; outside the site we return features unchanged.
        keep_mask = torch.where(site_mask.expand_as(keep_mask), keep_mask, torch.ones_like(keep_mask))
        patched = features.clone()
        patched[~keep_mask] = 0.0
        return patched


@dataclass(frozen=True)
class RandomMatchedActiveMaskPolicy:
    """
    Control policy: randomly remove the same number of *active* features as a base policy.

    This is useful for "structure beats random" controls at matched sparsity.
    """

    base_policy: FeaturePolicy
    random_seed: int = 42
    eps_active: float = 1e-6

    def apply(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = self.base_policy.apply(features, site_mask=site_mask, token_mask=token_mask)
        if self.eps_active < 0.0:
            raise ValueError("eps_active must be >= 0")

        active = features.abs() > float(self.eps_active)
        removed_target = active & ~(target.abs() > float(self.eps_active))
        if site_mask.ndim == 2:
            site_mask = site_mask.unsqueeze(-1)
        region = site_mask.to(device=features.device, dtype=torch.bool).expand_as(active)
        removed_counts = (removed_target & region).sum(dim=-1)  # (B, S)

        patched = features.clone()
        rng = torch.Generator(device=features.device).manual_seed(int(self.random_seed))
        batch_size, seq_len, d_sae = features.shape
        for b in range(batch_size):
            for s in range(seq_len):
                if not bool(site_mask[b, s, 0]):
                    continue
                k = int(removed_counts[b, s].item())
                if k <= 0:
                    continue
                active_idx = active[b, s].nonzero(as_tuple=False).squeeze(-1)
                if active_idx.numel() < 1:
                    continue
                if k > int(active_idx.numel()):
                    k = int(active_idx.numel())
                perm = active_idx[torch.randperm(active_idx.numel(), generator=rng, device=features.device)[:k]]
                patched[b, s, perm] = 0.0
        return patched


@dataclass(frozen=True)
class SAEPatchSite:
    """
    Patch site for SAE-based interventions.

    Specifies where in the model to apply SAE and which features to manipulate.
    """

    layer: int
    token_indices: Tuple[int, ...]
    feature_mask: Optional[torch.Tensor] = None  # Boolean mask of features to ablate
    feature_values: Optional[torch.Tensor] = None  # Override feature values


@dataclass(frozen=True)
class SAEPatchConfig:
    """
    Configuration for SAE patching strategy.

    Args:
        decode_strategy: 'safe_2decode' (correctness-first, decode clean and patched separately)
                        or 'delta_1decode' (efficient but requires careful norm handling)
        check_normalization_stateful: If True, warn if SAE appears to use stateful normalization
        random_seed: Seed for random mask controls
    """

    decode_strategy: Literal["safe_2decode", "delta_1decode"] = "safe_2decode"
    check_normalization_stateful: bool = True
    random_seed: int = 42
    eps_active: float = 1e-6
    dtype_policy: Literal["sae", "model"] = "sae"
    roundtrip_replace: bool = False

    def __post_init__(self):
        if self.decode_strategy not in ("safe_2decode", "delta_1decode"):
            raise ValueError(f"Unknown decode_strategy: {self.decode_strategy}")
        if self.eps_active < 0.0:
            raise ValueError("eps_active must be >= 0")
        if self.dtype_policy not in ("sae", "model"):
            raise ValueError(f"Unknown dtype_policy: {self.dtype_policy}")


class SAEAdapter:
    """
    Adapter for SAE-based interventions in AoM patching system.

    This adapter provides:
    1. Clean hookpoints for SAE encode/decode
    2. Safe 2-decode strategy (decode clean and patched separately)
    3. Guards for stateful normalization
    4. Random mask control generation

    Example:
        >>> sae = load_your_sae(...)  # Must have encode/decode methods
        >>> adapter = SAEAdapter(sae, config=SAEPatchConfig())
        >>>
        >>> # Create a feature ablation mask
        >>> feature_mask = torch.zeros(sae.d_sae, dtype=torch.bool)
        >>> feature_mask[important_features] = True
        >>>
        >>> # Patch site
        >>> site = SAEPatchSite(
        ...     layer=10,
        ...     token_indices=(5,),
        ...     feature_mask=feature_mask
        ... )
        >>>
        >>> # Get reconstructed activations with ablation
        >>> recon_ablated = adapter.reconstruct_with_intervention(
        ...     hidden_states, site
        ... )
    """

    def __init__(self, sae: SAEProtocol, config: Optional[SAEPatchConfig] = None):
        self.sae = sae
        self.config = config or SAEPatchConfig()
        self._normalization_warned = False

        if self.config.check_normalization_stateful:
            self._check_sae_normalization()

    def _check_sae_normalization(self):
        """
        Check if SAE uses stateful normalization (e.g., running stats).

        This is important because 2-decode strategy assumes decode() is stateless.
        If the SAE has BatchNorm or similar with running stats, decoding the same
        features twice could give different results.
        """
        for name, module in self.sae.named_modules() if hasattr(self.sae, "named_modules") else []:  # type: ignore
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.track_running_stats:
                    warnings.warn(
                        f"SAE contains BatchNorm layer '{name}' with track_running_stats=True. "
                        "This may cause issues with 2-decode strategy if running stats are updated. "
                        "Consider using LayerNorm or setting track_running_stats=False.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._normalization_warned = True

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode hidden states to SAE features.

        Args:
            hidden_states: (batch, seq, hidden_dim)

        Returns:
            features: (batch, seq, d_sae)
        """
        return self.sae.encode(hidden_states)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode SAE features to reconstructed hidden states.

        Args:
            features: (batch, seq, d_sae)

        Returns:
            reconstructed: (batch, seq, hidden_dim)
        """
        return self.sae.decode(features)

    def reconstruct_with_intervention(
        self,
        hidden_states: torch.Tensor,
        site: SAEPatchSite,
        donor_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct hidden states with SAE intervention.

        This implements the SAFE 2-decode strategy:
        1. Encode clean activations to features
        2. Apply intervention (ablation, feature steering, etc.)
        3. Decode intervened features to reconstructed activations

        This avoids issues with bias/norm that can arise from computing deltas.

        Args:
            hidden_states: Clean activations (batch, seq, hidden_dim)
            site: Patch site specifying tokens and features to intervene on
            donor_hidden_states: Optional donor activations for feature replacement

        Returns:
            reconstructed: (batch, seq, hidden_dim) with intervention applied
        """
        # Encode clean activations
        features = self.encode(hidden_states)  # (batch, seq, d_sae)

        # Apply intervention to features
        features_intervened = features.clone()

        # Apply feature mask (ablation)
        if site.feature_mask is not None:
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, site.feature_mask] = 0.0

        # Apply feature value overrides
        if site.feature_values is not None:
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, :] = site.feature_values

        # If donor provided, replace features from donor
        if donor_hidden_states is not None:
            donor_features = self.encode(donor_hidden_states)
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, :] = donor_features[:, token_idx, :]

        # Decode intervened features (2-decode strategy: decode ONCE with intervened features)
        reconstructed = self.decode(features_intervened)

        return reconstructed

    def compute_delta_loss(
        self,
        hidden_states: torch.Tensor,
        site: SAEPatchSite,
        donor_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction delta loss (L2 norm of activation change).

        This is the "delta" between clean and intervened reconstructions.
        NOTE: This is NOT KL divergence - it's L2 reconstruction error.

        Args:
            hidden_states: Clean activations
            site: Patch site
            donor_hidden_states: Optional donor activations

        Returns:
            delta_loss: Scalar L2 norm of (recon_intervened - recon_clean)
        """
        # Encode and decode clean (baseline reconstruction)
        features_clean = self.encode(hidden_states)
        recon_clean = self.decode(features_clean)

        # Reconstruct with intervention
        recon_intervened = self.reconstruct_with_intervention(
            hidden_states, site, donor_hidden_states
        )

        # Compute L2 delta
        delta = recon_intervened - recon_clean
        delta_loss = torch.norm(delta, p=2)

        return delta_loss

    def generate_random_mask_control(
        self, target_features: torch.Tensor, sparsity: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate random feature mask at matched sparsity.

        This is a critical control: randomly mask features at the same sparsity
        as the target intervention to check if effects are specific to chosen features.

        Args:
            target_features: Features from actual intervention (batch, seq, d_sae)
            sparsity: If provided, use this sparsity. Otherwise, match target sparsity.

        Returns:
            random_mask: Boolean mask (d_sae,) with matched sparsity
        """
        if sparsity is None:
            # Compute sparsity from target features
            # Sparsity = fraction of features that are non-zero
            active_features = (target_features.abs() > 1e-6).float().mean()
            sparsity = float(active_features.item())

        # Generate random mask at matched sparsity
        n_features = self.sae.d_sae
        n_active = int(sparsity * n_features)

        rng = torch.Generator().manual_seed(self.config.random_seed)
        indices = torch.randperm(n_features, generator=rng)[:n_active]

        random_mask = torch.zeros(n_features, dtype=torch.bool)
        random_mask[indices] = True

        return random_mask


def _infer_sae_device_dtype(
    sae: SAEProtocol,
    *,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    if isinstance(sae, nn.Module):
        p = next(sae.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    W_dec = getattr(sae, "W_dec", None)
    if isinstance(W_dec, torch.Tensor):
        return W_dec.device, W_dec.dtype
    return fallback_device, fallback_dtype


def _get_W_dec(sae: SAEProtocol, *, d_sae: int, d_in: int) -> torch.Tensor:
    W_dec = getattr(sae, "W_dec", None)
    if not isinstance(W_dec, torch.Tensor):
        raise AttributeError("SAE does not expose a tensor attribute 'W_dec' required for delta_1decode")
    if W_dec.ndim != 2:
        raise ValueError("SAE.W_dec must be rank-2")
    if tuple(W_dec.shape) == (d_sae, d_in):
        return W_dec
    if tuple(W_dec.shape) == (d_in, d_sae):
        return W_dec.t()
    raise ValueError(f"SAE.W_dec has unexpected shape {tuple(W_dec.shape)} (expected {(d_sae, d_in)} or {(d_in, d_sae)})")


class SAEInterventionHook:
    """
    HF forward-hook compatible SAE patcher.

    Implements:
    - input transform (scaling) before SAE encode
    - feature policy application restricted by site_mask
    - delta injection (adds decoded feature delta back into original hidden states)
    """

    def __init__(
        self,
        *,
        sae: SAEProtocol,
        token_indices: Optional[Tuple[int, ...]],
        policy: FeaturePolicy,
        transform: Optional[SAEInputTransform] = None,
        config: Optional[SAEPatchConfig] = None,
        token_mask: Optional[torch.Tensor] = None,
        state: Optional[SAEHookState] = None,
    ) -> None:
        self.sae = sae
        self.token_indices = token_indices
        self.policy = policy
        self.transform = transform or SAEInputTransform()
        self.config = config or SAEPatchConfig()
        self.token_mask = token_mask
        self.state = state or SAEHookState()

    def __call__(self, module: nn.Module, inputs: Tuple, output: Tuple | torch.Tensor):
        _ = module
        _ = inputs

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("expected hidden states tensor of shape (batch, seq, hidden_dim)")

        batch_size, seq_len, hidden_dim = hidden.shape
        if hasattr(self.sae, "d_in") and int(getattr(self.sae, "d_in")) != int(hidden_dim):
            raise ValueError(f"SAE d_in={int(getattr(self.sae, 'd_in'))} does not match hidden_dim={int(hidden_dim)}")

        if self.token_indices is None:
            site_mask = torch.ones((batch_size, seq_len, 1), device=hidden.device, dtype=torch.bool)
        else:
            if len(self.token_indices) < 1:
                raise ValueError("token_indices must be non-empty (or None for full sequence)")
            site_mask = torch.zeros((batch_size, seq_len, 1), device=hidden.device, dtype=torch.bool)
            for token_idx in self.token_indices:
                if token_idx < 0 or token_idx >= seq_len:
                    raise ValueError(f"token index {token_idx} out of range [0, {seq_len})")
                site_mask[:, int(token_idx), 0] = True

        sae_device, sae_dtype = _infer_sae_device_dtype(
            self.sae, fallback_device=hidden.device, fallback_dtype=torch.float32
        )
        compute_dtype = sae_dtype
        if self.config.dtype_policy == "model":
            if sae_dtype != hidden.dtype:
                warnings.warn(
                    f"dtype_policy='model' requested but SAE params are {sae_dtype} while model stream is {hidden.dtype}; "
                    "using SAE dtype to avoid dtype mismatch in encode/decode.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                compute_dtype = hidden.dtype

        hidden_sae = hidden.to(device=sae_device, dtype=compute_dtype)
        x_in = self.transform.forward(hidden_sae)
        features = self.sae.encode(x_in)
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("SAE.encode must return a tensor of shape (batch, seq, d_sae)")
        if hasattr(self.sae, "d_sae") and int(getattr(self.sae, "d_sae")) != int(features.size(-1)):
            raise ValueError(f"SAE d_sae={int(getattr(self.sae, 'd_sae'))} does not match features.size(-1)={int(features.size(-1))}")

        token_mask = self.token_mask
        if token_mask is not None:
            token_mask = token_mask.to(device=features.device, dtype=torch.bool)
            if token_mask.ndim == 3 and token_mask.size(-1) == 1:
                token_mask = token_mask.squeeze(-1)
            if token_mask.shape != (batch_size, seq_len):
                raise ValueError("token_mask must have shape (batch, seq)")

        site_mask_sae = site_mask.to(device=features.device, dtype=torch.bool)
        if self.config.roundtrip_replace:
            # Sterility/roundtrip mode: do not modify features; we will return the SAE reconstruction directly.
            features_patched = features
        else:
            features_patched = self.policy.apply(features, site_mask=site_mask_sae, token_mask=token_mask)

        with torch.no_grad():
            eps = float(self.config.eps_active)
            region = site_mask_sae
            if token_mask is not None:
                region = region & token_mask.unsqueeze(-1)
            active = (features.abs() > eps) & region
            preserved = (features_patched.abs() > eps) & active
            self.state.total_active += float(active.float().sum().item())
            self.state.total_preserved += float(preserved.float().sum().item())

        if self.config.roundtrip_replace:
            recon_clean = self.sae.decode(features)
            if not isinstance(recon_clean, torch.Tensor) or recon_clean.shape != hidden_sae.shape:
                raise ValueError("SAE.decode must return a tensor of shape (batch, seq, hidden_dim)")
            recon_model = self.transform.inverse(recon_clean)
            recon_model = recon_model.to(device=hidden.device, dtype=hidden.dtype)
            region_model = site_mask.to(device=hidden.device, dtype=torch.bool)
            if token_mask is not None:
                region_model = region_model & token_mask.to(device=hidden.device, dtype=torch.bool).unsqueeze(-1)
            hidden_new = torch.where(region_model.expand_as(hidden), recon_model, hidden)
            if rest is None:
                return hidden_new
            return (hidden_new,) + rest

        if self.config.decode_strategy == "delta_1decode":
            delta_f = features_patched - features
            W_dec = _get_W_dec(self.sae, d_sae=int(features.size(-1)), d_in=int(hidden_dim)).to(
                device=features.device, dtype=delta_f.dtype
            )
            recon_delta = delta_f @ W_dec
        else:
            recon_clean = self.sae.decode(features)
            recon_patched = self.sae.decode(features_patched)
            if (
                not isinstance(recon_clean, torch.Tensor)
                or not isinstance(recon_patched, torch.Tensor)
                or recon_clean.shape != hidden_sae.shape
                or recon_patched.shape != hidden_sae.shape
            ):
                raise ValueError("SAE.decode must return a tensor of shape (batch, seq, hidden_dim)")
            recon_delta = recon_patched - recon_clean

        recon_delta = self.transform.inverse_delta(recon_delta)
        hidden_new = hidden + recon_delta.to(device=hidden.device, dtype=hidden.dtype)

        if rest is None:
            return hidden_new
        return (hidden_new,) + rest


def create_sae_hook_fn(
    adapter: SAEAdapter,
    site: SAEPatchSite,
    donor_hidden_states: Optional[torch.Tensor] = None,
    *,
    transform: Optional[SAEInputTransform] = None,
    token_mask: Optional[torch.Tensor] = None,
    state: Optional[SAEHookState] = None,
) -> Callable:
    """
    Create a forward hook function for SAE-based patching.

    This integrates with the existing AoM patching infrastructure by returning
    a hook that can be registered on decoder blocks.

    Args:
        adapter: SAEAdapter instance
        site: Patch site
        donor_hidden_states: Optional donor activations

    Returns:
        hook_fn: Callable compatible with register_forward_hook
    """
    transform = transform or SAEInputTransform()
    state = state or SAEHookState()

    def hook(module: nn.Module, inputs: Tuple, output: Tuple | torch.Tensor):
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("expected hidden states tensor of shape (batch, seq, hidden_dim)")

        batch_size, seq_len, hidden_dim = hidden.shape
        if hasattr(adapter.sae, "d_in") and int(getattr(adapter.sae, "d_in")) != int(hidden_dim):
            raise ValueError(
                f"SAE d_in={int(getattr(adapter.sae, 'd_in'))} does not match hidden_dim={int(hidden_dim)}"
            )

        sae_device, sae_dtype = _infer_sae_device_dtype(
            adapter.sae, fallback_device=hidden.device, fallback_dtype=torch.float32
        )
        compute_dtype = sae_dtype
        if adapter.config.dtype_policy == "model":
            if sae_dtype != hidden.dtype:
                warnings.warn(
                    f"dtype_policy='model' requested but SAE params are {sae_dtype} while model stream is {hidden.dtype}; "
                    "using SAE dtype to avoid dtype mismatch in encode/decode.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                compute_dtype = hidden.dtype

        hidden_sae = hidden.to(device=sae_device, dtype=compute_dtype)
        x_in = transform.forward(hidden_sae)
        features = adapter.sae.encode(x_in)
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("SAE.encode must return a tensor of shape (batch, seq, d_sae)")
        if hasattr(adapter.sae, "d_sae") and int(getattr(adapter.sae, "d_sae")) != int(features.size(-1)):
            raise ValueError(
                f"SAE d_sae={int(getattr(adapter.sae, 'd_sae'))} does not match features.size(-1)={int(features.size(-1))}"
            )

        # Apply SAE intervention via delta injection, preserving SAE residual error term.
        features_intervened = features.clone()
        if site.feature_mask is not None:
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, site.feature_mask] = 0.0
        if site.feature_values is not None:
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, :] = site.feature_values
        if donor_hidden_states is not None:
            donor_sae = donor_hidden_states.to(device=sae_device, dtype=compute_dtype)
            donor_in = transform.forward(donor_sae)
            donor_features = adapter.sae.encode(donor_in)
            for token_idx in site.token_indices:
                features_intervened[:, token_idx, :] = donor_features[:, token_idx, :]

        site_mask = torch.zeros((batch_size, seq_len, 1), device=features.device, dtype=torch.bool)
        for token_idx in site.token_indices:
            if token_idx < 0 or token_idx >= seq_len:
                raise ValueError(f"token index {token_idx} out of range [0, {seq_len})")
            site_mask[:, int(token_idx), 0] = True

        if token_mask is not None:
            token_mask = token_mask.to(device=features.device, dtype=torch.bool)
            if token_mask.ndim == 3 and token_mask.size(-1) == 1:
                token_mask = token_mask.squeeze(-1)
            if token_mask.shape != (batch_size, seq_len):
                raise ValueError("token_mask must have shape (batch, seq)")

        with torch.no_grad():
            eps = float(adapter.config.eps_active)
            region = site_mask
            if token_mask is not None:
                region = region & token_mask.unsqueeze(-1)
            active = (features.abs() > eps) & region
            preserved = (features_intervened.abs() > eps) & active
            state.total_active += float(active.float().sum().item())
            state.total_preserved += float(preserved.float().sum().item())

        if adapter.config.decode_strategy == "delta_1decode":
            delta_f = features_intervened - features
            W_dec = _get_W_dec(adapter.sae, d_sae=int(delta_f.size(-1)), d_in=int(hidden_dim)).to(
                device=delta_f.device, dtype=delta_f.dtype
            )
            recon_delta = delta_f @ W_dec
        else:
            recon_clean = adapter.sae.decode(features)
            recon_intervened = adapter.sae.decode(features_intervened)
            if (
                not isinstance(recon_clean, torch.Tensor)
                or not isinstance(recon_intervened, torch.Tensor)
                or recon_clean.shape != hidden_sae.shape
                or recon_intervened.shape != hidden_sae.shape
            ):
                raise ValueError("SAE.decode must return a tensor of shape (batch, seq, hidden_dim)")
            recon_delta = recon_intervened - recon_clean

        recon_delta = transform.inverse_delta(recon_delta)
        hidden_intervened = hidden + recon_delta.to(device=hidden.device, dtype=hidden.dtype)

        # Return in same format as input
        if rest is None:
            return hidden_intervened
        return (hidden_intervened,) + rest

    return hook
