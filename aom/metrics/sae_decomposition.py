"""
SAE Channel Decomposition for CPT experiments.

Decomposes CPT signal into SAE-reconstructable component vs reconstruction residual:
- Reconstructable (S): SAE decode(encode(x))
- Residual (C): x - S (what the SAE fails to capture)

Uses 2x2 factorial design. The key identity that makes this a valid decomposition:
    delta_r = delta_S + delta_C  (exactly, by construction)

Where:
    delta_r = r_donor - r_recv (full CPT delta)
    delta_S = S_donor - S_recv (reconstructable delta)
    delta_C = C_donor - C_recv (residual delta)

Conditions:
- A (baseline): receiver unchanged
- B (recon-only): receiver + delta_S
- C (resid-only): receiver + delta_C
- D (full): receiver + delta_r = receiver + delta_S + delta_C

Main effects:
- Recon effect: (B - A)
- Resid effect: (C - A)
- Interaction: (D - A) - (B - A) - (C - A)

The interaction term is interpretable as compatibility/synergy ONLY when using
faithful deltas (norm_matching="none"). With energy-matched mode, interaction
is a derived artifact of comparing different interventions.

NOTE: This is NOT a literal orthogonal subspace decomposition. The SAE reconstruction
is the decoder span projection only if the encoder perfectly inverts the decoder.
Claims should be framed as "this SAE's reconstructable component" not "the SAE subspace".

STATISTICS NOTE: Bootstrap CIs are computed at the item (pair) level, not direction
level, because the two directions per pair (a→b, b→a) are not independent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.activation_patching import (
    PatchSpanSite,
    get_block_outputs,
)
from aom.interventions.sae_adapter import SAEInputTransform, SAEProtocol
from aom.metrics.disamb import (
    _encode_prompt,
    _margin,
    score_labels_next_continuations,
    score_labels_next_continuations_patched,
)
from aom.token_spans import token_span_for_substring
from aom.utils import bootstrap_ci


def _infer_sae_device_dtype(sae: SAEProtocol) -> tuple[torch.device, torch.dtype]:
    if isinstance(sae, torch.nn.Module):
        p = next(sae.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    W_dec = getattr(sae, "W_dec", None)
    if isinstance(W_dec, torch.Tensor):
        return W_dec.device, W_dec.dtype
    return torch.device("cpu"), torch.float32


def _decompose(
    h: torch.Tensor,
    sae: SAEProtocol,
    transform: SAEInputTransform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose h into SAE-reconstructable (S) and residual (C) components.

    Args:
        h: Tensor of shape (B, S, H) - batched span activations
        sae: SAE with encode/decode methods expecting (B, S, H)
        transform: Input scaling transform

    Returns:
        S: SAE-reconstructable component, shape (B, S, H)
        C: Residual component (h - S), shape (B, S, H)
    """
    h_scaled = transform.forward(h)
    S = transform.inverse(sae.decode(sae.encode(h_scaled)))
    C = h - S
    return S, C


def _norm_match(
    vec: torch.Tensor,
    target_norm: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale vec to have the same L2 norm as target_norm."""
    current_norm = float(vec.norm().item())
    if current_norm < eps:
        return vec
    return vec * (target_norm / current_norm)


def _bootstrap_ci_items(
    item_values: List[List[float]],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Block bootstrap by item: each item contributes 0-2 direction values.

    Returns (mean, ci_low, ci_high).
    """
    import random

    # Filter to items with at least one value
    valid_items = [v for v in item_values if v]
    if not valid_items:
        return float("nan"), float("nan"), float("nan")

    # Compute overall mean (across all directions)
    all_vals = [x for item in valid_items for x in item]
    if not all_vals:
        return float("nan"), float("nan"), float("nan")
    overall_mean = sum(all_vals) / len(all_vals)

    # Block bootstrap: resample items with replacement
    rng = random.Random(seed)
    n_items = len(valid_items)
    bootstrap_means: List[float] = []

    for _ in range(n_bootstrap):
        sampled_items = [valid_items[rng.randint(0, n_items - 1)] for _ in range(n_items)]
        sampled_vals = [x for item in sampled_items for x in item]
        if sampled_vals:
            bootstrap_means.append(sum(sampled_vals) / len(sampled_vals))

    if not bootstrap_means:
        return overall_mean, float("nan"), float("nan")

    bootstrap_means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * len(bootstrap_means))
    hi_idx = int((1.0 - alpha) * len(bootstrap_means)) - 1
    lo_idx = max(0, min(lo_idx, len(bootstrap_means) - 1))
    hi_idx = max(0, min(hi_idx, len(bootstrap_means) - 1))

    return overall_mean, bootstrap_means[lo_idx], bootstrap_means[hi_idx]


@torch.inference_mode()
def compute_cpt_channel_decomposition(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    layer: int,
    sae: SAEProtocol,
    transform: SAEInputTransform,
    norm_matching: Literal["none", "energy_matched_independent"] = "none",
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    include_random_donor_sham: bool = True,
    include_ablation_controls: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    CPT channel decomposition with 2x2 factorial design.

    For each donor->receiver direction, runs 4 conditions:
    - A (baseline): receiver unchanged
    - B (recon_only): receiver + delta_S (donor's recon - receiver's recon)
    - C (resid_only): receiver + delta_C (donor's resid - receiver's resid)
    - D (full): receiver + delta_r (donor - receiver) = baseline CPT

    Args:
        norm_matching: How to handle norm differences between components.
            "none" (default): Faithful decomposition where delta_r = delta_S + delta_C.
                Interaction term is interpretable as compatibility/synergy.
            "energy_matched_independent": Scale delta_S and delta_C independently to
                match ||delta_r||. This is a robustness check ("can recon-only reproduce
                the effect with equal L2 budget?") but breaks decomposition validity.
                Interaction term is NOT interpretable as synergy in this mode.
        include_random_donor_sham: If True, compute effect with random donor's
            residual (from different ITEM with matching span signature) as a
            distribution-shift control.

    Returns dict with:
        - total_effect, recon_main_effect, resid_main_effect, interaction
        - norm_stats: ||delta_r||, ||delta_S||, ||delta_C||, additivity_error
        - conditions: raw effects for A/B/C/D/sham
        - random_donor_sham: effect with random donor (if enabled)

    Note: Bootstrap CIs use block bootstrap by item (not direction) since the
    two directions per pair are not independent.
    """
    # Ensure eval mode
    model.eval()
    if isinstance(sae, torch.nn.Module):
        sae.eval()

    sae_device, sae_dtype = _infer_sae_device_dtype(sae)

    # Per-item accumulators (for block bootstrap)
    # Each entry is a list of direction values for that item
    item_total_effects: List[List[float]] = [[] for _ in items]
    item_recon_main: List[List[float]] = [[] for _ in items]
    item_resid_main: List[List[float]] = [[] for _ in items]
    item_interactions: List[List[float]] = [[] for _ in items]
    item_flips_D: List[List[float]] = [[] for _ in items]
    item_sham_effects: List[List[float]] = [[] for _ in items]
    item_random_donor_effects: List[List[float]] = [[] for _ in items]
    # Ablation control accumulators
    item_precontext_effects: List[List[float]] = [[] for _ in items]
    item_random_vec_effects: List[List[float]] = [[] for _ in items]

    # Flat accumulators for norms (these are per-direction, not bootstrapped)
    norms_delta_r: List[float] = []
    norms_delta_S: List[float] = []
    norms_delta_C: List[float] = []
    additivity_errors: List[float] = []

    n_total_directions = 0
    n_skipped_misaligned = 0
    n_patched = 0

    # Bucket donors by span signature for random-donor sham
    # Key: (span_len, token_ids_tuple) -> List of (r_donor on CPU, item_idx)
    # Store on CPU to avoid GPU memory blow-up
    donor_buckets: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[torch.Tensor, int]]] = {}

    for item_idx, it in enumerate(items):
        for donor, recv in ((it.a, it.b), (it.b, it.a)):
            n_total_directions += 1
            donor_expected = donor.expected_label

            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(
                tokenizer, recv.prompt, it.target, it.target_occurrence
            )

            if len(donor_span) != len(recv_span) or (
                require_token_id_match and donor_token_ids != recv_token_ids
            ):
                n_skipped_misaligned += 1
                continue

            n_patched += 1

            # Get block outputs
            donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)

            donor_out = get_block_outputs(model, donor_ids, layers=[layer])
            recv_out = get_block_outputs(model, recv_ids, layers=[layer])

            # Extract residuals at target span, keeping batch dimension (B=1, span_len, H)
            r_donor = donor_out[layer][:, donor_span, :].detach()
            r_recv = recv_out[layer][:, recv_span, :].detach()

            # Move to SAE device/dtype for decomposition (still 3D)
            r_donor_sae = r_donor.to(device=sae_device, dtype=sae_dtype)
            r_recv_sae = r_recv.to(device=sae_device, dtype=sae_dtype)

            # Decompose (3D in, 3D out)
            S_donor, C_donor = _decompose(r_donor_sae, sae, transform)
            S_recv, C_recv = _decompose(r_recv_sae, sae, transform)

            # Store in bucket for random-donor sham (by span signature, keyed by ITEM index)
            # Store on CPU as 2D to minimize memory footprint
            span_sig = (len(donor_span), tuple(donor_token_ids))
            r_donor_cpu = r_donor[0].detach().to("cpu", dtype=torch.float32)
            donor_buckets.setdefault(span_sig, []).append((r_donor_cpu, item_idx))

            # Compute deltas (difference-patching)
            delta_r = r_donor_sae - r_recv_sae  # full delta
            delta_S = S_donor - S_recv  # recon delta
            delta_C = C_donor - C_recv  # resid delta

            # Track norms and additivity
            norm_r = float(delta_r.norm().item())
            norm_S = float(delta_S.norm().item())
            norm_C = float(delta_C.norm().item())
            additivity_err = float((delta_r - (delta_S + delta_C)).norm().item())
            additivity_rel = additivity_err / (norm_r + 1e-8)

            norms_delta_r.append(norm_r)
            norms_delta_S.append(norm_S)
            norms_delta_C.append(norm_C)
            additivity_errors.append(additivity_rel)

            # Apply norm matching if requested (robustness check, NOT decomposition)
            if norm_matching == "energy_matched_independent":
                if norm_r > 1e-8:
                    delta_S = _norm_match(delta_S, norm_r)
                    delta_C = _norm_match(delta_C, norm_r)

            # Construct patched activations: squeeze to 2D (span_len, H) for patching API
            # Note: Use r_donor directly for D to guarantee baseline CPT even if SAE dtype differs
            model_dtype = r_recv.dtype
            patches = {
                # A: baseline - no patch, just score receiver
                "B": (r_recv_sae + delta_S)[0].to(device=device, dtype=model_dtype),  # recon-only
                "C": (r_recv_sae + delta_C)[0].to(device=device, dtype=model_dtype),  # resid-only
                "D": r_donor[0].to(device=device, dtype=model_dtype),  # full CPT (donor residual directly)
                "sham": r_recv[0],  # no change
            }

            # Ablation controls: test what aspects of CPT signal matter
            if include_ablation_controls:
                # E (pre-context): Replace with raw embeddings W_E[token_ids]
                # Tests: Does CPT require contextualization, or just token identity?
                embed_layer = model.get_input_embeddings()
                token_ids_tensor = torch.tensor(donor_token_ids, device=device)
                precontext_embed = embed_layer(token_ids_tensor).to(dtype=model_dtype)
                patches["E_precontext"] = precontext_embed

                # F (random-vec): Norm-matched random Gaussian vector
                # Tests: Is the effect specific to direction, or just about perturbation magnitude?
                rand_gen = torch.Generator(device=device).manual_seed(bootstrap_seed + item_idx)
                random_vec = torch.randn(
                    len(donor_span), r_donor.shape[-1],
                    generator=rand_gen, device=device, dtype=model_dtype
                )
                # Match norm to full delta_r
                random_vec = random_vec * (norm_r / (random_vec.norm().item() + 1e-8))
                patches["F_random_vec"] = r_recv[0] + random_vec  # receiver + random perturbation

            # Baseline scores (condition A - no patch)
            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device,
                normalize_by_length=normalize_by_length
            )
            base_pred = base_scores.argmax_label()
            base_margin = _margin(base_scores, expected=donor_expected)
            effect_A = 0.0  # baseline effect is 0 by definition

            # Score patched conditions
            condition_effects = {"A": effect_A}
            for cond_name, replacement in patches.items():
                patched_scores = score_labels_next_continuations_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    patch_site=PatchSpanSite(layer=layer, token_indices=tuple(recv_span)),
                    replacement=replacement,
                    normalize_by_length=normalize_by_length,
                )
                patched_margin = _margin(patched_scores, expected=donor_expected)
                effect = float(patched_margin - base_margin)
                condition_effects[cond_name] = effect

                if cond_name == "D":
                    patched_pred = patched_scores.argmax_label()
                    flipped = float((base_pred != donor_expected) and (patched_pred == donor_expected))
                    item_flips_D[item_idx].append(flipped)

            # Store sham effect
            item_sham_effects[item_idx].append(condition_effects["sham"])

            # Store ablation control effects
            if include_ablation_controls:
                if "E_precontext" in condition_effects:
                    item_precontext_effects[item_idx].append(condition_effects["E_precontext"])
                if "F_random_vec" in condition_effects:
                    item_random_vec_effects[item_idx].append(condition_effects["F_random_vec"])

            # Compute factorial decomposition
            e_A = effect_A
            e_B = condition_effects["B"]
            e_C = condition_effects["C"]
            e_D = condition_effects["D"]

            total_effect = e_D - e_A
            recon_effect = e_B - e_A
            resid_effect = e_C - e_A
            interaction_effect = total_effect - recon_effect - resid_effect

            item_total_effects[item_idx].append(total_effect)
            item_recon_main[item_idx].append(recon_effect)
            item_resid_main[item_idx].append(resid_effect)
            item_interactions[item_idx].append(interaction_effect)

    # Random-donor sham: use residual from a DIFFERENT ITEM with matching span signature
    if include_random_donor_sham:
        import random
        rng = random.Random(bootstrap_seed)

        for item_idx, it in enumerate(items):
            for donor, recv in ((it.a, it.b), (it.b, it.a)):
                donor_span, donor_token_ids = token_span_for_substring(
                    tokenizer, donor.prompt, it.target, it.target_occurrence
                )
                recv_span, recv_token_ids = token_span_for_substring(
                    tokenizer, recv.prompt, it.target, it.target_occurrence
                )

                if len(donor_span) != len(recv_span) or (
                    require_token_id_match and donor_token_ids != recv_token_ids
                ):
                    continue

                # Get matching bucket
                span_sig = (len(donor_span), tuple(donor_token_ids))
                bucket = donor_buckets.get(span_sig, [])

                # Find candidates from DIFFERENT ITEMS (not just different directions)
                candidates = [j for j, (_, idx_item) in enumerate(bucket) if idx_item != item_idx]

                if not candidates:
                    continue

                random_bucket_idx = rng.choice(candidates)
                r_random_cpu, _ = bucket[random_bucket_idx]

                # Get receiver residual (for baseline scoring)
                recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
                recv_out = get_block_outputs(model, recv_ids, layers=[layer])
                r_recv = recv_out[layer][0, recv_span, :].detach()

                # Use random donor directly (cleaner than delta computation)
                replacement = r_random_cpu.to(device=device, dtype=r_recv.dtype)

                base_scores = score_labels_next_continuations(
                    model, tokenizer, recv.prompt, it.choices, device,
                    normalize_by_length=normalize_by_length
                )
                base_margin = _margin(base_scores, expected=donor.expected_label)

                patched_scores = score_labels_next_continuations_patched(
                    model, tokenizer, recv.prompt, it.choices, device,
                    patch_site=PatchSpanSite(layer=layer, token_indices=tuple(recv_span)),
                    replacement=replacement,
                    normalize_by_length=normalize_by_length,
                )
                patched_margin = _margin(patched_scores, expected=donor.expected_label)
                item_random_donor_effects[item_idx].append(float(patched_margin - base_margin))

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def _median(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        s = sorted(vals)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    # Compute CIs using block bootstrap by item
    total_mean, total_lo, total_hi = _bootstrap_ci_items(
        item_total_effects, bootstrap_n, ci, bootstrap_seed
    )
    recon_mean, recon_lo, recon_hi = _bootstrap_ci_items(
        item_recon_main, bootstrap_n, ci, bootstrap_seed
    )
    resid_mean, resid_lo, resid_hi = _bootstrap_ci_items(
        item_resid_main, bootstrap_n, ci, bootstrap_seed
    )
    inter_mean, inter_lo, inter_hi = _bootstrap_ci_items(
        item_interactions, bootstrap_n, ci, bootstrap_seed
    )
    flip_D_mean, flip_D_lo, flip_D_hi = _bootstrap_ci_items(
        item_flips_D, bootstrap_n, ci, bootstrap_seed
    )
    sham_mean, sham_lo, sham_hi = _bootstrap_ci_items(
        item_sham_effects, bootstrap_n, ci, bootstrap_seed
    )

    # Ablation control CIs
    precontext_mean, precontext_lo, precontext_hi = float("nan"), float("nan"), float("nan")
    random_vec_mean, random_vec_lo, random_vec_hi = float("nan"), float("nan"), float("nan")
    if include_ablation_controls:
        all_precontext = [x for item in item_precontext_effects for x in item]
        all_random_vec = [x for item in item_random_vec_effects for x in item]
        if all_precontext:
            precontext_mean, precontext_lo, precontext_hi = _bootstrap_ci_items(
                item_precontext_effects, bootstrap_n, ci, bootstrap_seed
            )
        if all_random_vec:
            random_vec_mean, random_vec_lo, random_vec_hi = _bootstrap_ci_items(
                item_random_vec_effects, bootstrap_n, ci, bootstrap_seed
            )

    # Flat lists for condition means (for backwards compatibility in output)
    all_effects_B = [x for item in item_recon_main for x in item]
    all_effects_C = [x for item in item_resid_main for x in item]
    all_effects_D = [x for item in item_total_effects for x in item]

    result: Dict[str, Any] = {
        "layer": int(layer),
        "norm_matching": str(norm_matching),
        "n_pairs": len(items),
        "n_directions_total": n_total_directions,
        "n_directions_patched": n_patched,
        "n_directions_skipped_misaligned": n_skipped_misaligned,
        # Factorial decomposition (the key results) - block bootstrapped by item
        "total_effect": {"mean": total_mean, "ci_low": total_lo, "ci_high": total_hi},
        "recon_main_effect": {"mean": recon_mean, "ci_low": recon_lo, "ci_high": recon_hi},
        "resid_main_effect": {"mean": resid_mean, "ci_low": resid_lo, "ci_high": resid_hi},
        "interaction": {"mean": inter_mean, "ci_low": inter_lo, "ci_high": inter_hi},
        # Norm statistics (for transparency)
        "norm_stats": {
            "delta_r_mean": _mean(norms_delta_r),
            "delta_r_median": _median(norms_delta_r),
            "delta_S_mean": _mean(norms_delta_S),
            "delta_S_median": _median(norms_delta_S),
            "delta_C_mean": _mean(norms_delta_C),
            "delta_C_median": _median(norms_delta_C),
            "additivity_error_mean": _mean(additivity_errors),
            "additivity_error_median": _median(additivity_errors),
        },
        # Raw condition effects
        "conditions": {
            "A_baseline": {"mean_effect": 0.0, "ci_low": 0.0, "ci_high": 0.0},
            "B_recon_only": {"mean_effect": _mean(all_effects_B), "ci_low": recon_lo, "ci_high": recon_hi},
            "C_resid_only": {"mean_effect": _mean(all_effects_C), "ci_low": resid_lo, "ci_high": resid_hi},
            "D_full": {
                "mean_effect": _mean(all_effects_D), "ci_low": total_lo, "ci_high": total_hi,
                "flip_rate": flip_D_mean, "flip_rate_ci_low": flip_D_lo, "flip_rate_ci_high": flip_D_hi,
            },
            "sham": {"mean_effect": sham_mean, "ci_low": sham_lo, "ci_high": sham_hi},
        },
    }

    # Ablation controls (if enabled)
    if include_ablation_controls:
        all_precontext = [x for item in item_precontext_effects for x in item]
        all_random_vec = [x for item in item_random_vec_effects for x in item]
        result["ablation_controls"] = {
            "E_precontext": {
                "description": "Raw embedding W_E[token_ids] - tests if contextualization matters",
                "mean_effect": precontext_mean,
                "ci_low": precontext_lo,
                "ci_high": precontext_hi,
                "n_samples": len(all_precontext),
            },
            "F_random_vec": {
                "description": "Norm-matched random perturbation - tests if specific direction matters",
                "mean_effect": random_vec_mean,
                "ci_low": random_vec_lo,
                "ci_high": random_vec_hi,
                "n_samples": len(all_random_vec),
            },
        }

    # Random donor sham (also block bootstrapped)
    all_random_effects = [x for item in item_random_donor_effects for x in item]
    if include_random_donor_sham and all_random_effects:
        rand_mean, rand_lo, rand_hi = _bootstrap_ci_items(
            item_random_donor_effects, bootstrap_n, ci, bootstrap_seed
        )
        result["random_donor_sham"] = {
            "mean_effect": rand_mean,
            "ci_low": rand_lo,
            "ci_high": rand_hi,
            "n_samples": len(all_random_effects),
        }

    return result
