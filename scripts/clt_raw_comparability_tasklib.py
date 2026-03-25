from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_coherence_items, load_counterfactual_pairs
from aom.interventions.activation_patching import (
    PatchSpanSite,
    forward_with_patched_block_output_span,
    get_block_outputs,
    get_num_layers,
)
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig, reconstruct_with_error_preservation
from aom.interventions.clt_loader import load_clt
from aom.interventions.clt_patch import ReplaceLatentsAtIndicesPolicy, forward_with_clt_latent_patching_span
from aom.interventions.patching.base import PatchingCase
from aom.interventions.patching.cf_protocol import CFPatchingConfig, CFInterventionSwapProtocol
from aom.interventions.patching.coh_protocol import COHPatchingConfig, COHConstraintAblationProtocol
from aom.metrics.primary_logodds import (
    PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT,
    choices_token_ids_from_strings,
    evaluate_primary_applicability,
    safe_log_prob_mass,
    signed_delta_margin_from_logodds,
)
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, get_logprob_computation_config, set_seed

TASK_CHOICES = ("cf", "coh")


def _parse_int_list(arg: str) -> List[int]:
    out: List[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer in --layers")
    return out


def _logmeanexp(xs: Sequence[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _aggregate(vals: List[float], *, agg: str) -> float:
    if len(vals) < 1:
        raise ValueError("aggregation requires at least one value")
    if str(agg) == "logmeanexp":
        return float(_logmeanexp(vals))
    if str(agg) == "mean":
        return float(sum(float(v) for v in vals) / len(vals))
    raise ValueError(f"Unsupported label aggregation: {agg!r}")


def _argmax_label(scores: Mapping[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _margin(scores: Mapping[str, float], expected: str) -> float:
    exp = float(scores[expected])
    best_other = max(float(v) for k, v in scores.items() if str(k) != str(expected))
    return float(exp - best_other)


def _best_other_label(scores: Mapping[str, float], expected: str) -> str:
    return max(((k, v) for k, v in scores.items() if str(k) != str(expected)), key=lambda kv: kv[1])[0]


def _encode_cont(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _next_token_logits_probs(
    *,
    prompt_ids: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = forward_fn(prompt_ids)
    if logits.ndim != 3 or int(logits.size(0)) != 1 or int(logits.size(1)) < 1:
        raise ValueError("Forward output must have shape (1, seq, vocab) with seq>=1")
    # MPS does not support float64 tensors; keep high precision elsewhere.
    next_dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    next_logits = logits[:, -1, :].to(dtype=next_dtype)[0]
    next_probs = torch.softmax(next_logits, dim=-1)
    return next_logits, next_probs


def _safe_err(e: Exception) -> str:
    s = str(e).strip()
    if not s:
        s = e.__class__.__name__
    s = s.replace("\n", " ")
    return s[:400]


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _infer_clt_device_dtype(clt: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(clt.parameters(), None)
    if p is not None:
        return p.device, p.dtype
    w_dec = getattr(clt, "W_dec", None)
    if isinstance(w_dec, torch.Tensor):
        return w_dec.device, w_dec.dtype
    return torch.device("cpu"), torch.float32


def _norm_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.norm(a).item() / (torch.norm(b).item() + float(eps)))


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    na = float(torch.norm(aa).item())
    nb = float(torch.norm(bb).item())
    if na <= eps or nb <= eps:
        return float("nan")
    return float(torch.dot(aa, bb).item() / (na * nb + eps))


def _score_choices_with_forward(
    *,
    tokenizer,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    normalize_by_length: bool,
    label_aggregation: str,
) -> Dict[str, float]:
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode_cont(tokenizer, str(cont), device=device)
            c_len = int(cont_ids.size(1))
            if c_len < 1:
                raise ValueError(f"Empty continuation tokenization for label={label!r} continuation={cont!r}")
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_fn(full_ids)
            p_len = int(prompt_ids.size(1))
            logits_slice = logits[:, p_len - 1 : p_len + c_len - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability in scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _aggregate(vals, agg=str(label_aggregation))
    return scores


def _logits_slice_for_continuation(
    *,
    tokenizer,
    prompt_ids: torch.Tensor,
    continuation: str,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cont_ids = _encode_cont(tokenizer, continuation, device=device)
    c_len = int(cont_ids.size(1))
    if c_len < 1:
        raise ValueError(f"Empty continuation tokenization for continuation={continuation!r}")
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = forward_fn(full_ids)
    p_len = int(prompt_ids.size(1))
    logits_slice = logits[:, p_len - 1 : p_len + c_len - 1, :].to(dtype=torch.float32)
    return logits_slice, cont_ids


def _decomp_deltas(base_logits: torch.Tensor, patched_logits: torch.Tensor, cont_ids: torch.Tensor) -> Dict[str, float]:
    if base_logits.shape != patched_logits.shape:
        raise ValueError("base_logits and patched_logits must have the same shape")
    if cont_ids.ndim != 2 or int(cont_ids.size(0)) != int(base_logits.size(0)):
        raise ValueError("continuation IDs shape mismatch")

    base_logz = torch.logsumexp(base_logits, dim=-1)
    patch_logz = torch.logsumexp(patched_logits, dim=-1)
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    patch_logprobs = torch.log_softmax(patched_logits, dim=-1)
    base_probs = torch.softmax(base_logits, dim=-1)

    gather_idx = cont_ids.unsqueeze(-1)
    base_target_logit = base_logits.gather(2, gather_idx).squeeze(-1)
    patch_target_logit = patched_logits.gather(2, gather_idx).squeeze(-1)
    base_target_logprob = base_logprobs.gather(2, gather_idx).squeeze(-1)
    patch_target_logprob = patch_logprobs.gather(2, gather_idx).squeeze(-1)

    delta_logit = patch_target_logit - base_target_logit
    delta_logz = patch_logz - base_logz
    delta_logprob = patch_target_logprob - base_target_logprob

    kl_base_patch = (base_probs * (base_logprobs - patch_logprobs)).sum(dim=-1)
    rms_logit_change = torch.sqrt(torch.mean((patched_logits - base_logits) ** 2, dim=-1))

    return {
        "delta_logit_target_mean": float(delta_logit.mean().item()),
        "delta_logz_mean": float(delta_logz.mean().item()),
        "delta_logprob_target_mean": float(delta_logprob.mean().item()),
        "kl_base_to_patch_mean": float(kl_base_patch.mean().item()),
        "rms_logit_change_mean": float(rms_logit_change.mean().item()),
    }


def _cluster_bootstrap_mean(
    *,
    rows: Sequence[Dict[str, Any]],
    key: str,
    pair_key: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Tuple[float, float, float, int, int]:
    pair_to_vals: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        value = r.get(key, float("nan"))
        if not isinstance(value, (int, float)):
            continue
        v = float(value)
        if not math.isfinite(v):
            continue
        pair_to_vals[str(r[pair_key])].append(v)
    if not pair_to_vals:
        return float("nan"), float("nan"), float("nan"), 0, 0

    pair_means: Dict[str, float] = {
        str(pid): float(sum(vals) / len(vals)) for pid, vals in pair_to_vals.items() if len(vals) > 0
    }
    if not pair_means:
        return float("nan"), float("nan"), float("nan"), 0, 0

    mean_val = float(sum(pair_means.values()) / len(pair_means))
    pairs = list(pair_means.keys())
    n_pairs = len(pairs)
    n_vals = int(sum(len(vals) for vals in pair_to_vals.values()))

    rng = random.Random(int(seed))
    boots: List[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_means: List[float] = []
        for _ in range(n_pairs):
            pid = pairs[rng.randrange(n_pairs)]
            sampled_means.append(float(pair_means[pid]))
        if sampled_means:
            boots.append(float(sum(sampled_means) / len(sampled_means)))
    if not boots:
        return mean_val, float("nan"), float("nan"), n_pairs, n_vals
    boots.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lo_idx = max(0, min(len(boots) - 1, int(alpha * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, int((1.0 - alpha) * len(boots)) - 1))
    return mean_val, float(boots[lo_idx]), float(boots[hi_idx]), n_pairs, n_vals


def _cluster_bootstrap_ratio(
    *,
    rows: Sequence[Dict[str, Any]],
    num_key: str,
    den_key: str,
    pair_key: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
    den_eps: float,
) -> Tuple[float, float, float, int]:
    pair_to_num_den: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        n = r.get(num_key, float("nan"))
        d = r.get(den_key, float("nan"))
        if not isinstance(n, (int, float)) or not isinstance(d, (int, float)):
            continue
        nf = float(n)
        df = float(d)
        if not (math.isfinite(nf) and math.isfinite(df)):
            continue
        pair_to_num_den[str(r[pair_key])].append((nf, df))
    if not pair_to_num_den:
        return float("nan"), float("nan"), float("nan"), 0

    pair_num_mean: Dict[str, float] = {}
    pair_den_mean: Dict[str, float] = {}
    for pid, vals in pair_to_num_den.items():
        if len(vals) < 1:
            continue
        pair_num_mean[str(pid)] = float(sum(v[0] for v in vals) / len(vals))
        pair_den_mean[str(pid)] = float(sum(v[1] for v in vals) / len(vals))
    if not pair_num_mean:
        return float("nan"), float("nan"), float("nan"), 0

    pairs = list(pair_num_mean.keys())
    n_pairs = len(pairs)
    mean_num = float(sum(pair_num_mean.values()) / len(pair_num_mean))
    mean_den = float(sum(pair_den_mean.values()) / len(pair_den_mean))
    point = float("nan") if abs(mean_den) <= float(den_eps) else float(mean_num / mean_den)

    rng = random.Random(int(seed))
    boots: List[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_num: List[float] = []
        sampled_den: List[float] = []
        for _ in range(n_pairs):
            pid = pairs[rng.randrange(n_pairs)]
            sampled_num.append(float(pair_num_mean[pid]))
            sampled_den.append(float(pair_den_mean[pid]))
        if not sampled_num:
            continue
        b_num = float(sum(sampled_num) / len(sampled_num))
        b_den = float(sum(sampled_den) / len(sampled_den))
        if abs(b_den) <= float(den_eps):
            continue
        boots.append(float(b_num / b_den))
    if not boots:
        return point, float("nan"), float("nan"), n_pairs
    boots.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lo_idx = max(0, min(len(boots) - 1, int(alpha * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, int((1.0 - alpha) * len(boots)) - 1))
    return point, float(boots[lo_idx]), float(boots[hi_idx]), n_pairs


def _compute_clt_writeback_delta(
    *,
    clt,
    transform: CLTInputTransform,
    recv_slice: torch.Tensor,
    recv_latents: torch.Tensor,
    donor_latents: torch.Tensor,
    decode_strategy: str,
) -> torch.Tensor:
    if str(decode_strategy) == "delta_1decode":
        y_hat_t = clt.decode(recv_latents)
        y_patched_t = clt.decode(donor_latents)
        return transform.inverse_delta(y_patched_t - y_hat_t)
    if str(decode_strategy) == "safe_2decode":
        y_prime, _y_hat, _err = reconstruct_with_error_preservation(
            clt=clt,
            receiver_x=recv_slice,
            receiver_y=recv_slice,
            z_prime=donor_latents,
            transform=transform,
        )
        return y_prime - recv_slice
    raise ValueError(f"Unknown decode_strategy={decode_strategy!r}")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    _ensure_dir(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _pair_id_for_case(case: PatchingCase, *, task: str) -> str:
    if str(task) == "coh":
        return str(case.case_id).split("__", 1)[0]
    return str(case.case_id)


def _aggregate_layer(
    *,
    rows: Sequence[Dict[str, Any]],
    layer: int,
    n_bootstrap: int,
    ci: float,
    seed: int,
    ratio_den_eps: float,
) -> Dict[str, Any]:
    layer_rows = [r for r in rows if int(r["layer"]) == int(layer)]
    out: Dict[str, Any] = {
        "layer": int(layer),
        "n_rows": int(len(layer_rows)),
        "n_pairs": int(len({str(r["pair_id"]) for r in layer_rows})),
    }
    metrics = [
        "effect_A",
        "effect_B",
        "effect_C",
        "effect_D",
        "effect_Cp",
        "effect_Dp",
        "effect_RI",
        "effect_CI",
        "d_BA",
        "d_CA",
        "d_DA",
        "d_CCp",
        "d_DDp",
        "sign_agree_BA",
        "sign_agree_CA",
        "sign_agree_DA",
        "sign_agree_CCp",
        "sign_agree_DDp",
        "gate_activation_ratio",
        "gate_margin_abs_diff",
        "gate_score_abs_diff_exp",
        "gate_score_abs_diff_other",
        "gate_clt_effect_abs_diff_C",
        "gate_clt_effect_abs_diff_D",
        "gate_identity_abs_effect_raw",
        "gate_identity_abs_effect_clt",
        "raw_delta_norm",
        "raw_delta_norm_small",
        "recon_norm_ratio",
        "clt_delta_norm_ratio_C",
        "clt_delta_norm_ratio_D",
        "clt_delta_cosine_raw_C",
        "clt_delta_cosine_raw_D",
        "fidelity_rel_mse",
        "fidelity_rel_l2",
        "fidelity_cosine",
        "active_latent_frac",
        "decomp_exp_delta_logit_target_mean_C",
        "decomp_exp_delta_logz_mean_C",
        "decomp_exp_delta_logprob_target_mean_C",
        "decomp_exp_kl_base_to_patch_mean_C",
        "decomp_exp_rms_logit_change_mean_C",
        "decomp_exp_delta_logit_target_mean_A",
        "decomp_exp_delta_logz_mean_A",
        "decomp_exp_delta_logprob_target_mean_A",
        "decomp_exp_kl_base_to_patch_mean_A",
        "decomp_exp_rms_logit_change_mean_A",
        "primary_logodds_applicable_flag",
        "primary_cancellation_pass_flag",
        "P_E_base",
        "P_O_base",
        "P_rest_base",
        "P_E_patch_A",
        "P_O_patch_A",
        "P_rest_patch_A",
        "P_E_patch_C",
        "P_O_patch_C",
        "P_rest_patch_C",
        "dlogPE_A",
        "dlogPO_A",
        "dlogPE_C",
        "dlogPO_C",
        "dPE_A",
        "dPO_A",
        "dPrest_A",
        "dPE_C",
        "dPO_C",
        "dPrest_C",
        "d_CA_logPE",
        "d_CA_logPO",
        "primary_delta_m_from_logodds_A",
        "primary_delta_m_from_logodds_C",
        "primary_delta_m_residual_A",
        "primary_delta_m_residual_C",
        "primary_base_logodds_residual",
    ]
    for key in metrics:
        m, lo, hi, n_pairs, n_vals = _cluster_bootstrap_mean(
            rows=layer_rows,
            key=key,
            pair_key="pair_id",
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        out[f"{key}_mean"] = m
        out[f"{key}_ci_low"] = lo
        out[f"{key}_ci_high"] = hi
        out[f"{key}_n_pairs"] = int(n_pairs)
        out[f"{key}_n_vals"] = int(n_vals)

    for num_key, den_key, name in (("effect_C", "effect_A", "crr_C_over_A"), ("effect_D", "effect_A", "crr_D_over_A")):
        ratio, lo, hi, n_pairs = _cluster_bootstrap_ratio(
            rows=layer_rows,
            num_key=num_key,
            den_key=den_key,
            pair_key="pair_id",
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            den_eps=ratio_den_eps,
        )
        out[f"{name}_mean"] = ratio
        out[f"{name}_ci_low"] = lo
        out[f"{name}_ci_high"] = hi
        out[f"{name}_n_pairs"] = int(n_pairs)
    return out


@dataclass(frozen=True)
class LayerCLTBundle:
    clt: torch.nn.Module
    transform: CLTInputTransform
    meta: Any


def _default_out_paths(task: str) -> Tuple[Path, Path]:
    stem = f"clt_raw_comparability_{str(task)}_l4_l8_l12"
    return ROOT / "results" / f"{stem}.csv", ROOT / "results" / f"{stem}.summary.json"


def build_arg_parser(task_default: Optional[str] = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Raw-vs-CLT comparability with A≈B invariant gate, identity controls, and decomposition telemetry "
            "for AoM-CF / AoM-COH protocol cases."
        )
    )
    p.add_argument("--task", type=str, choices=list(TASK_CHOICES), default=task_default)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--cf_path", type=str, default=str(ROOT / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(ROOT / "data" / "coherence.jsonl"))
    p.add_argument("--layers", type=str, default="4,8,12")
    p.add_argument("--max_items", type=int, default=0, help="If >0, limit loaded dataset items before protocol expansion.")

    p.add_argument("--cf_max_total_len_delta", type=int, default=5)
    p.add_argument("--cf_include_expected_effects", type=str, default="shift,invariant")
    p.add_argument(
        "--cf_span_mode",
        type=str,
        default="divergent_only",
        choices=["divergent_only", "divergent_plus_downstream", "left_aligned_truncated"],
    )
    p.add_argument("--coh_max_total_len_delta", type=int, default=20)

    p.add_argument("--clt_repo", type=str, required=True)
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_run_name", type=str, default=None)
    p.add_argument("--clt_l0_target", type=int, default=None)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype_policy", type=str, default="clt", choices=["clt", "model"])
    p.add_argument("--clt_eps_active", type=float, default=1e-6)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalize_by_length", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--gate_activation_ratio_tol", type=float, default=1e-6)
    p.add_argument("--gate_margin_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_score_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_clt_equiv_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_identity_abs_tol", type=float, default=1e-4)
    p.add_argument("--hard_fail_invariant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--hard_fail_primary_logodds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if primary single-token candidate-set log-odds identity residual exceeds tolerance on applicable rows.",
    )
    p.add_argument(
        "--primary_logodds_residual_tol",
        type=float,
        default=float(PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT),
        help="Absolute tolerance for primary single-token log-odds residual checks.",
    )
    p.add_argument("--ratio_den_eps", type=float, default=1e-6)
    p.add_argument("--raw_delta_norm_eps", type=float, default=1e-8)
    p.add_argument("--fidelity_rel_mse_warn", type=float, default=0.1)

    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--bootstrap_seed", type=int, default=42)

    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    return p


def _build_cases_for_task(
    *,
    task: str,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[PatchingCase], Dict[str, int], int]:
    if str(task) == "cf":
        items = load_counterfactual_pairs(str(args.cf_path), validate=True)
        if int(args.max_items) > 0:
            items = items[: int(args.max_items)]
        include_effects = tuple(
            s.strip() for s in str(args.cf_include_expected_effects).split(",") if s.strip()
        ) or ("shift", "invariant")
        protocol = CFInterventionSwapProtocol(
            config=CFPatchingConfig(
                max_total_len_delta=int(args.cf_max_total_len_delta),
                include_expected_effects=tuple(str(x) for x in include_effects),
                span_mode=str(args.cf_span_mode),
            )
        )
        cases, skips = protocol.build_cases(tokenizer=tokenizer, items=items, device=device)
        skip_counts = dict(sorted(Counter(str(s.reason) for s in skips).items()))
        return list(cases), {str(k): int(v) for k, v in skip_counts.items()}, int(len(items))

    if str(task) == "coh":
        items = load_coherence_items(str(args.coh_path), validate=True)
        if int(args.max_items) > 0:
            items = items[: int(args.max_items)]
        protocol = COHConstraintAblationProtocol(
            config=COHPatchingConfig(max_total_len_delta=int(args.coh_max_total_len_delta))
        )
        cases, skips = protocol.build_cases(tokenizer=tokenizer, items=items, device=device)
        skip_counts = dict(sorted(Counter(str(s.reason) for s in skips).items()))
        return list(cases), {str(k): int(v) for k, v in skip_counts.items()}, int(len(items))

    raise ValueError(f"Unsupported task={task!r}")


def run(args: argparse.Namespace) -> None:
    task = str(getattr(args, "task", "")).strip().lower()
    if task not in TASK_CHOICES:
        raise ValueError(f"--task must be one of {TASK_CHOICES}, got {task!r}")

    out_csv_default, out_json_default = _default_out_paths(task)
    out_csv_path = Path(str(args.out_csv).strip() or str(out_csv_default))
    out_json_path = Path(str(args.out_json).strip() or str(out_json_default))

    set_seed(int(args.seed))
    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        revision=getattr(args, "revision", None),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.eval()

    n_layers_model = int(get_num_layers(model))
    layers = _parse_int_list(str(args.layers))
    for layer in layers:
        if layer < 0 or layer >= n_layers_model:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers_model})")

    cases, protocol_skip_counts, n_items_loaded = _build_cases_for_task(
        task=task,
        tokenizer=tokenizer,
        device=device,
        args=args,
    )
    if not cases:
        raise ValueError(f"No protocol cases built for task={task!r}.")

    bundles: Dict[int, LayerCLTBundle] = {}
    for layer in layers:
        clt, meta = load_clt(
            str(args.clt_repo),
            layer=int(layer),
            width=str(args.clt_width),
            run_name=args.clt_run_name,
            l0_target=args.clt_l0_target,
            device=str(device),
            dtype=str(args.clt_dtype),
            local_files_only=bool(args.local_files_only),
        )
        if str(meta.site_mode) != "same_site_v1":
            raise ValueError(f"Layer {layer}: unsupported CLT site_mode={meta.site_mode!r}")
        if not (
            str(meta.encode_site) == "resid_post"
            and str(meta.decode_site) == "resid_post"
            and str(meta.writeback_site) == "resid_post"
        ):
            raise ValueError(
                f"Layer {layer}: CLT site mismatch, expected resid_post; "
                f"got encode={meta.encode_site}, decode={meta.decode_site}, writeback={meta.writeback_site}"
            )
        bundles[int(layer)] = LayerCLTBundle(
            clt=clt,
            transform=CLTInputTransform(scale=float(args.clt_scale)),
            meta=meta,
        )

    cfg_c = CLTPatchConfig(
        decode_strategy="delta_1decode",
        dtype_policy=str(args.clt_dtype_policy),
        eps_active=float(args.clt_eps_active),
    )
    cfg_d = CLTPatchConfig(
        decode_strategy="safe_2decode",
        dtype_policy=str(args.clt_dtype_policy),
        eps_active=float(args.clt_eps_active),
    )

    rows: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = defaultdict(int)
    fail_reasons: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _record_arm_failure(arm: str, reason: str) -> None:
        fail_counts[str(arm)] += 1
        fail_reasons[str(arm)][str(reason)] += 1

    with torch.inference_mode():
        for case in cases:
            pair_id = _pair_id_for_case(case, task=task)
            row_common: Dict[str, Any] = {
                "task": str(task),
                "pair_id": str(pair_id),
                "case_id": str(case.case_id),
                "receiver_span_len": int(len(case.receiver_span)),
                "donor_span_len": int(len(case.donor_span)),
                "effect_sign": float(case.effect_sign),
                "expected_label": str(case.expected_label),
                "label_aggregation": str(case.label_aggregation),
            }
            for k, v in case.strata.items():
                row_common[f"stratum_{str(k)}"] = str(v)

            if len(case.receiver_span) < 1 or len(case.donor_span) < 1 or len(case.receiver_span) != len(case.donor_span):
                for layer in layers:
                    r = dict(row_common)
                    r.update(
                        {
                            "layer": int(layer),
                            "all_arms_success": False,
                            "invariant_all_pass": False,
                            "analysis_included": False,
                            "skip_reason": "span_mismatch",
                        }
                    )
                    rows.append(r)
                continue

            if str(case.expected_label) not in case.choices:
                fail_counts["case_missing_expected_label"] += 1
                for layer in layers:
                    r = dict(row_common)
                    r.update(
                        {
                            "layer": int(layer),
                            "all_arms_success": False,
                            "invariant_all_pass": False,
                            "analysis_included": False,
                            "skip_reason": "missing_expected_label",
                        }
                    )
                    rows.append(r)
                continue

            prompt_ids_donor = case.donor_ids
            prompt_ids_recv = case.receiver_ids

            base_forward = lambda full_ids: model(input_ids=full_ids, use_cache=False, return_dict=True).logits
            base_scores = _score_choices_with_forward(
                tokenizer=tokenizer,
                prompt_ids=prompt_ids_recv,
                choices=case.choices,
                forward_fn=base_forward,
                device=device,
                normalize_by_length=bool(args.normalize_by_length),
                label_aggregation=str(case.label_aggregation),
            )
            base_pred = _argmax_label(base_scores)
            donor_expected = str(case.expected_label)
            other_label = _best_other_label(base_scores, expected=donor_expected)
            base_margin = _margin(base_scores, expected=donor_expected)

            # Primary endpoint-native mechanism metadata (single-token candidate-set log-odds).
            choices_token_ids = choices_token_ids_from_strings(tokenizer, case.choices)
            primary_label_agg_logmeanexp = bool(str(case.label_aggregation) == "logmeanexp")
            prompt_boundary_pos = int(prompt_ids_recv.size(1) - 1)
            scored_positions: List[int] = []
            for _seq in choices_token_ids.get(donor_expected, ()):
                scored_positions.append(prompt_boundary_pos)
            for _seq in choices_token_ids.get(other_label, ()):
                scored_positions.append(prompt_boundary_pos)
            primary_meta = evaluate_primary_applicability(
                choices_token_ids=choices_token_ids,
                expected_label=donor_expected,
                other_label=other_label,
                scored_positions=scored_positions,
                require_binary_labels=True,
                require_equal_candidate_counts=True,
            )

            primary_labels_binary = bool(primary_meta.labels_binary)
            expected_all_single = bool(primary_meta.expected_all_single)
            other_all_single = bool(primary_meta.other_all_single)
            expected_tokens = [int(x) for x in primary_meta.expected_tokens]
            other_tokens = [int(x) for x in primary_meta.other_tokens]
            expected_unique = [int(x) for x in primary_meta.expected_unique_tokens]
            other_unique = [int(x) for x in primary_meta.other_unique_tokens]
            expected_no_duplicates = bool(primary_meta.no_duplicates_expected)
            other_no_duplicates = bool(primary_meta.no_duplicates_other)
            token_sets_disjoint = bool(primary_meta.token_sets_disjoint)
            candidate_count_equal = bool(primary_meta.candidate_count_equal)
            same_scored_position = bool(primary_meta.same_scored_position)

            primary_static_applicable = bool(primary_label_agg_logmeanexp and primary_meta.static_applicable)
            if not primary_labels_binary:
                primary_static_reason = "labels_not_binary"
            elif not primary_label_agg_logmeanexp:
                primary_static_reason = "label_aggregation_not_logmeanexp"
            else:
                primary_static_reason = str(primary_meta.reason)

            primary_base_stats: Dict[str, float] = {
                "P_E_base": float("nan"),
                "P_O_base": float("nan"),
                "P_rest_base": float("nan"),
                "logPE_base": float("nan"),
                "logPO_base": float("nan"),
                "margin_from_logodds_base": float("nan"),
                "base_logodds_residual": float("nan"),
            }
            if primary_static_applicable:
                _base_logits, base_next_probs = _next_token_logits_probs(
                    prompt_ids=prompt_ids_recv, forward_fn=base_forward
                )
                idx_exp = torch.tensor(expected_unique, device=base_next_probs.device, dtype=torch.long)
                idx_oth = torch.tensor(other_unique, device=base_next_probs.device, dtype=torch.long)
                p_e_base = float(base_next_probs.index_select(0, idx_exp).sum().item())
                p_o_base = float(base_next_probs.index_select(0, idx_oth).sum().item())
                p_rest_base = float(1.0 - p_e_base - p_o_base)
                log_pe_base = safe_log_prob_mass(p_e_base)
                log_po_base = safe_log_prob_mass(p_o_base)
                margin_from_logodds_base_raw = float(log_pe_base - log_po_base)
                base_logodds_residual = float(base_margin - margin_from_logodds_base_raw)
                primary_base_stats.update(
                    {
                        "P_E_base": p_e_base,
                        "P_O_base": p_o_base,
                        "P_rest_base": p_rest_base,
                        "logPE_base": log_pe_base,
                        "logPO_base": log_po_base,
                        "margin_from_logodds_base": margin_from_logodds_base_raw,
                        "base_logodds_residual": base_logodds_residual,
                    }
                )
                if (
                    bool(args.hard_fail_primary_logodds)
                    and abs(float(base_logodds_residual)) > float(args.primary_logodds_residual_tol)
                ):
                    raise RuntimeError(
                        "Primary log-odds base residual exceeded tolerance: "
                        f"case={case.case_id}, residual={base_logodds_residual:.3e}, "
                        f"tol={float(args.primary_logodds_residual_tol):.3e}"
                    )

            donor_out = get_block_outputs(model, prompt_ids_donor, layers=layers)
            recv_out = get_block_outputs(model, prompt_ids_recv, layers=layers)

            for layer in layers:
                raw_donor = donor_out[int(layer)][0, list(case.donor_span), :].detach()
                raw_recv = recv_out[int(layer)][0, list(case.receiver_span), :].detach()
                raw_delta = raw_donor - raw_recv
                raw_swap_replacement = raw_donor
                raw_delta_replacement = raw_recv + raw_delta

                gate_activation_ratio = _norm_ratio(raw_swap_replacement - raw_delta_replacement, raw_swap_replacement)
                gate_activation_pass = bool(gate_activation_ratio < float(args.gate_activation_ratio_tol))

                bundle = bundles[int(layer)]
                clt = bundle.clt
                transform = bundle.transform
                clt_device, clt_dtype = _infer_clt_device_dtype(clt)
                donor_slice = raw_donor.unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                recv_slice = raw_recv.unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                recv_latents = clt.encode(transform.forward(recv_slice))
                donor_latents = clt.encode(transform.forward(donor_slice))
                active_latent_mask = recv_latents.abs() > float(args.clt_eps_active)
                active_latent_count = int(active_latent_mask.sum().item())
                active_latent_frac = float(active_latent_mask.to(dtype=torch.float32).mean().item())

                recon = transform.inverse(clt.decode(recv_latents))
                recon_err = recon - recv_slice
                err_sse = float((recon_err**2).sum().item())
                recv_sse = float((recv_slice**2).sum().item())
                fidelity_rel_mse = float(err_sse / (recv_sse + 1e-12))
                fidelity_rel_l2 = float(torch.norm(recon_err).item() / (torch.norm(recv_slice).item() + 1e-12))
                fidelity_cosine = _cosine(recon, recv_slice)
                recon_norm_ratio = _norm_ratio(recon, recv_slice)

                clt_delta_c = _compute_clt_writeback_delta(
                    clt=clt,
                    transform=transform,
                    recv_slice=recv_slice,
                    recv_latents=recv_latents,
                    donor_latents=donor_latents,
                    decode_strategy="delta_1decode",
                ).to(device=raw_delta.device, dtype=raw_delta.dtype)[0]
                clt_delta_d = _compute_clt_writeback_delta(
                    clt=clt,
                    transform=transform,
                    recv_slice=recv_slice,
                    recv_latents=recv_latents,
                    donor_latents=donor_latents,
                    decode_strategy="safe_2decode",
                ).to(device=raw_delta.device, dtype=raw_delta.dtype)[0]

                raw_delta_norm = float(torch.norm(raw_delta).item())
                clt_delta_norm_c = float(torch.norm(clt_delta_c).item())
                clt_delta_norm_d = float(torch.norm(clt_delta_d).item())
                raw_delta_norm_small = bool(raw_delta_norm <= float(args.raw_delta_norm_eps))
                clt_delta_ratio_c = (
                    float(clt_delta_norm_c / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                )
                clt_delta_ratio_d = (
                    float(clt_delta_norm_d / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                )
                clt_delta_cos_c = _cosine(clt_delta_c, raw_delta)
                clt_delta_cos_d = _cosine(clt_delta_d, raw_delta)

                site = PatchSpanSite(layer=int(layer), token_indices=tuple(int(i) for i in case.receiver_span))
                clt_policy = ReplaceLatentsAtIndicesPolicy(
                    token_indices=[int(i) for i in case.receiver_span], replacement_latents=donor_latents
                )
                clt_identity_policy = ReplaceLatentsAtIndicesPolicy(
                    token_indices=[int(i) for i in case.receiver_span], replacement_latents=recv_latents
                )
                raw_cprime_replacement = raw_recv + clt_delta_c
                raw_dprime_replacement = raw_recv + clt_delta_d

                arm_results: Dict[str, Dict[str, Any]] = {}
                exp_cont = str(case.choices[donor_expected][0])
                oth_cont = str(case.choices[other_label][0])
                base_exp_logits, base_exp_cont_ids = _logits_slice_for_continuation(
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids_recv,
                    continuation=exp_cont,
                    forward_fn=base_forward,
                    device=device,
                )

                forward_a = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_swap_replacement,
                )
                forward_b = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_delta_replacement,
                )
                forward_c = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_policy,
                    transform=transform,
                    config=cfg_c,
                )
                forward_d = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_policy,
                    transform=transform,
                    config=cfg_d,
                )
                forward_cp = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_cprime_replacement,
                )
                forward_dp = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_dprime_replacement,
                )
                forward_ri = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_recv,
                )
                forward_ci = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_identity_policy,
                    transform=transform,
                    config=cfg_c,
                )
                arm_forward_fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                    "A": forward_a,
                    "B": forward_b,
                    "C": forward_c,
                    "D": forward_d,
                    "Cp": forward_cp,
                    "Dp": forward_dp,
                    "RI": forward_ri,
                    "CI": forward_ci,
                }
                required_arms = ("A", "B", "C", "D", "Cp", "Dp", "RI", "CI")
                base_oth_logits, base_oth_cont_ids = _logits_slice_for_continuation(
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids_recv,
                    continuation=oth_cont,
                    forward_fn=base_forward,
                    device=device,
                )

                def _evaluate_arm(arm_name: str, forward_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
                    scores = _score_choices_with_forward(
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids_recv,
                        choices=case.choices,
                        forward_fn=forward_fn,
                        device=device,
                        normalize_by_length=bool(args.normalize_by_length),
                        label_aggregation=str(case.label_aggregation),
                    )
                    patched_margin = _margin(scores, expected=donor_expected)
                    raw_effect = float(patched_margin - base_margin)
                    effect = float(float(case.effect_sign) * raw_effect)
                    pred = _argmax_label(scores)
                    delta_score_exp = float(scores[donor_expected] - base_scores[donor_expected])
                    delta_score_other = float(scores[other_label] - base_scores[other_label])

                    patch_exp_logits, _ = _logits_slice_for_continuation(
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids_recv,
                        continuation=exp_cont,
                        forward_fn=forward_fn,
                        device=device,
                    )
                    patch_oth_logits, _ = _logits_slice_for_continuation(
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids_recv,
                        continuation=oth_cont,
                        forward_fn=forward_fn,
                        device=device,
                    )
                    decomp_exp = _decomp_deltas(base_exp_logits, patch_exp_logits, base_exp_cont_ids)
                    decomp_oth = _decomp_deltas(base_oth_logits, patch_oth_logits, base_oth_cont_ids)

                    arm_results[arm_name] = {
                        "success": True,
                        "effect": effect,
                        "pred": str(pred),
                        "delta_score_exp": delta_score_exp,
                        "delta_score_other": delta_score_other,
                        "decomp_exp": decomp_exp,
                        "decomp_other": decomp_oth,
                    }

                def _arm_fail(arm_name: str, e: Exception) -> None:
                    reason = _safe_err(e)
                    _record_arm_failure(arm_name, reason)
                    arm_results[arm_name] = {"success": False, "error": reason}

                for arm in required_arms:
                    try:
                        _evaluate_arm(arm, arm_forward_fns[arm])
                    except Exception as e:
                        _arm_fail(arm, e)
                all_success = all(bool(arm_results.get(arm, {}).get("success", False)) for arm in required_arms)

                primary_logodds_applicable = bool(all_success and primary_static_applicable)
                primary_logodds_reason = "ok" if primary_logodds_applicable else (
                    "all_arms_not_success" if not all_success else str(primary_static_reason)
                )
                primary_P_E_patch_A = float("nan")
                primary_P_O_patch_A = float("nan")
                primary_P_rest_patch_A = float("nan")
                primary_P_E_patch_C = float("nan")
                primary_P_O_patch_C = float("nan")
                primary_P_rest_patch_C = float("nan")
                primary_dlogPE_A = float("nan")
                primary_dlogPO_A = float("nan")
                primary_dlogPE_C = float("nan")
                primary_dlogPO_C = float("nan")
                primary_dPE_A = float("nan")
                primary_dPO_A = float("nan")
                primary_dPrest_A = float("nan")
                primary_dPE_C = float("nan")
                primary_dPO_C = float("nan")
                primary_dPrest_C = float("nan")
                primary_d_CA_logPE = float("nan")
                primary_d_CA_logPO = float("nan")
                primary_delta_m_from_logodds_A = float("nan")
                primary_delta_m_from_logodds_C = float("nan")
                primary_delta_m_residual_A = float("nan")
                primary_delta_m_residual_C = float("nan")
                primary_cancellation_pass = False

                if primary_logodds_applicable:
                    idx_exp = torch.tensor(expected_unique, device=prompt_ids_recv.device, dtype=torch.long)
                    idx_oth = torch.tensor(other_unique, device=prompt_ids_recv.device, dtype=torch.long)

                    _logits_a, probs_a = _next_token_logits_probs(prompt_ids=prompt_ids_recv, forward_fn=arm_forward_fns["A"])
                    _logits_c, probs_c = _next_token_logits_probs(prompt_ids=prompt_ids_recv, forward_fn=arm_forward_fns["C"])

                    primary_P_E_patch_A = float(probs_a.index_select(0, idx_exp).sum().item())
                    primary_P_O_patch_A = float(probs_a.index_select(0, idx_oth).sum().item())
                    primary_P_rest_patch_A = float(1.0 - primary_P_E_patch_A - primary_P_O_patch_A)
                    primary_P_E_patch_C = float(probs_c.index_select(0, idx_exp).sum().item())
                    primary_P_O_patch_C = float(probs_c.index_select(0, idx_oth).sum().item())
                    primary_P_rest_patch_C = float(1.0 - primary_P_E_patch_C - primary_P_O_patch_C)

                    primary_dlogPE_A = float(
                        safe_log_prob_mass(primary_P_E_patch_A) - float(primary_base_stats["logPE_base"])
                    )
                    primary_dlogPO_A = float(
                        safe_log_prob_mass(primary_P_O_patch_A) - float(primary_base_stats["logPO_base"])
                    )
                    primary_dlogPE_C = float(
                        safe_log_prob_mass(primary_P_E_patch_C) - float(primary_base_stats["logPE_base"])
                    )
                    primary_dlogPO_C = float(
                        safe_log_prob_mass(primary_P_O_patch_C) - float(primary_base_stats["logPO_base"])
                    )

                    primary_dPE_A = float(primary_P_E_patch_A - float(primary_base_stats["P_E_base"]))
                    primary_dPO_A = float(primary_P_O_patch_A - float(primary_base_stats["P_O_base"]))
                    primary_dPrest_A = float(primary_P_rest_patch_A - float(primary_base_stats["P_rest_base"]))
                    primary_dPE_C = float(primary_P_E_patch_C - float(primary_base_stats["P_E_base"]))
                    primary_dPO_C = float(primary_P_O_patch_C - float(primary_base_stats["P_O_base"]))
                    primary_dPrest_C = float(primary_P_rest_patch_C - float(primary_base_stats["P_rest_base"]))

                    primary_d_CA_logPE = float(primary_dlogPE_C - primary_dlogPE_A)
                    primary_d_CA_logPO = float(primary_dlogPO_C - primary_dlogPO_A)

                    sign = float(case.effect_sign)
                    primary_delta_m_from_logodds_A = float(
                        signed_delta_margin_from_logodds(dlogpe=primary_dlogPE_A, dlogpo=primary_dlogPO_A, effect_sign=sign)
                    )
                    primary_delta_m_from_logodds_C = float(
                        signed_delta_margin_from_logodds(dlogpe=primary_dlogPE_C, dlogpo=primary_dlogPO_C, effect_sign=sign)
                    )
                    primary_delta_m_residual_A = float(float(arm_results["A"]["effect"]) - primary_delta_m_from_logodds_A)
                    primary_delta_m_residual_C = float(float(arm_results["C"]["effect"]) - primary_delta_m_from_logodds_C)
                    primary_cancellation_pass = bool(
                        abs(primary_delta_m_residual_A) <= float(args.primary_logodds_residual_tol)
                        and abs(primary_delta_m_residual_C) <= float(args.primary_logodds_residual_tol)
                    )
                    if bool(args.hard_fail_primary_logodds) and not primary_cancellation_pass:
                        raise RuntimeError(
                            "Primary log-odds cancellation residual exceeded tolerance: "
                            f"case={case.case_id}, layer={layer}, "
                            f"resA={primary_delta_m_residual_A:.3e}, resC={primary_delta_m_residual_C:.3e}, "
                            f"tol={float(args.primary_logodds_residual_tol):.3e}"
                        )

                gate_margin_abs_diff = float("nan")
                gate_score_abs_diff_exp = float("nan")
                gate_score_abs_diff_other = float("nan")
                gate_clt_effect_abs_diff_c = float("nan")
                gate_clt_effect_abs_diff_d = float("nan")
                gate_identity_abs_effect_raw = float("nan")
                gate_identity_abs_effect_clt = float("nan")
                gate_margin_pass = False
                gate_score_pass = False
                gate_clt_equiv_pass = False
                gate_identity_pass = False

                if all_success:
                    eff_a = float(arm_results["A"]["effect"])
                    eff_b = float(arm_results["B"]["effect"])
                    gate_margin_abs_diff = abs(eff_a - eff_b)
                    gate_margin_pass = bool(gate_margin_abs_diff < float(args.gate_margin_abs_tol))
                    gate_score_abs_diff_exp = abs(
                        float(arm_results["A"]["delta_score_exp"]) - float(arm_results["B"]["delta_score_exp"])
                    )
                    gate_score_abs_diff_other = abs(
                        float(arm_results["A"]["delta_score_other"]) - float(arm_results["B"]["delta_score_other"])
                    )
                    gate_score_pass = bool(
                        gate_score_abs_diff_exp < float(args.gate_score_abs_tol)
                        and gate_score_abs_diff_other < float(args.gate_score_abs_tol)
                    )
                    gate_clt_effect_abs_diff_c = abs(float(arm_results["C"]["effect"]) - float(arm_results["Cp"]["effect"]))
                    gate_clt_effect_abs_diff_d = abs(float(arm_results["D"]["effect"]) - float(arm_results["Dp"]["effect"]))
                    gate_clt_equiv_pass = bool(
                        gate_clt_effect_abs_diff_c < float(args.gate_clt_equiv_abs_tol)
                        and gate_clt_effect_abs_diff_d < float(args.gate_clt_equiv_abs_tol)
                    )
                    gate_identity_abs_effect_raw = abs(float(arm_results["RI"]["effect"]))
                    gate_identity_abs_effect_clt = abs(float(arm_results["CI"]["effect"]))
                    gate_identity_pass = bool(
                        gate_identity_abs_effect_raw < float(args.gate_identity_abs_tol)
                        and gate_identity_abs_effect_clt < float(args.gate_identity_abs_tol)
                    )

                invariant_all_pass = bool(
                    gate_activation_pass
                    and gate_margin_pass
                    and gate_score_pass
                    and gate_clt_equiv_pass
                    and gate_identity_pass
                )
                analysis_included = bool(all_success and invariant_all_pass)

                row: Dict[str, Any] = dict(row_common)
                row.update(
                    {
                        "layer": int(layer),
                        "base_pred": str(base_pred),
                        "base_margin": float(base_margin),
                        "other_label": str(other_label),
                        "raw_delta_norm": raw_delta_norm,
                        "raw_delta_norm_small": bool(raw_delta_norm_small),
                        "clt_delta_norm_C": clt_delta_norm_c,
                        "clt_delta_norm_D": clt_delta_norm_d,
                        "clt_delta_norm_ratio_C": clt_delta_ratio_c,
                        "clt_delta_norm_ratio_D": clt_delta_ratio_d,
                        "clt_delta_cosine_raw_C": clt_delta_cos_c,
                        "clt_delta_cosine_raw_D": clt_delta_cos_d,
                        "recon_norm_ratio": float(recon_norm_ratio),
                        "active_latent_count": int(active_latent_count),
                        "active_latent_frac": float(active_latent_frac),
                        "fidelity_rel_mse": float(fidelity_rel_mse),
                        "fidelity_rel_l2": float(fidelity_rel_l2),
                        "fidelity_cosine": float(fidelity_cosine),
                        "fidelity_rel_mse_warn": bool(float(fidelity_rel_mse) > float(args.fidelity_rel_mse_warn)),
                        "gate_activation_ratio": float(gate_activation_ratio),
                        "gate_activation_pass": bool(gate_activation_pass),
                        "gate_margin_abs_diff": float(gate_margin_abs_diff),
                        "gate_margin_pass": bool(gate_margin_pass),
                        "gate_score_abs_diff_exp": float(gate_score_abs_diff_exp),
                        "gate_score_abs_diff_other": float(gate_score_abs_diff_other),
                        "gate_score_pass": bool(gate_score_pass),
                        "gate_clt_effect_abs_diff_C": float(gate_clt_effect_abs_diff_c),
                        "gate_clt_effect_abs_diff_D": float(gate_clt_effect_abs_diff_d),
                        "gate_clt_equiv_pass": bool(gate_clt_equiv_pass),
                        "gate_identity_abs_effect_raw": float(gate_identity_abs_effect_raw),
                        "gate_identity_abs_effect_clt": float(gate_identity_abs_effect_clt),
                        "gate_identity_pass": bool(gate_identity_pass),
                        "invariant_all_pass": bool(invariant_all_pass),
                        "all_arms_success": bool(all_success),
                        "analysis_included": bool(analysis_included),
                        "primary_labels_binary": bool(primary_labels_binary),
                        "primary_label_aggregation_logmeanexp": bool(primary_label_agg_logmeanexp),
                        "primary_candidate_count_expected": int(len(expected_tokens)),
                        "primary_candidate_count_other": int(len(other_tokens)),
                        "primary_expected_all_single": bool(expected_all_single),
                        "primary_other_all_single": bool(other_all_single),
                        "primary_no_duplicates_expected": bool(expected_no_duplicates),
                        "primary_no_duplicates_other": bool(other_no_duplicates),
                        "primary_token_sets_disjoint": bool(token_sets_disjoint),
                        "primary_candidate_count_equal": bool(candidate_count_equal),
                        "primary_same_scored_position": bool(same_scored_position),
                        "primary_static_applicable": bool(primary_static_applicable),
                        "primary_logodds_applicable": bool(primary_logodds_applicable),
                        "primary_logodds_applicable_flag": float(1.0 if primary_logodds_applicable else 0.0),
                        "primary_logodds_reason": str(primary_logodds_reason),
                        "P_E_base": float(primary_base_stats["P_E_base"]),
                        "P_O_base": float(primary_base_stats["P_O_base"]),
                        "P_rest_base": float(primary_base_stats["P_rest_base"]),
                        "P_E_patch_A": float(primary_P_E_patch_A),
                        "P_O_patch_A": float(primary_P_O_patch_A),
                        "P_rest_patch_A": float(primary_P_rest_patch_A),
                        "P_E_patch_C": float(primary_P_E_patch_C),
                        "P_O_patch_C": float(primary_P_O_patch_C),
                        "P_rest_patch_C": float(primary_P_rest_patch_C),
                        "dlogPE_A": float(primary_dlogPE_A),
                        "dlogPO_A": float(primary_dlogPO_A),
                        "dlogPE_C": float(primary_dlogPE_C),
                        "dlogPO_C": float(primary_dlogPO_C),
                        "dPE_A": float(primary_dPE_A),
                        "dPO_A": float(primary_dPO_A),
                        "dPrest_A": float(primary_dPrest_A),
                        "dPE_C": float(primary_dPE_C),
                        "dPO_C": float(primary_dPO_C),
                        "dPrest_C": float(primary_dPrest_C),
                        "d_CA_logPE": float(primary_d_CA_logPE),
                        "d_CA_logPO": float(primary_d_CA_logPO),
                        "primary_delta_m_from_logodds_A": float(primary_delta_m_from_logodds_A),
                        "primary_delta_m_from_logodds_C": float(primary_delta_m_from_logodds_C),
                        "primary_delta_m_residual_A": float(primary_delta_m_residual_A),
                        "primary_delta_m_residual_C": float(primary_delta_m_residual_C),
                        "primary_base_logodds_residual": float(primary_base_stats["base_logodds_residual"]),
                        "primary_cancellation_pass": bool(primary_cancellation_pass),
                        "primary_cancellation_pass_flag": float(1.0 if primary_cancellation_pass else 0.0),
                    }
                )

                for arm in required_arms:
                    arm_ok = bool(arm_results.get(arm, {}).get("success", False))
                    row[f"success_{arm}"] = arm_ok
                    row[f"error_{arm}"] = "" if arm_ok else str(arm_results.get(arm, {}).get("error", ""))
                    if not arm_ok:
                        row[f"effect_{arm}"] = float("nan")
                        row[f"delta_score_exp_{arm}"] = float("nan")
                        row[f"delta_score_other_{arm}"] = float("nan")
                        continue

                    row[f"effect_{arm}"] = float(arm_results[arm]["effect"])
                    row[f"delta_score_exp_{arm}"] = float(arm_results[arm]["delta_score_exp"])
                    row[f"delta_score_other_{arm}"] = float(arm_results[arm]["delta_score_other"])

                    for scope in ("exp", "other"):
                        decomp = arm_results[arm][f"decomp_{scope}"]
                        for k, v in decomp.items():
                            row[f"decomp_{scope}_{k}_{arm}"] = float(v)

                if all_success:
                    row["d_BA"] = float(row["effect_B"] - row["effect_A"])
                    row["d_CA"] = float(row["effect_C"] - row["effect_A"])
                    row["d_DA"] = float(row["effect_D"] - row["effect_A"])
                    row["d_CCp"] = float(row["effect_C"] - row["effect_Cp"])
                    row["d_DDp"] = float(row["effect_D"] - row["effect_Dp"])
                    row["sign_agree_BA"] = float(_sign(float(row["effect_B"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_CA"] = float(_sign(float(row["effect_C"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_DA"] = float(_sign(float(row["effect_D"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_CCp"] = float(_sign(float(row["effect_C"])) == _sign(float(row["effect_Cp"])))
                    row["sign_agree_DDp"] = float(_sign(float(row["effect_D"])) == _sign(float(row["effect_Dp"])))
                else:
                    row["d_BA"] = float("nan")
                    row["d_CA"] = float("nan")
                    row["d_DA"] = float("nan")
                    row["d_CCp"] = float("nan")
                    row["d_DDp"] = float("nan")
                    row["sign_agree_BA"] = float("nan")
                    row["sign_agree_CA"] = float("nan")
                    row["sign_agree_DA"] = float("nan")
                    row["sign_agree_CCp"] = float("nan")
                    row["sign_agree_DDp"] = float("nan")

                rows.append(row)

    _write_csv(rows, out_csv_path)

    rows_success = [r for r in rows if bool(r.get("all_arms_success", False))]
    rows_analysis = [r for r in rows if bool(r.get("analysis_included", False))]
    rows_invariant_fail = [r for r in rows_success if not bool(r.get("invariant_all_pass", False))]

    per_layer: List[Dict[str, Any]] = []
    for layer in layers:
        per_layer.append(
            _aggregate_layer(
                rows=rows_analysis,
                layer=int(layer),
                n_bootstrap=int(args.bootstrap_n),
                ci=float(args.ci),
                seed=int(args.bootstrap_seed),
                ratio_den_eps=float(args.ratio_den_eps),
            )
        )

    dataset_path = str(args.cf_path) if str(task) == "cf" else str(args.coh_path)
    summary: Dict[str, Any] = {
        "task": str(task),
        "model_name_or_path": str(args.model_name_or_path),
        "dataset_path": str(dataset_path),
        "n_items_loaded": int(n_items_loaded),
        "n_cases_protocol": int(len(cases)),
        "protocol_skip_reason_counts": dict(protocol_skip_counts),
        "n_layers_model": int(n_layers_model),
        "layers": [int(l) for l in layers],
        "run_config": {
            "normalize_by_length": bool(args.normalize_by_length),
            "seed": int(args.seed),
            "bootstrap_n": int(args.bootstrap_n),
            "ci": float(args.ci),
            "bootstrap_seed": int(args.bootstrap_seed),
            "gate_activation_ratio_tol": float(args.gate_activation_ratio_tol),
            "gate_margin_abs_tol": float(args.gate_margin_abs_tol),
            "gate_score_abs_tol": float(args.gate_score_abs_tol),
            "gate_clt_equiv_abs_tol": float(args.gate_clt_equiv_abs_tol),
            "gate_identity_abs_tol": float(args.gate_identity_abs_tol),
            "hard_fail_invariant": bool(args.hard_fail_invariant),
            "hard_fail_primary_logodds": bool(args.hard_fail_primary_logodds),
            "primary_logodds_residual_tol": float(args.primary_logodds_residual_tol),
            "ratio_den_eps": float(args.ratio_den_eps),
            "raw_delta_norm_eps": float(args.raw_delta_norm_eps),
            "fidelity_rel_mse_warn": float(args.fidelity_rel_mse_warn),
            "clt_decode_strategies": ["delta_1decode", "safe_2decode"],
            "clt_dtype_policy": str(args.clt_dtype_policy),
            "clt_eps_active": float(args.clt_eps_active),
        },
        "counts": {
            "n_rows_total": int(len(rows)),
            "n_rows_all_arms_success": int(len(rows_success)),
            "n_rows_analysis_included": int(len(rows_analysis)),
            "n_pairs_analysis_included": int(len({str(r['pair_id']) for r in rows_analysis})),
            "n_invariant_fail_rows": int(len(rows_invariant_fail)),
        },
        "site_equivalence": {
            "raw_site": "resid_post (decoder block output hook)",
            "clt_site_mode": "same_site_v1",
            "clt_sites": {
                "encode": sorted({str(bundles[int(l)].meta.encode_site) for l in layers}),
                "decode": sorted({str(bundles[int(l)].meta.decode_site) for l in layers}),
                "writeback": sorted({str(bundles[int(l)].meta.writeback_site) for l in layers}),
            },
        },
        "arm_fail_counts": {k: int(v) for k, v in sorted(fail_counts.items())},
        "arm_fail_reasons": {k: dict(sorted(v.items())) for k, v in sorted(fail_reasons.items())},
        "per_layer": per_layer,
    }

    _ensure_dir(out_json_path)
    out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.hard_fail_invariant) and rows_invariant_fail:
        sample = rows_invariant_fail[:3]
        sample_msg = "; ".join(
            f"case={r.get('case_id')} layer={r.get('layer')} "
            f"act={r.get('gate_activation_ratio')} margin={r.get('gate_margin_abs_diff')} "
            f"score_exp={r.get('gate_score_abs_diff_exp')} score_other={r.get('gate_score_abs_diff_other')} "
            f"clt_c={r.get('gate_clt_effect_abs_diff_C')} clt_d={r.get('gate_clt_effect_abs_diff_D')} "
            f"id_raw={r.get('gate_identity_abs_effect_raw')} id_clt={r.get('gate_identity_abs_effect_clt')}"
            for r in sample
        )
        raise RuntimeError(
            f"A≈B invariant failed on {len(rows_invariant_fail)} row(s). "
            f"See {out_json_path} and {out_csv_path}. "
            f"Examples: {sample_msg}"
        )

    print(
        json.dumps(
            {
                "task": str(task),
                "out_csv": str(out_csv_path),
                "out_json": str(out_json_path),
                "n_rows_total": int(len(rows)),
                "n_rows_analysis_included": int(len(rows_analysis)),
            },
            ensure_ascii=False,
        )
    )


def main_for_task(task: Optional[str] = None) -> None:
    parser = build_arg_parser(task_default=task)
    args = parser.parse_args()
    if task is not None:
        setattr(args, "task", str(task))
    run(args)


def main() -> None:
    main_for_task(None)


if __name__ == "__main__":
    main()
