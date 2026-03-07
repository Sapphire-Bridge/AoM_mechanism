from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from aom.data.loaders import load_disamb_pairs
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
from aom.interventions.clt_loader import calibrate_clt_scale, list_clt_runs, load_clt
from aom.interventions.clt_patch import IdentityLatentPolicy, forward_with_clt_latent_patching_span
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, set_seed


def _git_commit() -> str:
    try:
        import subprocess

        root = Path(__file__).resolve().parent
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
    except Exception:
        return ""


def _parse_scales(arg: str) -> List[float]:
    out: List[float] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("no calibration scales provided")
    return out


def _collect_disamb_prompts(disamb_items: Sequence[Any], n_examples: int) -> List[str]:
    prompts: List[str] = []
    for it in disamb_items:
        prompts.append(str(it.a.prompt))
        prompts.append(str(it.b.prompt))
        if len(prompts) >= int(n_examples):
            break
    return prompts[: int(n_examples)]


@dataclass(frozen=True)
class IdentityDriftResult:
    n_prompts: int
    mean_abs_logit_delta: float
    median_abs_logit_delta: float
    max_abs_logit_delta: float
    passed: bool


@dataclass(frozen=True)
class ReconTelemetryResult:
    n_prompts: int
    recon_mse_mean: float
    recon_mse_median: float
    recon_rel_l2_mean: float
    recon_rel_l2_median: float
    recon_cos_mean: float
    recon_cos_median: float


def _infer_clt_device_dtype(clt) -> tuple[torch.device, torch.dtype]:
    if isinstance(clt, torch.nn.Module):
        p = next(clt.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    w_dec = getattr(clt, "W_dec", None)
    if isinstance(w_dec, torch.Tensor):
        return w_dec.device, w_dec.dtype
    return torch.device("cpu"), torch.float32


@torch.no_grad()
def compute_identity_logit_drift(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    clt,
    layer: int,
    transform: CLTInputTransform,
    config: CLTPatchConfig,
    device: torch.device,
    tolerance: float,
) -> IdentityDriftResult:
    if not prompts:
        raise ValueError("prompts must be non-empty")

    mean_deltas: List[float] = []
    max_deltas: List[float] = []
    for prompt in prompts:
        enc = tokenizer(str(prompt), return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        base_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        token_indices = tuple(range(int(input_ids.size(1))))
        patched_logits = forward_with_clt_latent_patching_span(
            model,
            input_ids=input_ids,
            site=PatchSpanSite(layer=int(layer), token_indices=token_indices),
            clt=clt,
            policy=IdentityLatentPolicy(),
            transform=transform,
            config=config,
            attention_mask=attention_mask,
        )
        diff = (patched_logits - base_logits).abs()
        mean_deltas.append(float(diff.mean().item()))
        max_deltas.append(float(diff.max().item()))

    max_abs = float(max(max_deltas)) if max_deltas else 0.0
    return IdentityDriftResult(
        n_prompts=len(prompts),
        mean_abs_logit_delta=float(mean(mean_deltas)) if mean_deltas else 0.0,
        median_abs_logit_delta=float(median(mean_deltas)) if mean_deltas else 0.0,
        max_abs_logit_delta=max_abs,
        passed=bool(max_abs <= float(tolerance)),
    )


@torch.no_grad()
def compute_reconstruction_telemetry(
    *,
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    clt,
    layer: int,
    transform: CLTInputTransform,
    device: torch.device,
) -> ReconTelemetryResult:
    if not prompts:
        raise ValueError("prompts must be non-empty")

    clt_device, clt_dtype = _infer_clt_device_dtype(clt)
    recon_mse_vals: List[float] = []
    rel_l2_vals: List[float] = []
    cos_vals: List[float] = []

    for prompt in prompts:
        enc = tokenizer(str(prompt), return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        hidden_by_layer = get_block_outputs(
            model,
            input_ids,
            layers=[int(layer)],
            attention_mask=attention_mask,
        )
        hidden = hidden_by_layer[int(layer)]
        hidden_work = hidden.to(device=clt_device, dtype=clt_dtype)

        z = clt.encode(transform.forward(hidden_work))
        recon = transform.inverse(clt.decode(z))
        if recon.shape != hidden_work.shape:
            raise RuntimeError(
                f"CLT decode shape mismatch at layer {int(layer)}: recon={tuple(recon.shape)}, "
                f"hidden={tuple(hidden_work.shape)}"
            )

        err = recon - hidden_work
        mse = float(torch.mean(err**2).item())
        rel_l2 = float(torch.norm(err).item() / (torch.norm(hidden_work).item() + 1e-12))
        cos = float(F.cosine_similarity(hidden_work.reshape(1, -1), recon.reshape(1, -1), dim=1).item())
        recon_mse_vals.append(mse)
        rel_l2_vals.append(rel_l2)
        cos_vals.append(cos)

    return ReconTelemetryResult(
        n_prompts=len(prompts),
        recon_mse_mean=float(mean(recon_mse_vals)),
        recon_mse_median=float(median(recon_mse_vals)),
        recon_rel_l2_mean=float(mean(rel_l2_vals)),
        recon_rel_l2_median=float(median(rel_l2_vals)),
        recon_cos_mean=float(mean(cos_vals)),
        recon_cos_median=float(median(cos_vals)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLT preflight: calibration + recon telemetry + identity drift.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--clt_repo", type=str, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--width", type=str, default="16k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--l0_target", type=int, default=None)
    p.add_argument("--list_runs", action="store_true", help="List available run names for this layer/width and exit.")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--scale", type=float, default=1.0, help="CLT input scale (used if --calibrate is off).")
    p.add_argument("--calibrate", action="store_true", help="Calibrate scale via reconstruction-MSE grid search.")
    p.add_argument(
        "--calibration_scales",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,10.0",
        help="Comma-separated scales to try when --calibrate is set.",
    )

    p.add_argument("--decode_strategy", type=str, default="safe_2decode", choices=["safe_2decode", "delta_1decode"])
    p.add_argument("--dtype_policy", type=str, default="clt", choices=["clt", "model"])
    p.add_argument("--eps_active", type=float, default=1e-6)
    p.add_argument("--identity_tolerance", type=float, default=1e-5)

    p.add_argument(
        "--disamb_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "disamb_pairs.jsonl"),
    )
    p.add_argument("--n_examples", type=int, default=8, help="Number of prompts (A/B sides combined) for checks.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default="", help="Optional path to write JSON result.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if bool(args.list_runs):
        runs = list_clt_runs(
            str(args.clt_repo),
            layer=int(args.layer),
            width=str(args.width),
            local_files_only=bool(args.local_files_only),
        )
        out: Dict[str, Any] = {
            "clt_repo": str(args.clt_repo),
            "layer": int(args.layer),
            "width": str(args.width),
            "runs": list(runs),
        }
        if args.l0_target is not None:
            tag = f"average_l0_{int(args.l0_target)}"
            out["runs_matching_l0_target"] = [r for r in runs if tag in r]
        print(json.dumps(out, indent=2, sort_keys=True), flush=True)
        return

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])

    disamb_items = load_disamb_pairs(str(args.disamb_path)) if args.disamb_path else []
    if not disamb_items:
        raise ValueError("disamb dataset is empty; provide --disamb_path")
    prompts = _collect_disamb_prompts(disamb_items, int(args.n_examples))
    if not prompts:
        raise ValueError("no prompts selected for checking; increase --n_examples")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )
    clt, clt_meta = load_clt(
        str(args.clt_repo),
        layer=int(args.layer),
        width=str(args.width),
        run_name=args.run_name,
        l0_target=args.l0_target,
        device=str(device),
        dtype=str(args.torch_dtype or "float32"),
        local_files_only=bool(args.local_files_only),
    )

    scale = float(args.scale)
    calibration: Dict[str, Any] = {}
    if bool(args.calibrate):
        scales = tuple(_parse_scales(str(args.calibration_scales)))
        sample_enc = loaded.tokenizer(prompts[0], return_tensors="pt", add_special_tokens=False)
        calibration_input_ids = sample_enc["input_ids"].to(device)
        sample_attention_mask = sample_enc.get("attention_mask", None)
        if sample_attention_mask is not None:
            sample_attention_mask = sample_attention_mask.to(device)
        scale, mse_by_scale = calibrate_clt_scale(
            model=loaded.model,
            clt=clt,
            layer=int(args.layer),
            scales=scales,
            calibration_input_ids=calibration_input_ids,
            attention_mask=sample_attention_mask,
            device=device,
        )
        calibration = {"scales": list(scales), "mse_by_scale": mse_by_scale}

    transform = CLTInputTransform(scale=float(scale))
    patch_cfg = CLTPatchConfig(
        decode_strategy=str(args.decode_strategy),
        dtype_policy=str(args.dtype_policy),
        eps_active=float(args.eps_active),
    )

    identity = compute_identity_logit_drift(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompts=prompts,
        clt=clt,
        layer=int(args.layer),
        transform=transform,
        config=patch_cfg,
        device=device,
        tolerance=float(args.identity_tolerance),
    )
    recon = compute_reconstruction_telemetry(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        prompts=prompts,
        clt=clt,
        layer=int(args.layer),
        transform=transform,
        device=device,
    )

    out: Dict[str, Any] = {
        "model_id": str(args.model_name_or_path),
        "clt_repo": str(args.clt_repo),
        "clt_layer": int(args.layer),
        "clt_width": str(args.width),
        "clt_run_name": str(clt_meta.run_name),
        "clt_site_mode": str(clt_meta.site_mode),
        "clt_encode_site": str(clt_meta.encode_site),
        "clt_decode_site": str(clt_meta.decode_site),
        "clt_writeback_site": str(clt_meta.writeback_site),
        "calibrated_scale": float(scale),
        "clt_patch_config": asdict(patch_cfg),
        "identity": asdict(identity),
        "reconstruction": asdict(recon),
        "n_prompts": int(len(prompts)),
        "git_commit": _git_commit(),
    }
    if calibration:
        out["calibration"] = calibration

    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    if args.out_json:
        p = Path(str(args.out_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
