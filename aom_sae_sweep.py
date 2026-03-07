from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from aom.data.loaders import load_disamb_pairs
from aom.interventions.activation_patching import get_num_layers
from aom.interventions.sae_adapter import SAEInputTransform, SAEPatchConfig
from aom.interventions.sae_loader import calibrate_sae_scale, load_gemma_scope_sae
from aom.interventions.sae_sweep import run_threshold_sweep
from aom.models.loader import load_causal_lm
from aom.utils import configure_logprob_computation, get_best_device, set_seed


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


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run SAE threshold sweep with AoM-DISAMB scoring + controls.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--sae_repo", type=str, required=True, help="HF repo id or local path to Gemma Scope SAE bundle.")
    p.add_argument("--layer", type=int, required=True, help="Target layer for sweep.")
    p.add_argument("--width", type=str, default="16k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--l0_target", type=int, default=None)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--thresholds", type=str, default="0.0,0.1,0.5,1.0,2.0,5.0")
    p.add_argument("--scale_mode", type=str, default="quantile", choices=["quantile", "max"])
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--token_buffer", type=int, default=2)
    p.add_argument("--off_target_layer_offset", type=int, default=2)

    p.add_argument("--scale", type=float, default=1.0, help="SAE input scale (used if --calibrate is off).")
    p.add_argument("--calibrate", action="store_true", help="Calibrate per-layer scale via recon MSE grid search.")
    p.add_argument(
        "--calibration_scales",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,10.0",
        help="Comma-separated scales to try when --calibrate is set.",
    )

    p.add_argument(
        "--decode_strategy",
        type=str,
        default="delta_1decode",
        choices=["safe_2decode", "delta_1decode"],
        help="Decode strategy for SAE delta injection.",
    )
    p.add_argument("--dtype_policy", type=str, default="sae", choices=["sae", "model"])
    p.add_argument("--eps_active", type=float, default=1e-6)

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_length_norm", action="store_true")

    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument(
        "--strict_finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast on NaN/Inf during logprob scoring.",
    )

    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def _parse_floats(arg: str) -> List[float]:
    out: List[float] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("no floats provided")
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    configure_logprob_computation(
        logprobs_dtype=getattr(torch, str(args.logprobs_dtype)),
        strict_finite=bool(args.strict_finite),
    )

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])

    items = load_disamb_pairs(str(args.disamb_path)) if args.disamb_path else []
    if not items:
        raise ValueError("disamb dataset is empty; provide --disamb_path")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )

    n_layers = int(get_num_layers(loaded.model))
    target_layer = int(args.layer)
    if target_layer < 0 or target_layer >= n_layers:
        raise ValueError(f"--layer {target_layer} out of range [0, {n_layers})")

    layers_to_load = {target_layer}
    off = int(args.off_target_layer_offset)
    for control_layer in (target_layer - off, target_layer + off):
        if 0 <= int(control_layer) < n_layers:
            layers_to_load.add(int(control_layer))

    sae_by_layer = {}
    for l in sorted(layers_to_load):
        sae, _meta = load_gemma_scope_sae(
            str(args.sae_repo),
            layer=int(l),
            width=str(args.width),
            run_name=args.run_name,
            l0_target=args.l0_target,
            device=str(device),
            dtype=str(args.torch_dtype or "auto"),
            local_files_only=bool(args.local_files_only),
        )
        sae_by_layer[int(l)] = sae

    transform_by_layer: Dict[int, SAEInputTransform] = {}
    calibration: Dict[str, Any] = {}
    if bool(args.calibrate):
        scales = tuple(_parse_floats(str(args.calibration_scales)))
        sample_text = items[0].a.prompt
        enc = loaded.tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        best_by_layer: Dict[int, float] = {}
        mse_by_layer: Dict[int, Dict[float, float]] = {}
        for l in sorted(layers_to_load):
            best_s, mse_by_s = calibrate_sae_scale(
                model=loaded.model,
                sae=sae_by_layer[int(l)],
                layer=int(l),
                scales=scales,
                calibration_input_ids=input_ids,
                attention_mask=attention_mask,
                device=device,
            )
            best_by_layer[int(l)] = float(best_s)
            mse_by_layer[int(l)] = dict(mse_by_s)
            transform_by_layer[int(l)] = SAEInputTransform(scale=float(best_s))
        calibration = {"scales": list(scales), "best_scale_by_layer": best_by_layer, "mse_by_layer": mse_by_layer}
    else:
        for l in sorted(layers_to_load):
            transform_by_layer[int(l)] = SAEInputTransform(scale=float(args.scale))

    sae_cfg = SAEPatchConfig(
        decode_strategy=str(args.decode_strategy),
        eps_active=float(args.eps_active),
        dtype_policy=str(args.dtype_policy),
    )

    thresholds = _parse_floats(str(args.thresholds))
    baseline, points = run_threshold_sweep(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        device=device,
        target_layer=int(target_layer),
        sae_by_layer=sae_by_layer,
        transform_by_layer=transform_by_layer,
        thresholds=thresholds,
        scale_mode=str(args.scale_mode),
        quantile=float(args.quantile),
        token_buffer=int(args.token_buffer),
        off_target_layer_offset=int(args.off_target_layer_offset),
        config=sae_cfg,
        normalize_by_length=not bool(args.no_length_norm),
    )

    rows: List[Dict[str, Any]] = []
    for pt in points:
        row = pt.to_row()
        row.update(
            {
                "model": str(args.model_name_or_path),
                "sae_repo": str(args.sae_repo),
                "sae_width": str(args.width),
                "sae_run_name": str(args.run_name or ""),
                "sae_l0_target": str(args.l0_target or ""),
                "target_layer": int(target_layer),
                "baseline_accuracy": float(baseline.accuracy),
                "baseline_mean_expected_nll": float(baseline.mean_expected_nll),
                "baseline_n_pairs": int(baseline.n_pairs),
                "decode_strategy": str(args.decode_strategy),
                "dtype_policy": str(args.dtype_policy),
                "eps_active": float(args.eps_active),
                "scale_mode": str(args.scale_mode),
                "quantile": float(args.quantile),
                "token_buffer": int(args.token_buffer),
                "off_target_layer_offset": int(args.off_target_layer_offset),
                "seed": int(args.seed),
                "git_commit": _git_commit(),
            }
        )
        rows.append(row)

    manifest: Dict[str, Any] = {
        "model_id": str(args.model_name_or_path),
        "sae_repo": str(args.sae_repo),
        "sae_width": str(args.width),
        "layers_loaded": sorted(int(x) for x in layers_to_load),
        "target_layer": int(target_layer),
        "thresholds": thresholds,
        "scale_mode": str(args.scale_mode),
        "quantile": float(args.quantile),
        "token_buffer": int(args.token_buffer),
        "off_target_layer_offset": int(args.off_target_layer_offset),
        "scale": float(args.scale),
        "calibrate": bool(args.calibrate),
        "sae_patch_config": asdict(sae_cfg),
        "baseline": asdict(baseline),
        "points": [pt.to_row() for pt in points],
        "git_commit": _git_commit(),
    }
    if calibration:
        manifest["calibration"] = calibration

    if args.out_csv:
        write_csv(rows, str(args.out_csv))
    if args.out_json:
        p = Path(str(args.out_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
