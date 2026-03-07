from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from aom.data.loaders import load_disamb_pairs
from aom.interventions.sae_loader import calibrate_sae_scale, list_gemma_scope_runs, load_gemma_scope_sae
from aom.interventions.sae_sterility import check_sae_sterility
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAE sterility (roundtrip) check on AoM-DISAMB.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--sae_repo", type=str, required=True, help="HF repo id or local path to Gemma Scope SAE bundle.")
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

    p.add_argument("--scale", type=float, default=1.0, help="SAE input scale (used if --calibrate is off).")
    p.add_argument("--calibrate", action="store_true", help="Calibrate scale via recon MSE grid search.")
    p.add_argument(
        "--calibration_scales",
        type=str,
        default="0.1,0.5,1.0,2.0,5.0,10.0",
        help="Comma-separated scales to try when --calibrate is set.",
    )
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--kl_tolerance", type=float, default=0.1, help="Mean per-token KL(p_base||p_roundtrip) threshold in nats.")
    p.add_argument("--disamb_path", type=str, default=str(Path(__file__).resolve().parent / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default="", help="Optional path to write a JSON result/manifest.")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if bool(args.list_runs):
        runs = list_gemma_scope_runs(
            str(args.sae_repo),
            layer=int(args.layer),
            width=str(args.width),
            local_files_only=bool(args.local_files_only),
        )
        out: Dict[str, Any] = {
            "sae_repo": str(args.sae_repo),
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

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )

    sae, sae_meta = load_gemma_scope_sae(
        str(args.sae_repo),
        layer=int(args.layer),
        width=str(args.width),
        run_name=args.run_name,
        l0_target=args.l0_target,
        device=str(device),
        dtype=str(args.torch_dtype or "auto"),
        local_files_only=bool(args.local_files_only),
    )

    scale = float(args.scale)
    calibration: Dict[str, Any] = {}
    if bool(args.calibrate):
        scales = tuple(_parse_scales(str(args.calibration_scales)))
        # Use first DISAMB prompt as a cheap calibration sample.
        sample_text = disamb_items[0].a.prompt
        enc = loaded.tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        scale, mse_by_scale = calibrate_sae_scale(
            model=loaded.model,
            sae=sae,
            layer=int(args.layer),
            scales=scales,
            calibration_input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
        )
        calibration = {"scales": list(scales), "mse_by_scale": mse_by_scale}

    ster = check_sae_sterility(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        sae=sae,
        layer=int(args.layer),
        scale=float(scale),
        disamb_items=disamb_items,
        device=device,
        tolerance=float(args.tolerance),
        kl_tolerance=float(args.kl_tolerance),
    )

    out: Dict[str, Any] = {
        "model_id": str(args.model_name_or_path),
        "sae_repo": str(args.sae_repo),
        "sae_layer": int(args.layer),
        "sae_width": str(args.width),
        "sae_run_name": str(sae_meta.run_name),
        "calibrated_scale": float(scale),
        "sterility": asdict(ster),
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
