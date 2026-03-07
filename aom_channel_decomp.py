#!/usr/bin/env python3
"""
SAE Channel Decomposition CLI.

Decomposes CPT signal into SAE-reconstructable component vs reconstruction residual
using a 2x2 factorial design.

Key identity (faithful decomposition): delta_r = delta_S + delta_C

Primary analysis (default, norm_matching="none"):
  Uses unscaled deltas so decomposition is valid.
  Interaction term is interpretable as compatibility/synergy.

Secondary analysis (norm_matching="energy_matched_independent"):
  Scales delta_S and delta_C independently to match ||delta_r||.
  Tests "can component X reproduce effect with equal L2 budget?"
  Interaction term is NOT interpretable as synergy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from aom.data.loaders import load_disamb_pairs
from aom.interventions.sae_adapter import SAEInputTransform
from aom.interventions.sae_loader import load_gemma_scope_sae
from aom.metrics.sae_decomposition import compute_cpt_channel_decomposition
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, set_seed


def _git_commit() -> str:
    try:
        import subprocess
        root = Path(__file__).resolve().parent
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAE channel decomposition for CPT experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python aom_channel_decomp.py \\
    --model_name_or_path google/gemma-2-2b \\
    --sae_repo google/gemma-scope-2b-pt-res \\
    --layers 15,18

Output interpretation (with norm_matching="none", the default):
  - If recon_main ≈ total and resid_main ≈ 0, interaction ≈ 0:
    "CPT signal is largely mediated by SAE-reconstructable component (this SAE)"
  - If resid_main ≈ total and recon_main ≈ 0, interaction ≈ 0:
    "CPT signal is in reconstruction residual; this SAE doesn't capture causal directions"
  - If interaction is large:
    "Effect requires compatibility between components (non-separable)"

IMPORTANT: Interaction is only interpretable as compatibility/synergy when using
faithful deltas (norm_matching="none"). With energy_matched_independent, the
interaction term is a derived artifact, not a meaningful synergy measure.
""",
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--sae_repo", type=str, required=True, help="HF repo or local path to SAE.")
    p.add_argument("--layers", type=str, required=True, help="Comma-separated layer indices (e.g., '15,18').")
    p.add_argument("--width", type=str, default="16k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--l0_target", type=int, default=None)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--scale", type=float, default=1.0, help="SAE input scale.")
    p.add_argument(
        "--norm_matching",
        type=str,
        default="none",
        choices=["none", "energy_matched_independent"],
        help=(
            "Norm matching strategy. 'none' (default): faithful decomposition where "
            "delta_r = delta_S + delta_C. 'energy_matched_independent': scales delta_S "
            "and delta_C to match ||delta_r|| independently (robustness check, breaks "
            "decomposition validity)."
        ),
    )
    p.add_argument("--no_random_donor_sham", action="store_true", help="Disable random-donor sham control.")
    p.add_argument("--disamb_path", type=str, default=str(Path(__file__).resolve().parent / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--max_pairs", type=int, default=None, help="Limit number of pairs (for quick testing).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--out_json", type=str, default="", help="Path to write JSON results.")
    return p.parse_args()


def _parse_layers(arg: str) -> List[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    layers = _parse_layers(str(args.layers))
    if not layers:
        raise ValueError("No layers specified")

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device(args.device)

    disamb_items = load_disamb_pairs(str(args.disamb_path))
    if not disamb_items:
        raise ValueError("No disamb pairs loaded")

    if args.max_pairs is not None and args.max_pairs > 0:
        disamb_items = disamb_items[: int(args.max_pairs)]

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )

    transform = SAEInputTransform(scale=float(args.scale))

    results: Dict[str, Any] = {
        "model_id": str(args.model_name_or_path),
        "sae_repo": str(args.sae_repo),
        "sae_width": str(args.width),
        "scale": float(args.scale),
        "norm_matching": str(args.norm_matching),
        "n_pairs": len(disamb_items),
        "git_commit": _git_commit(),
        "layers": {},
    }

    for layer in layers:
        print(f"Processing layer {layer}...", flush=True)

        sae, sae_meta = load_gemma_scope_sae(
            str(args.sae_repo),
            layer=int(layer),
            width=str(args.width),
            run_name=args.run_name,
            l0_target=args.l0_target,
            device=str(device),
            dtype=str(args.torch_dtype or "float32"),
            local_files_only=bool(args.local_files_only),
        )

        layer_result = compute_cpt_channel_decomposition(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            items=disamb_items,
            device=device,
            layer=int(layer),
            sae=sae,
            transform=transform,
            norm_matching=args.norm_matching,
            include_random_donor_sham=not args.no_random_donor_sham,
            bootstrap_n=int(args.bootstrap_n),
        )

        layer_result["sae_run_name"] = str(sae_meta.run_name)
        results["layers"][str(layer)] = layer_result

        # Print factorial decomposition summary
        total = layer_result["total_effect"]
        recon = layer_result["recon_main_effect"]
        resid = layer_result["resid_main_effect"]
        inter = layer_result["interaction"]
        norms = layer_result["norm_stats"]

        print(f"  Layer {layer} (2x2 factorial, norm_matching={args.norm_matching}):")
        print(f"    Total effect:      {total['mean']:+.4f} [{total['ci_low']:+.4f}, {total['ci_high']:+.4f}]")
        print(f"    Recon main effect: {recon['mean']:+.4f} [{recon['ci_low']:+.4f}, {recon['ci_high']:+.4f}]")
        print(f"    Resid main effect: {resid['mean']:+.4f} [{resid['ci_low']:+.4f}, {resid['ci_high']:+.4f}]")
        print(f"    Interaction:       {inter['mean']:+.4f} [{inter['ci_low']:+.4f}, {inter['ci_high']:+.4f}]")

        # Norm statistics
        print(f"    Norms (median): ||delta_r||={norms['delta_r_median']:.2f}, "
              f"||delta_S||={norms['delta_S_median']:.2f}, ||delta_C||={norms['delta_C_median']:.2f}")
        print(f"    Additivity error: {norms['additivity_error_median']:.2e} (should be ~0)")

        if "random_donor_sham" in layer_result:
            rand = layer_result["random_donor_sham"]
            print(f"    Random donor sham: {rand['mean_effect']:+.4f} [{rand['ci_low']:+.4f}, {rand['ci_high']:+.4f}] (n={rand['n_samples']})")

        # Ablation controls
        if "ablation_controls" in layer_result:
            abl = layer_result["ablation_controls"]
            if "E_precontext" in abl:
                pc = abl["E_precontext"]
                if pc["n_samples"] > 0:
                    print(f"    Pre-context embed: {pc['mean_effect']:+.4f} [{pc['ci_low']:+.4f}, {pc['ci_high']:+.4f}] (n={pc['n_samples']})")
            if "F_random_vec" in abl:
                rv = abl["F_random_vec"]
                if rv["n_samples"] > 0:
                    print(f"    Random vector:     {rv['mean_effect']:+.4f} [{rv['ci_low']:+.4f}, {rv['ci_high']:+.4f}] (n={rv['n_samples']})")

        # Interpretation hint (only valid for faithful decomposition)
        if args.norm_matching == "none":
            total_ci_excludes_zero = (total['ci_low'] > 0) or (total['ci_high'] < 0)
            if total_ci_excludes_zero and abs(total['mean']) > 0.01:
                recon_frac = recon['mean'] / total['mean']
                resid_frac = resid['mean'] / total['mean']
                inter_frac = inter['mean'] / total['mean']
                print(f"    Fractions of total: recon={recon_frac:.1%}, resid={resid_frac:.1%}, interaction={inter_frac:.1%}")
        else:
            print(f"    (Fractions not shown: interaction not interpretable with {args.norm_matching})")

    print("\n" + json.dumps(results, indent=2), flush=True)

    if args.out_json:
        p = Path(str(args.out_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
