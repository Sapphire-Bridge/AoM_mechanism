#!/usr/bin/env python3
"""
Authority-disambiguation "language game" experiment runner.

Runs:
  - Raw CPT-style activation patching at substring-defined sites
  - Fixed-layer routing criterion at ℓ* (quarter-depth by default)
  - Optional SAE subset-copy recovery for atomic features and families
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from aom.data.loaders import load_authority_pairs_with_manifest
from aom.interventions.activation_patching import get_num_layers
from aom.interventions.patching import run_activation_patching
from aom.interventions.patching.authority_protocol import AuthorityLanguageGameProtocol, AuthorityPatchingConfig
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, set_seed


def _parse_int_list(raw: str) -> Optional[List[int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out if out else None


def _parse_str_list(raw: str) -> Optional[List[str]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out: List[str] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(str(p))
    return out if out else None


def _layer_star(*, n_layers: int, depth_frac: float) -> int:
    if int(n_layers) <= 0:
        raise ValueError("n_layers must be positive")
    if not (0.0 <= float(depth_frac) <= 1.0):
        raise ValueError("depth_frac must be in [0, 1]")
    return int(round(float(depth_frac) * float(int(n_layers) - 1)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Authority language-game CPT + SAE recovery runner.")

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])

    p.add_argument("--authority_path", type=str, default=str(root / "data" / "authority_pairs.jsonl"))
    p.add_argument("--data_error_policy", type=str, default="warn_skip", choices=["raise", "warn_skip"])

    p.add_argument("--sites", type=str, default="", help="Comma-separated site names (default: all sites in dataset).")
    p.add_argument("--depth_frac", type=float, default=0.25, help="ℓ* = round(depth_frac*(L-1)) (default: 0.25).")
    p.add_argument("--layer_star", type=int, default=None, help="Override ℓ* layer index.")
    p.add_argument("--patch_layers", type=str, default="", help="Comma-separated layers to sweep (default: all).")
    p.add_argument(
        "--max_tokens_per_site",
        type=int,
        default=0,
        help="If >0, truncate each site span to at most this many tokens (reduces span-length confounds).",
    )
    p.add_argument(
        "--span_take",
        type=str,
        default="last",
        choices=["first", "last"],
        help="When truncating spans, take tokens from the first or last part of the span (default: last).",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--no_length_norm", action="store_true", help="Use sum logprob instead of mean logprob.")

    p.add_argument("--run_sae", action="store_true", help="Run SAE subset-copy recovery at ℓ*.")
    p.add_argument("--analysis_site", type=str, default="marker", help="Site for SAE recovery (default: marker).")
    p.add_argument("--n_features", type=int, default=64, help="Top-K active features to consider (default: 64).")
    p.add_argument("--n_atomic", type=int, default=16, help="Number of atomic features to test (default: 16).")
    p.add_argument("--feature_ids", type=str, default="", help="Optional comma-separated explicit feature ids.")

    p.add_argument("--families_json", type=str, default="", help="Optional JSON mapping family_id -> [feature_ids].")
    p.add_argument("--max_families", type=int, default=50)
    p.add_argument("--compute_synergy", action="store_true", help="Compute S(F)=E(F)-sum E(f) for families.")

    p.add_argument("--sae_repo", type=str, default="", help="HF repo id or local path for Gemma Scope SAE bundle.")
    p.add_argument("--sae_width", type=str, default="16k")
    p.add_argument("--sae_run_name", type=str, default=None)
    p.add_argument("--sae_l0_target", type=int, default=None)
    p.add_argument("--sae_scale", type=float, default=1.0)
    p.add_argument("--sae_dtype", type=str, default="auto")
    p.add_argument("--sae_decode_strategy", type=str, default="delta_1decode", choices=["safe_2decode", "delta_1decode"])
    p.add_argument("--sae_dtype_policy", type=str, default="sae", choices=["sae", "model"])
    p.add_argument("--sae_eps_active", type=float, default=1e-6)

    p.add_argument("--max_pairs", type=int, default=64, help="Limit number of authority pairs for SAE feature selection.")
    p.add_argument("--out_json", type=str, default="", help="Optional path to write JSON results.")
    p.add_argument("--smoke", action="store_true", help="Fast smoke settings.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if bool(args.smoke):
        args.bootstrap_n = min(int(args.bootstrap_n), 200)
        args.n_features = min(int(args.n_features), 16)
        args.n_atomic = min(int(args.n_atomic), 8)
        args.max_pairs = min(int(args.max_pairs), 16)
        args.max_families = min(int(args.max_families), 10)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    items, data_manifest = load_authority_pairs_with_manifest(
        str(args.authority_path),
        role="authority",
        validate=True,
        error_policy=str(args.data_error_policy),
    )
    if not items:
        raise ValueError("authority dataset is empty")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    n_layers = int(get_num_layers(model))
    layer_star = int(args.layer_star) if args.layer_star is not None else _layer_star(n_layers=n_layers, depth_frac=float(args.depth_frac))
    if layer_star < 0 or layer_star >= n_layers:
        raise ValueError(f"layer_star {layer_star} out of range [0, {n_layers})")

    patch_layers = _parse_int_list(str(args.patch_layers))
    if patch_layers is None:
        patch_layers = list(range(n_layers))
    patch_layers = [int(x) for x in patch_layers if 0 <= int(x) < n_layers]
    if not patch_layers:
        raise ValueError("No valid patch layers selected")

    site_list = _parse_str_list(str(args.sites))
    if site_list is None:
        site_list = sorted({str(k) for k in items[0].sites.keys()})
    if not site_list:
        raise ValueError("No sites selected")

    out: Dict[str, Any] = {
        "model_id": str(args.model_name_or_path),
        "arch": str(loaded.architecture),
        "device": str(device),
        "torch_dtype": str(args.torch_dtype or ""),
        "layer_star": int(layer_star),
        "depth_frac": float(args.depth_frac),
        "patch_layers": [int(x) for x in patch_layers],
        "sites": list(site_list),
        "max_tokens_per_site": int(args.max_tokens_per_site),
        "span_take": str(args.span_take),
        "authority_path": str(args.authority_path),
        "dataset_manifest": data_manifest.as_dict(),
        "raw": {},
        "raw_sweep": {},
        "routing": {},
        "sae": {},
    }

    normalize_by_length = not bool(args.no_length_norm)
    max_tokens_per_site = None if int(args.max_tokens_per_site) <= 0 else int(args.max_tokens_per_site)

    # Raw CPT patching (site-by-site).
    raw_star_by_site: Dict[str, float] = {}
    for site in site_list:
        protocol = AuthorityLanguageGameProtocol(
            config=AuthorityPatchingConfig(
                include_sites=(str(site),),
                max_tokens_per_site=max_tokens_per_site,
                span_take=str(args.span_take),
            )
        )
        star_res = run_activation_patching(
            model=model,
            tokenizer=tokenizer,
            protocol=protocol,
            items=list(items),
            device=device,
            layers=[int(layer_star)],
            normalize_by_length=bool(normalize_by_length),
            ci=float(args.ci),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        out["raw"][str(site)] = dict(star_res)
        raw_star_by_site[str(site)] = float(star_res.get("mean_max_effect", float("nan")))

        sweep_res = run_activation_patching(
            model=model,
            tokenizer=tokenizer,
            protocol=protocol,
            items=list(items),
            device=device,
            layers=list(patch_layers),
            normalize_by_length=bool(normalize_by_length),
            ci=float(args.ci),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        out["raw_sweep"][str(site)] = dict(sweep_res)

    # Routing criterion at ℓ*: marker vs distractor.
    marker = "marker"
    distractor = "distractor"
    if marker in raw_star_by_site and distractor in raw_star_by_site:
        e_marker = float(raw_star_by_site[marker])
        e_distr = float(raw_star_by_site[distractor])
        by_dir: Dict[str, Any] = {}
        for direction in ("a_to_b", "b_to_a"):
            key = f"stratum_direction__{direction}_mean_max_effect"
            em = float(out["raw"].get(marker, {}).get(key, float("nan")))
            ed = float(out["raw"].get(distractor, {}).get(key, float("nan")))
            by_dir[str(direction)] = {
                "E_marker": float(em),
                "E_distractor": float(ed),
                "diff": float(em - ed) if math.isfinite(em) and math.isfinite(ed) else float("nan"),
                "criterion_holds": bool(math.isfinite(em) and math.isfinite(ed) and (em > ed)),
            }

        out["routing"] = {
            "layer_star": int(layer_star),
            "E_marker": e_marker,
            "E_distractor": e_distr,
            "diff": float(e_marker - e_distr),
            "criterion_holds": bool(math.isfinite(e_marker) and math.isfinite(e_distr) and (e_marker > e_distr)),
            "by_direction": by_dir,
            "criterion_holds_all_directions": bool(
                all(bool(v.get("criterion_holds", False)) for v in by_dir.values()) if by_dir else False
            ),
        }
    else:
        out["routing"] = {"error": "marker/distractor sites missing from dataset or --sites selection"}

    # Optional SAE subset-copy recovery (atomic + families).
    if bool(args.run_sae):
        if not str(args.sae_repo).strip():
            raise ValueError("--run_sae requires --sae_repo")

        from aom.interventions.sae_adapter import SAEInputTransform, SAEPatchConfig
        from aom.interventions.sae_loader import load_gemma_scope_sae
        from aom.metrics.authority_game import (
            compute_authority_sae_subset_copy_effect,
            family_synergy,
            semantic_gap,
            select_top_features_by_delta_authority,
        )

        sae, sae_meta = load_gemma_scope_sae(
            str(args.sae_repo),
            layer=int(layer_star),
            width=str(args.sae_width),
            run_name=args.sae_run_name,
            l0_target=args.sae_l0_target,
            device=str(device),
            dtype=str(args.sae_dtype),
            local_files_only=bool(args.local_files_only),
        )
        transform = SAEInputTransform(scale=float(args.sae_scale))
        sae_cfg = SAEPatchConfig(
            decode_strategy=str(args.sae_decode_strategy),
            eps_active=float(args.sae_eps_active),
            dtype_policy=str(args.sae_dtype_policy),
        )

        analysis_site = str(args.analysis_site)
        if analysis_site not in set(site_list):
            raise ValueError(f"--analysis_site={analysis_site!r} not found in selected sites={site_list!r}")

        e_raw = float(raw_star_by_site.get(analysis_site, float("nan")))
        feature_ids_cli = _parse_int_list(str(args.feature_ids))
        if feature_ids_cli is not None:
            feature_ids = [int(x) for x in feature_ids_cli]
        else:
            feature_ids = select_top_features_by_delta_authority(
                model=model,
                tokenizer=tokenizer,
                items=items,
                device=device,
                layer=int(layer_star),
                site=str(analysis_site),
                sae=sae,
                transform=transform,
                n_features=int(args.n_features),
                max_pairs=int(args.max_pairs),
                max_tokens_per_site=max_tokens_per_site,
                span_take=str(args.span_take),
            )

        atomic_ids = list(feature_ids)[: int(args.n_atomic)]
        atomic_rows: List[Dict[str, Any]] = []
        atomic_effects: Dict[int, float] = {}
        for fid in atomic_ids:
            res = compute_authority_sae_subset_copy_effect(
                model=model,
                tokenizer=tokenizer,
                items=items,
                device=device,
                layer=int(layer_star),
                site=str(analysis_site),
                sae=sae,
                transform=transform,
                feature_ids=[int(fid)],
                config=sae_cfg,
                normalize_by_length=bool(normalize_by_length),
                max_tokens_per_site=max_tokens_per_site,
                span_take=str(args.span_take),
                ci=float(args.ci),
                bootstrap_n=int(args.bootstrap_n),
                bootstrap_seed=int(args.bootstrap_seed),
            )
            d = res.to_dict()
            d["feature_id"] = int(fid)
            d["semantic_gap"] = float(semantic_gap(e_raw=e_raw, e_sae=float(res.mean_effect)))
            atomic_rows.append(d)
            atomic_effects[int(fid)] = float(res.mean_effect)

        best_atomic = max(atomic_rows, key=lambda r: float(r.get("mean_effect", float("-inf"))), default=None)

        sham_empty = compute_authority_sae_subset_copy_effect(
            model=model,
            tokenizer=tokenizer,
            items=items,
            device=device,
            layer=int(layer_star),
            site=str(analysis_site),
            sae=sae,
            transform=transform,
            feature_ids=[],
            config=sae_cfg,
            normalize_by_length=bool(normalize_by_length),
            max_tokens_per_site=max_tokens_per_site,
            span_take=str(args.span_take),
            ci=float(args.ci),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
        )

        families_out: Dict[str, Any] = {}
        synergies_out: Dict[str, Any] = {}
        fams_path = str(args.families_json or "").strip()
        if fams_path:
            raw = json.loads(Path(fams_path).read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("--families_json must be a JSON object mapping family_id -> [feature_ids]")
            for i, (fam_id, members_raw) in enumerate(sorted(raw.items(), key=lambda kv: str(kv[0]))):
                if i >= int(args.max_families):
                    break
                if not isinstance(members_raw, list):
                    continue
                members = [int(x) for x in members_raw if isinstance(x, (int, float, str)) and str(x).strip()]
                if not members:
                    continue
                fam_res = compute_authority_sae_subset_copy_effect(
                    model=model,
                    tokenizer=tokenizer,
                    items=items,
                    device=device,
                    layer=int(layer_star),
                    site=str(analysis_site),
                    sae=sae,
                    transform=transform,
                    feature_ids=members,
                    config=sae_cfg,
                    normalize_by_length=bool(normalize_by_length),
                    max_tokens_per_site=max_tokens_per_site,
                    span_take=str(args.span_take),
                    ci=float(args.ci),
                    bootstrap_n=int(args.bootstrap_n),
                    bootstrap_seed=int(args.bootstrap_seed),
                )
                families_out[str(fam_id)] = {
                    "feature_ids": members,
                    **fam_res.to_dict(),
                    "semantic_gap": float(semantic_gap(e_raw=e_raw, e_sae=float(fam_res.mean_effect))),
                }

                if bool(args.compute_synergy):
                    atom_vals: List[float] = []
                    for fid in members:
                        if int(fid) not in atomic_effects:
                            single = compute_authority_sae_subset_copy_effect(
                                model=model,
                                tokenizer=tokenizer,
                                items=items,
                                device=device,
                                layer=int(layer_star),
                                site=str(analysis_site),
                                sae=sae,
                                transform=transform,
                                feature_ids=[int(fid)],
                                config=sae_cfg,
                                normalize_by_length=bool(normalize_by_length),
                                max_tokens_per_site=max_tokens_per_site,
                                span_take=str(args.span_take),
                                ci=float(args.ci),
                                bootstrap_n=int(args.bootstrap_n),
                                bootstrap_seed=int(args.bootstrap_seed),
                            )
                            atomic_effects[int(fid)] = float(single.mean_effect)
                        atom_vals.append(float(atomic_effects[int(fid)]))
                    synergies_out[str(fam_id)] = {
                        "S_family": float(family_synergy(e_family=float(fam_res.mean_effect), e_atoms=atom_vals)),
                        "sum_atoms": float(sum(float(x) for x in atom_vals)),
                        "n_atoms": int(len(atom_vals)),
                    }

        out["sae"] = {
            "analysis_site": str(analysis_site),
            "sae_repo": str(args.sae_repo),
            "sae_width": str(args.sae_width),
            "sae_run_name": str(args.sae_run_name or ""),
            "sae_l0_target": str(args.sae_l0_target or ""),
            "sae_meta_run_name": str(getattr(sae_meta, "run_name", "")),
            "layer": int(layer_star),
            "scale": float(args.sae_scale),
            "decode_strategy": str(args.sae_decode_strategy),
            "dtype_policy": str(args.sae_dtype_policy),
            "eps_active": float(args.sae_eps_active),
            "E_raw": float(e_raw),
            "sham_empty_subset": sham_empty.to_dict(),
            "atomic": atomic_rows,
            "best_atomic": best_atomic,
            "families": families_out,
            "synergy": synergies_out,
        }

    print(json.dumps(out, indent=2), flush=True)
    if str(args.out_json).strip():
        p = Path(str(args.out_json))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
