from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from aom.data.loaders import load_disamb_pairs
from aom.features.cluster_validation import validate_clusters_against_null
from aom.features.clustering import bootstrap_stability, cluster_feature_families, cluster_summary_rows
from aom.features.effect_matrix import (
    build_disamb_feature_effect_matrix,
    matrix_from_rows,
    select_top_features_by_activation,
)
from aom.interventions.sae_adapter import SAEInputTransform
from aom.interventions.sae_loader import load_gemma_scope_sae
from aom.models.loader import load_causal_lm
from aom.repro import ReproConfig, collect_versions, seed_everything
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.utils import get_best_device


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(k) for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [{str(k): v for k, v in row.items()} for row in r]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Feature-family clustering and validation from SAE effect matrix.")
    p.add_argument("--effect_matrix_csv", type=str, default="", help="Optional precomputed feature effect matrix CSV.")

    p.add_argument("--model_name_or_path", type=str, default="")
    p.add_argument("--sae_repo", type=str, default="")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--width", type=str, default="16k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--l0_target", type=int, default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--n_features", type=int, default=64)
    p.add_argument("--max_pairs", type=int, default=32)
    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--similarity_threshold", type=float, default=0.8)
    p.add_argument("--stability_bootstrap_n", type=int, default=200)
    p.add_argument("--null_n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)

    p.add_argument("--out_effect_matrix_csv", type=str, default=str(root / "results" / "feature_effects.csv"))
    p.add_argument("--out_families_json", type=str, default=str(root / "results" / "feature_families.json"))
    p.add_argument(
        "--out_families_summary_csv",
        type=str,
        default=str(root / "results" / "feature_families_summary.csv"),
    )
    p.add_argument(
        "--out_family_validation_csv",
        type=str,
        default=str(root / "results" / "feature_family_validation.csv"),
    )
    p.add_argument("--manifest_path", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        args.max_pairs = min(int(args.max_pairs), 8)
        args.n_features = min(int(args.n_features), 16)
        args.stability_bootstrap_n = min(int(args.stability_bootstrap_n), 50)
        args.null_n = min(int(args.null_n), 100)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    repro = seed_everything(ReproConfig(seed=int(args.seed), determinism="best_effort"), device=device)
    versions = collect_versions()

    out_effect_csv = Path(str(args.out_effect_matrix_csv))
    if str(args.effect_matrix_csv).strip():
        matrix_rows = _read_csv_rows(Path(str(args.effect_matrix_csv)))
        matrix = matrix_from_rows(matrix_rows)
    else:
        if not str(args.model_name_or_path).strip():
            raise ValueError("--model_name_or_path is required when --effect_matrix_csv is not provided")
        if not str(args.sae_repo).strip() or int(args.layer) < 0:
            raise ValueError("--sae_repo and non-negative --layer are required when --effect_matrix_csv is not provided")

        items = load_disamb_pairs(str(args.disamb_path))
        loaded = load_causal_lm(
            str(args.model_name_or_path),
            device=device,
            torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            attn_implementation=str(args.attn_implementation),
            device_map=None,
        )
        sae, _meta = load_gemma_scope_sae(
            str(args.sae_repo),
            layer=int(args.layer),
            width=str(args.width),
            run_name=args.run_name,
            l0_target=args.l0_target,
            device=str(device),
            dtype=str(args.torch_dtype or "float32"),
            local_files_only=bool(args.local_files_only),
        )
        transform = SAEInputTransform(scale=float(args.scale))
        feature_ids = select_top_features_by_activation(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            items=items,
            device=device,
            layer=int(args.layer),
            sae=sae,
            transform=transform,
            n_features=int(args.n_features),
            max_pairs=int(args.max_pairs),
        )
        matrix = build_disamb_feature_effect_matrix(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            items=items,
            device=device,
            layer=int(args.layer),
            sae=sae,
            transform=transform,
            feature_ids=feature_ids,
            max_pairs=int(args.max_pairs),
        )
        _write_csv(out_effect_csv, matrix.to_rows())

    clusters = cluster_feature_families(matrix, similarity_threshold=float(args.similarity_threshold))
    stability = bootstrap_stability(
        matrix,
        similarity_threshold=float(args.similarity_threshold),
        n_bootstrap=int(args.stability_bootstrap_n),
        seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    summary_rows = cluster_summary_rows(matrix, clusters, stability=stability)
    validation = validate_clusters_against_null(
        matrix,
        clusters,
        n_null=int(args.null_n),
        seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    validation_rows = [x.to_row() for x in validation]

    out_families_json = Path(str(args.out_families_json))
    out_families_json.parent.mkdir(parents=True, exist_ok=True)
    out_families_json.write_text(
        json.dumps({str(cid): [int(x) for x in fids] for cid, fids in clusters.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_summary_csv = Path(str(args.out_families_summary_csv))
    out_validation_csv = Path(str(args.out_family_validation_csv))
    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_validation_csv, validation_rows)

    manifest_path = (
        Path(str(args.manifest_path))
        if str(args.manifest_path).strip()
        else out_summary_csv.with_suffix(".manifest.json")
    )
    result_row = {
        "n_features": int(len(matrix.feature_ids)),
        "n_conditions": int(len(matrix.condition_ids)),
        "n_clusters": int(len(clusters)),
        "similarity_threshold": float(args.similarity_threshold),
        "stability_mean_ari": float(stability.mean_ari),
        "stability_ari_ci_low": float(stability.ari_ci_low),
        "stability_ari_ci_high": float(stability.ari_ci_high),
        "out_effect_matrix_csv": str(out_effect_csv),
        "out_families_json": str(out_families_json),
        "out_families_summary_csv": str(out_summary_csv),
        "out_family_validation_csv": str(out_validation_csv),
        "out_families_summary_csv_sha256": _sha256_file(out_summary_csv) if out_summary_csv.exists() else "",
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path),
        csv_path=str(out_summary_csv),
        csv_sha256=result_row["out_families_summary_csv_sha256"],
        csv_n_rows=int(len(summary_rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(matrix.feature_ids)),
        "succeeded": int(len(matrix.feature_ids)),
        "failed": 0,
        "skipped": 0,
        "invalid": 0,
        "fail_rate": 0.0,
        "skip_rate": 0.0,
        "invalid_rate": 0.0,
        "top_failure_types": [],
        "top_skip_types": [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    manifest["versions"] = versions
    manifest["repro"] = repro
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote feature families: {out_families_json}", flush=True)
    print(f"Wrote summary CSV: {out_summary_csv}", flush=True)
    print(f"Wrote validation CSV: {out_validation_csv}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
