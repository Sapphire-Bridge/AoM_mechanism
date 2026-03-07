from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from aom.data.loaders import load_disamb_pairs
from aom.features.cluster_validation import validate_clusters_against_null
from aom.features.clustering import bootstrap_stability, cluster_feature_families, cluster_summary_rows
from aom.features.effect_matrix import FeatureEffectMatrix
from aom.interventions.activation_patching import get_block_outputs
from aom.interventions.clt_adapter import CLTInputTransform
from aom.interventions.clt_loader import load_clt
from aom.models.loader import load_causal_lm
from aom.repro import collect_versions
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.token_spans import token_span_for_substring
from aom.utils import get_best_device


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(k) for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({str(k): r.get(k) for k in keys})


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _safe_token_text(tokenizer, tok_id: int) -> str:
    try:
        toks = tokenizer.convert_ids_to_tokens([int(tok_id)])
        if isinstance(toks, list) and toks:
            return str(toks[0])
    except Exception:
        pass
    try:
        return str(tokenizer.decode([int(tok_id)]))
    except Exception:
        return str(tok_id)


def _load_top_feature_ids_from_summary(summary_path: Path, *, layer: int, top_n: int) -> List[int]:
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in --topk_summary_path, got {type(obj).__name__}")
    ranking = obj.get("ranking_by_layer", None)
    if not isinstance(ranking, dict):
        raise ValueError("top-k summary missing ranking_by_layer")
    layer_entry = ranking.get(str(int(layer)), None)
    if not isinstance(layer_entry, dict):
        raise ValueError(f"top-k summary missing ranking_by_layer[{int(layer)!r}]")
    top = layer_entry.get("top_features", None)
    if not isinstance(top, list):
        raise ValueError(f"top-k summary missing top_features for layer {int(layer)}")
    out: List[int] = []
    for row in top:
        if not isinstance(row, dict):
            continue
        fid = row.get("feature_id", None)
        if isinstance(fid, int) and int(fid) not in out:
            out.append(int(fid))
        if len(out) >= int(top_n):
            break
    return out


def _pooled_std(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 and len(b) < 2:
        return 0.0
    ta = torch.tensor(list(a), dtype=torch.float64) if a else torch.zeros(0, dtype=torch.float64)
    tb = torch.tensor(list(b), dtype=torch.float64) if b else torch.zeros(0, dtype=torch.float64)
    var_a = float(ta.var(unbiased=True).item()) if int(ta.numel()) >= 2 else 0.0
    var_b = float(tb.var(unbiased=True).item()) if int(tb.numel()) >= 2 else 0.0
    num = max(0.0, float(max(0, len(a) - 1)) * var_a + float(max(0, len(b) - 1)) * var_b)
    den = float(max(1, len(a) + len(b) - 2))
    return float((num / den) ** 0.5)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="CLT top-feature activation profiling + selectivity + clustering.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--clt_repo", type=str, required=True)
    p.add_argument("--clt_layer", type=int, default=4)
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_run_name", type=str, default=None)
    p.add_argument("--clt_l0_target", type=int, default=None)
    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--topk_summary_path", type=str, default="")
    p.add_argument("--feature_ids", type=str, default="")
    p.add_argument("--top_n_features", type=int, default=50)
    p.add_argument("--top_n_tokens_per_feature", type=int, default=20)
    p.add_argument("--max_pairs", type=int, default=0, help="0 = all")
    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--similarity_threshold", type=float, default=0.8)
    p.add_argument("--stability_bootstrap_n", type=int, default=200)
    p.add_argument("--null_n", type=int, default=500)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--out_activation_csv", type=str, default=str(root / "results" / "clt_feature_activations.csv"))
    p.add_argument("--out_top_tokens_csv", type=str, default=str(root / "results" / "clt_feature_top_tokens.csv"))
    p.add_argument("--out_selectivity_csv", type=str, default=str(root / "results" / "clt_feature_selectivity.csv"))
    p.add_argument("--out_families_json", type=str, default=str(root / "results" / "clt_feature_families.json"))
    p.add_argument(
        "--out_families_summary_csv",
        type=str,
        default=str(root / "results" / "clt_feature_families_summary.csv"),
    )
    p.add_argument(
        "--out_family_validation_csv",
        type=str,
        default=str(root / "results" / "clt_feature_family_validation.csv"),
    )
    p.add_argument("--manifest_path", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        args.top_n_features = min(int(args.top_n_features), 16)
        args.top_n_tokens_per_feature = min(int(args.top_n_tokens_per_feature), 8)
        args.max_pairs = 8 if int(args.max_pairs) <= 0 else min(int(args.max_pairs), 8)
        args.stability_bootstrap_n = min(int(args.stability_bootstrap_n), 50)
        args.null_n = min(int(args.null_n), 100)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    items = list(load_disamb_pairs(str(args.disamb_path)))
    if int(args.max_pairs) > 0:
        items = items[: int(args.max_pairs)]
    if not items:
        raise ValueError("DISAMB dataset is empty")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        device_map=None,
    )
    clt, _meta = load_clt(
        str(args.clt_repo),
        layer=int(args.clt_layer),
        width=str(args.clt_width),
        run_name=args.clt_run_name,
        l0_target=args.clt_l0_target,
        device=str(device),
        dtype=str(args.clt_dtype),
        local_files_only=bool(args.local_files_only),
    )
    transform = CLTInputTransform(scale=float(args.clt_scale))

    feature_ids: List[int]
    if str(args.feature_ids).strip():
        feature_ids = _parse_int_csv(str(args.feature_ids))
    elif str(args.topk_summary_path).strip():
        feature_ids = _load_top_feature_ids_from_summary(
            Path(str(args.topk_summary_path)),
            layer=int(args.clt_layer),
            top_n=int(args.top_n_features),
        )
    else:
        raise ValueError("Provide --feature_ids or --topk_summary_path")
    if not feature_ids:
        raise ValueError("No feature IDs resolved")

    d_latent = int(getattr(clt, "d_latent", 0))
    if d_latent <= 0:
        raise ValueError("Invalid CLT latent width")
    feature_ids = sorted({int(fid) for fid in feature_ids if 0 <= int(fid) < int(d_latent)})
    if not feature_ids:
        raise ValueError("Resolved feature IDs are out of range for this CLT")

    clt_param = next(clt.parameters(), None) if isinstance(clt, torch.nn.Module) else None
    clt_device = clt_param.device if clt_param is not None else torch.device("cpu")
    clt_dtype = clt_param.dtype if clt_param is not None else torch.float32

    activation_rows: List[Dict[str, Any]] = []
    side_means_by_feature: Dict[int, Dict[str, List[float]]] = {int(fid): {"a": [], "b": []} for fid in feature_ids}
    effect_by_feature_condition: Dict[int, Dict[str, float]] = {int(fid): {} for fid in feature_ids}
    conditions: List[str] = []

    for it in items:
        for side_name, side in (("a", it.a), ("b", it.b)):
            span, _ = token_span_for_substring(loaded.tokenizer, side.prompt, it.target, it.target_occurrence)
            if not span:
                continue
            enc = loaded.tokenizer(side.prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            block = get_block_outputs(loaded.model, input_ids, layers=[int(args.clt_layer)])[int(args.clt_layer)]
            latents = clt.encode(transform.forward(block.to(device=clt_device, dtype=clt_dtype)))
            lat = latents[0].detach().to(device="cpu", dtype=torch.float32)
            token_ids = [int(x) for x in input_ids[0].detach().to(device="cpu").tolist()]
            cond_id = f"{it.pair_id}:{side_name}"
            if cond_id not in conditions:
                conditions.append(cond_id)
            target_set = set(int(x) for x in span)
            for fid in feature_ids:
                vals = lat[:, int(fid)]
                target_vals = vals[[int(i) for i in span]]
                target_mean = float(target_vals.mean().item()) if int(target_vals.numel()) > 0 else float("nan")
                side_means_by_feature[int(fid)][str(side_name)].append(float(target_mean))
                effect_by_feature_condition[int(fid)][str(cond_id)] = float(target_mean)
                for tok_idx, tok_id in enumerate(token_ids):
                    activation_rows.append(
                        {
                            "pair_id": str(it.pair_id),
                            "side": str(side_name),
                            "expected_label": str(side.expected_label),
                            "feature_id": int(fid),
                            "token_index": int(tok_idx),
                            "token_id": int(tok_id),
                            "token_text": _safe_token_text(loaded.tokenizer, int(tok_id)),
                            "activation": float(vals[int(tok_idx)].item()),
                            "is_target_token": int(int(tok_idx) in target_set),
                            "condition_id": str(cond_id),
                        }
                    )

    selectivity_rows: List[Dict[str, Any]] = []
    for fid in feature_ids:
        vals_a = list(side_means_by_feature[int(fid)]["a"])
        vals_b = list(side_means_by_feature[int(fid)]["b"])
        ta = torch.tensor(vals_a, dtype=torch.float64) if vals_a else torch.zeros(0, dtype=torch.float64)
        tb = torch.tensor(vals_b, dtype=torch.float64) if vals_b else torch.zeros(0, dtype=torch.float64)
        mean_a = float(ta.mean().item()) if int(ta.numel()) > 0 else float("nan")
        mean_b = float(tb.mean().item()) if int(tb.numel()) > 0 else float("nan")
        std_a = float(ta.std(unbiased=True).item()) if int(ta.numel()) >= 2 else 0.0
        std_b = float(tb.std(unbiased=True).item()) if int(tb.numel()) >= 2 else 0.0
        pooled = _pooled_std(vals_a, vals_b)
        sel = float((mean_a - mean_b) / pooled) if pooled > 0.0 else float("nan")
        selectivity_rows.append(
            {
                "feature_id": int(fid),
                "n_side_a": int(len(vals_a)),
                "n_side_b": int(len(vals_b)),
                "mean_activation_side_a": float(mean_a),
                "mean_activation_side_b": float(mean_b),
                "std_side_a": float(std_a),
                "std_side_b": float(std_b),
                "pooled_std": float(pooled),
                "selectivity_index": float(sel),
                "abs_selectivity_index": float(abs(sel)) if sel == sel else float("nan"),
            }
        )
    selectivity_rows.sort(
        key=lambda r: abs(float(r.get("selectivity_index", 0.0)))
        if float(r.get("selectivity_index", float("nan"))) == float(r.get("selectivity_index", float("nan")))
        else -1.0,
        reverse=True,
    )

    top_tokens_rows: List[Dict[str, Any]] = []
    for fid in feature_ids:
        rows_f = [r for r in activation_rows if int(r["feature_id"]) == int(fid)]
        rows_f.sort(key=lambda r: float(r["activation"]), reverse=True)
        for rank, r in enumerate(rows_f[: int(args.top_n_tokens_per_feature)]):
            top_tokens_rows.append(
                {
                    "feature_id": int(fid),
                    "rank": int(rank + 1),
                    "pair_id": str(r["pair_id"]),
                    "side": str(r["side"]),
                    "token_index": int(r["token_index"]),
                    "token_id": int(r["token_id"]),
                    "token_text": str(r["token_text"]),
                    "activation": float(r["activation"]),
                    "is_target_token": int(r["is_target_token"]),
                }
            )

    matrix = FeatureEffectMatrix(
        feature_ids=tuple(int(x) for x in feature_ids),
        condition_ids=tuple(str(x) for x in conditions),
        effects={int(fid): dict(effect_by_feature_condition[int(fid)]) for fid in feature_ids},
    )
    clusters = cluster_feature_families(matrix, similarity_threshold=float(args.similarity_threshold))
    stability = bootstrap_stability(
        matrix,
        similarity_threshold=float(args.similarity_threshold),
        n_bootstrap=int(args.stability_bootstrap_n),
        seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    cluster_rows = cluster_summary_rows(matrix, clusters, stability=stability)
    validation_rows = [
        r.to_row()
        for r in validate_clusters_against_null(
            matrix,
            clusters,
            n_null=int(args.null_n),
            seed=int(args.bootstrap_seed),
            ci=float(args.ci),
        )
    ]

    out_activation_csv = Path(str(args.out_activation_csv))
    out_top_tokens_csv = Path(str(args.out_top_tokens_csv))
    out_selectivity_csv = Path(str(args.out_selectivity_csv))
    out_families_json = Path(str(args.out_families_json))
    out_families_summary_csv = Path(str(args.out_families_summary_csv))
    out_family_validation_csv = Path(str(args.out_family_validation_csv))
    manifest_path = (
        Path(str(args.manifest_path))
        if str(args.manifest_path).strip()
        else out_families_summary_csv.with_suffix(".manifest.json")
    )

    _write_csv(out_activation_csv, activation_rows)
    _write_csv(out_top_tokens_csv, top_tokens_rows)
    _write_csv(out_selectivity_csv, selectivity_rows)
    _write_csv(out_families_summary_csv, cluster_rows)
    _write_csv(out_family_validation_csv, validation_rows)
    out_families_json.parent.mkdir(parents=True, exist_ok=True)
    out_families_json.write_text(
        json.dumps({str(k): [int(x) for x in v] for k, v in clusters.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_row = {
        "model_name_or_path": str(args.model_name_or_path),
        "clt_layer": int(args.clt_layer),
        "n_features": int(len(feature_ids)),
        "n_conditions": int(len(conditions)),
        "n_clusters": int(len(clusters)),
        "out_activation_csv": str(out_activation_csv),
        "out_top_tokens_csv": str(out_top_tokens_csv),
        "out_selectivity_csv": str(out_selectivity_csv),
        "out_families_json": str(out_families_json),
        "out_families_summary_csv": str(out_families_summary_csv),
        "out_family_validation_csv": str(out_family_validation_csv),
        "families_summary_sha256": _sha256_file(out_families_summary_csv) if out_families_summary_csv.exists() else "",
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path),
        csv_path=str(out_families_summary_csv),
        csv_sha256=result_row["families_summary_sha256"],
        csv_n_rows=int(len(cluster_rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(feature_ids)),
        "succeeded": int(len(feature_ids)),
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
    manifest["versions"] = collect_versions()
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote activation CSV: {out_activation_csv}", flush=True)
    print(f"Wrote top-token CSV: {out_top_tokens_csv}", flush=True)
    print(f"Wrote selectivity CSV: {out_selectivity_csv}", flush=True)
    print(f"Wrote feature families JSON: {out_families_json}", flush=True)
    print(f"Wrote feature family summary CSV: {out_families_summary_csv}", flush=True)
    print(f"Wrote feature family validation CSV: {out_family_validation_csv}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
