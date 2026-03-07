from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from aom.config import load_config
from aom.data.loaders import load_coherence_items, load_counterfactual_pairs, load_disamb_pairs
from aom.metrics.completeness import compute_task_completeness, parse_explanation_config
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_layers(raw: str) -> Optional[List[int]]:
    s = str(raw or "").strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        p = str(part).strip()
        if not p:
            continue
        out.append(int(p))
    return out if out else None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Compute explanatory completeness and dark-matter metrics.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--determinism", type=str, default="best_effort", choices=["strict", "best_effort", "off"])

    p.add_argument("--tasks", type=str, default="disamb,cf,coh")
    p.add_argument("--layers", type=str, default="", help="Optional comma-separated layer ids for total effect.")
    p.add_argument("--explanation_path", type=str, default="", help="Optional YAML/JSON explanation config.")

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--cf_path", type=str, default=str(root / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(root / "data" / "coherence.jsonl"))

    p.add_argument("--bootstrap_n", type=int, default=500)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--csv_path", type=str, default=str(root / "results" / "completeness.csv"))
    p.add_argument("--manifest_path", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        args.bootstrap_n = min(int(args.bootstrap_n), 100)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    repro = seed_everything(ReproConfig(seed=int(args.seed), determinism=str(args.determinism)), device=device)
    versions = collect_versions()
    if str(args.determinism) == "strict" and not bool(repro.get("determinism_enforced", False)):
        raise ValueError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        device_map=None,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    explanation_cfg: Dict[str, Any] = {}
    if str(args.explanation_path).strip():
        cfg = load_config(Path(str(args.explanation_path)))
        if "explanation" in cfg and isinstance(cfg["explanation"], dict):
            explanation_cfg = parse_explanation_config(cfg["explanation"])
        else:
            explanation_cfg = parse_explanation_config(cfg)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    layers = _parse_layers(str(args.layers))
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        if task == "disamb":
            items = load_disamb_pairs(str(args.disamb_path))
        elif task == "cf":
            items = load_counterfactual_pairs(str(args.cf_path))
        elif task == "coh":
            items = load_coherence_items(str(args.coh_path))
        else:
            raise ValueError(f"Unsupported task: {task!r}")
        res = compute_task_completeness(
            model=model,
            tokenizer=tokenizer,
            items=items,
            task=str(task),
            device=device,
            total_layers=layers,
            explanation=explanation_cfg,
            normalize_by_length=True,
            ci=float(args.ci),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        row = res.to_row()
        row.update(
            {
                "model": str(args.model_name_or_path),
                "arch": str(loaded.architecture),
                "seed": int(args.seed),
                "bootstrap_n": int(args.bootstrap_n),
                "bootstrap_seed": int(args.bootstrap_seed),
                "ci": float(args.ci),
            }
        )
        rows.append(row)

    csv_path = Path(str(args.csv_path))
    _write_csv(csv_path, rows)

    manifest_path = Path(str(args.manifest_path)) if str(args.manifest_path).strip() else csv_path.with_suffix(".manifest.json")
    result_row = {
        "model": str(args.model_name_or_path),
        "n_tasks": int(len(rows)),
        "csv_path": str(csv_path),
        "csv_sha256": _sha256_file(csv_path) if csv_path.exists() else "",
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path="",
        csv_path=str(csv_path),
        csv_sha256=result_row["csv_sha256"],
        csv_n_rows=int(len(rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(rows)),
        "succeeded": int(len(rows)),
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

    print(f"Wrote completeness CSV: {csv_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
