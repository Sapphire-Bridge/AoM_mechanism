#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_requirements import (
    PAPER_CLT_REQUIRED_RUNS as README_CORE_BUNDLE_RUNS,
    PAPER_SCOPE_REPO_ID as HF_REPO_ID,
    PAPER_SCOPE_REVISION as README_CORE_BUNDLE_REVISION,
    resolve_cached_snapshot as _resolve_cached_snapshot,
)

GEMMA2_2B_NUM_LAYERS = 26
GEMMA2_2B_HIDDEN_SIZE = 2304
WIDTH_TO_D_LATENT = {
    "16k": 16384,
    "32k": 32768,
    "65k": 65536,
    "131k": 131072,
    "262k": 262144,
    "524k": 524288,
    "1m": 1048576,
}


def _parse_layers(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("--layers must contain at least one integer layer id")
    for layer in vals:
        if layer < 0 or layer >= GEMMA2_2B_NUM_LAYERS:
            raise ValueError(f"layer {layer} out of range [0, {GEMMA2_2B_NUM_LAYERS})")
    return vals


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _choose_run_name(
    *,
    repo_id: str,
    layer: int,
    width: str,
    run_name: str | None,
    l0_target: int | None,
    revision: str | None,
    local_files_only: bool,
) -> str:
    if run_name:
        return str(run_name)

    prefix = f"layer_{layer}/width_{width}/"
    run_names = _list_runs_for_layer(
        repo_id=repo_id,
        layer=layer,
        width=width,
        revision=revision,
        local_files_only=local_files_only,
    )
    if not run_names:
        raise FileNotFoundError(f"No params.npz runs found under {prefix} in {repo_id}")

    if l0_target is not None:
        tag = f"average_l0_{int(l0_target)}"
        run_names = [x for x in run_names if tag in x]
        if len(run_names) != 1:
            raise ValueError(f"Expected exactly one run matching {tag!r}; found {run_names}")
        return run_names[0]

    preferred = [x for x in run_names if "average_l0_71" in x]
    if len(preferred) == 1:
        return preferred[0]
    if len(run_names) == 1:
        return run_names[0]
    raise ValueError(f"Multiple runs found for {prefix}. Pass --run_name or --l0_target. Found: {run_names}")


def _list_runs_for_layer(
    *,
    repo_id: str,
    layer: int,
    width: str,
    revision: str | None,
    local_files_only: bool,
) -> list[str]:
    prefix = f"layer_{layer}/width_{width}/"
    if local_files_only:
        snapshot = _resolve_cached_snapshot(repo_id, revision)
        base = snapshot / f"layer_{layer}" / f"width_{width}"
        if not base.exists():
            return []
        return sorted(path.name for path in base.iterdir() if path.is_dir() and (path / "params.npz").exists())

    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, revision=revision)
    return sorted({Path(f).parent.name for f in files if f.startswith(prefix) and f.endswith("params.npz")})


def _download_params(
    *,
    repo_id: str,
    layer: int,
    width: str,
    run_name: str,
    out_dir: Path,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    rel = f"layer_{layer}/width_{width}/{run_name}/params.npz"
    if local_files_only:
        src = _resolve_cached_snapshot(repo_id, revision) / rel
        if not src.exists():
            runs = _list_runs_for_layer(
                repo_id=repo_id,
                layer=layer,
                width=width,
                revision=revision,
                local_files_only=True,
            )
            raise FileNotFoundError(
                f"Cached params not found for run '{run_name}' at layer_{layer}/width_{width} in {repo_id}. "
                f"Available cached runs: {runs}"
            )
    else:
        try:
            src = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=rel,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        except EntryNotFoundError as e:
            runs = _list_runs_for_layer(
                repo_id=repo_id,
                layer=layer,
                width=width,
                revision=revision,
                local_files_only=False,
            )
            raise FileNotFoundError(
                f"Run '{run_name}' not found at layer_{layer}/width_{width} in {repo_id}. "
                f"Available runs: {runs}"
            ) from e
    dest = out_dir / f"layer_{layer}" / f"width_{width}" / run_name
    dest.mkdir(parents=True, exist_ok=True)
    dst_file = dest / "params.npz"
    if not dst_file.exists() or _sha256(src) != _sha256(dst_file):
        shutil.copy2(src, dst_file)
    return dest


def _verify_and_write_cfg(run_dir: Path, *, width: str) -> dict[str, Any]:
    params_path = run_dir / "params.npz"
    with np.load(str(params_path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if key not in arrays:
            raise ValueError(f"{params_path} missing {key}; found keys={sorted(arrays.keys())}")

    d_model = GEMMA2_2B_HIDDEN_SIZE
    d_latent = WIDTH_TO_D_LATENT.get(width, int(np.asarray(arrays["b_enc"]).reshape(-1).shape[0]))
    w_enc = np.asarray(arrays["W_enc"])
    w_dec = np.asarray(arrays["W_dec"])
    b_enc = np.asarray(arrays["b_enc"]).reshape(-1)
    b_dec = np.asarray(arrays["b_dec"]).reshape(-1)
    if b_enc.shape != (d_latent,):
        raise ValueError(f"b_enc shape mismatch: got {b_enc.shape}, expected ({d_latent},)")
    if b_dec.shape != (d_model,):
        raise ValueError(f"b_dec shape mismatch: got {b_dec.shape}, expected ({d_model},)")
    if w_enc.shape not in {(d_model, d_latent), (d_latent, d_model)}:
        raise ValueError(f"W_enc shape mismatch: got {w_enc.shape}, expected {(d_model, d_latent)} or transpose")
    if w_dec.shape not in {(d_latent, d_model), (d_model, d_latent)}:
        raise ValueError(f"W_dec shape mismatch: got {w_dec.shape}, expected {(d_latent, d_model)} or transpose")

    has_threshold = "threshold" in arrays
    if has_threshold:
        threshold = np.asarray(arrays["threshold"]).reshape(-1)
        if threshold.shape != (d_latent,):
            raise ValueError(f"threshold shape mismatch: got {threshold.shape}, expected ({d_latent},)")

    cfg = {
        "d_in": d_model,
        "d_latent": d_latent,
        "d_out": d_model,
        "d_sae": d_latent,
        "activation": "jumprelu" if has_threshold else "relu",
        # Keep default False unless explicitly validated otherwise.
        "pre_encoder_bias": False,
        "encode_site": "resid_post",
        "decode_site": "resid_post",
        "writeback_site": "resid_post",
        "site_mode": "same_site_v1",
        "source": "gemma-scope-2b-pt-res",
        "source_format": "jumprelu_sae" if has_threshold else "sae",
        "params_sha256": _sha256(params_path),
    }
    cfg_path = run_dir / "cfg.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Download/prepare Gemma Scope bundles for AoM CLT loader.")
    p.add_argument("--layers", type=str, default="", help="Comma-separated layer indices, e.g. 6,12,18")
    p.add_argument("--width", type=str, default="16k")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--l0_target", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="clt_bundles/gemma-scope-2b-pt-res")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--repo_id", type=str, default=HF_REPO_ID)
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--local_files_only", action="store_true", help="Use only the local HF cache; do not query/download from the Hub.")
    p.add_argument(
        "--preset",
        type=str,
        default="",
        choices=["", "readme_core_bundle"],
        help="Named preset for a fixed multi-layer bundle.",
    )
    p.add_argument("--list_runs", action="store_true", help="List available runs for requested layers and exit.")
    args = p.parse_args()

    run_name_map: dict[int, str] = {}
    if args.preset == "readme_core_bundle":
        layers = sorted(README_CORE_BUNDLE_RUNS)
        run_name_map = dict(README_CORE_BUNDLE_RUNS)
        if args.revision is None:
            args.revision = README_CORE_BUNDLE_REVISION
    elif args.layers:
        layers = _parse_layers(args.layers)
    else:
        raise ValueError("Pass --layers or --preset readme_core_bundle")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing CLT bundle: repo={args.repo_id} width={args.width} layers={layers} out={out_dir}")
    if args.list_runs:
        for layer in layers:
            runs = _list_runs_for_layer(
                repo_id=args.repo_id,
                layer=layer,
                width=args.width,
                revision=args.revision,
                local_files_only=bool(args.local_files_only),
            )
            print(f"layer {layer} runs: {runs}")
        return

    for layer in layers:
        if args.skip_download:
            base = out_dir / f"layer_{layer}" / f"width_{args.width}"
            if not base.exists():
                raise FileNotFoundError(f"{base} does not exist; remove --skip_download for first run")
            runs = [x for x in base.iterdir() if x.is_dir() and (x / "params.npz").exists()]
            if len(runs) != 1:
                raise ValueError(f"Expected exactly one run under {base}; found {len(runs)}")
            run_dir = runs[0]
        else:
            run = _choose_run_name(
                repo_id=args.repo_id,
                layer=layer,
                width=args.width,
                run_name=run_name_map.get(layer, args.run_name),
                l0_target=args.l0_target,
                revision=args.revision,
                local_files_only=bool(args.local_files_only),
            )
            print(f"Layer {layer}: using run {run}")
            run_dir = _download_params(
                repo_id=args.repo_id,
                layer=layer,
                width=args.width,
                run_name=run,
                out_dir=out_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
            )

        cfg = _verify_and_write_cfg(run_dir, width=args.width)
        print(f"Layer {layer}: wrote {run_dir / 'cfg.json'} activation={cfg['activation']}")

    layer_str = ",".join(str(x) for x in layers)
    print("\nREADY\n")
    print("python aom_eval.py \\")
    print("  --model_name_or_path google/gemma-2-2b \\")
    print("  --run_clt_patching \\")
    print(f"  --clt_repo {out_dir} \\")
    print(f"  --clt_layers {layer_str} \\")
    print(f"  --clt_width {args.width} \\")
    if args.run_name:
        print(f"  --clt_run_name {args.run_name} \\")
    elif args.l0_target is not None:
        print(f"  --clt_l0_target {args.l0_target} \\")
    print("  --clt_scale 1.0")


if __name__ == "__main__":
    main()
