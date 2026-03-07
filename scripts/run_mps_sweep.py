from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _sanitize_filename(s: str) -> str:
    s = s.strip().replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _infer_num_layers_from_config(model_name_or_path: str, *, local_files_only: bool, trust_remote_code: bool) -> Optional[int]:
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception:
        return None

    try:
        cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return None

    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


def _auto_patch_layers(model_name_or_path: str, *, local_files_only: bool, trust_remote_code: bool) -> Optional[str]:
    n_layers = _infer_num_layers_from_config(
        model_name_or_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code
    )
    if n_layers is None:
        return None
    if n_layers == 1:
        return "0"
    # A small, depth-spread subset to keep CPT patching tractable on consumer hardware.
    picks = sorted(
        {
            0,
            int(round(0.25 * (n_layers - 1))),
            int(round(0.50 * (n_layers - 1))),
            int(round(0.75 * (n_layers - 1))),
            n_layers - 1,
        }
    )
    return ",".join(str(x) for x in picks)


def _build_aom_eval_cmd(
    *,
    model_name_or_path: str,
    seeds: Sequence[int],
    out_csv: Path,
    disamb_path: Path,
    cf_path: Path,
    coh_path: Path,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
    logprobs_dtype: str,
    local_files_only: bool,
    trust_remote_code: bool,
    bootstrap_n: int,
    bootstrap_seed: int,
    ci: float,
    run_patching: bool,
    patch_layers: Optional[str],
    run_patching_specificity: bool,
    patch_specificity_depth_frac: float,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "aom_eval.py"),
        "--model_name_or_path",
        model_name_or_path,
        "--device",
        device,
        "--torch_dtype",
        torch_dtype,
        "--attn_implementation",
        attn_implementation,
        "--logprobs_dtype",
        logprobs_dtype,
        "--bootstrap_n",
        str(int(bootstrap_n)),
        "--bootstrap_seed",
        str(int(bootstrap_seed)),
        "--ci",
        str(float(ci)),
        "--disamb_path",
        str(disamb_path),
        "--cf_path",
        str(cf_path),
        "--coh_path",
        str(coh_path),
        "--csv_path",
        str(out_csv),
        "--sweep_seeds",
        *[str(int(s)) for s in seeds],
    ]

    if local_files_only:
        cmd.append("--local_files_only")
    if trust_remote_code:
        cmd.append("--trust_remote_code")

    if run_patching:
        cmd.append("--run_patching")
        if patch_layers is not None:
            cmd.extend(["--patch_layers", patch_layers])

    if run_patching_specificity:
        cmd.append("--run_patching_specificity")
        cmd.extend(["--patch_specificity_depth_frac", str(float(patch_specificity_depth_frac))])

    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="Model IDs or local paths (HuggingFace-compatible).")
    p.add_argument("--seeds", nargs="*", type=int, default=[0], help="Random seeds to sweep (default: 0).")
    p.add_argument("--out_csv", type=str, default=str(ROOT / "results" / "mps_sweep.csv"))
    p.add_argument("--log_dir", type=str, default=str(ROOT / "results" / "logs"))
    p.add_argument("--fresh", action="store_true", help="Overwrite out_csv instead of appending.")

    p.add_argument("--device", type=str, default="mps", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Use float16 on MPS for memory efficiency (fallback to float32 if unstable).",
    )
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])

    p.add_argument("--local_files_only", action="store_true", help="Disable downloads (recommended for sweeps).")
    p.add_argument("--trust_remote_code", action="store_true", help="Only enable for repos you trust.")

    p.add_argument("--bootstrap_n", type=int, default=200, help="Lower for faster first-draft runs.")
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)

    p.add_argument(
        "--patch_layers",
        type=str,
        default="auto",
        help="CPT patch layers: 'auto' (5 spread layers), 'all' (default aom_eval behavior), or a comma list like '0,6,12,18,23'.",
    )
    p.add_argument("--no_patching", action="store_true", help="Skip CPT context-swap patching (fast).")
    p.add_argument("--no_specificity", action="store_true", help="Skip CPT target-specificity control.")
    p.add_argument("--patch_specificity_depth_frac", type=float, default=0.25)

    p.add_argument("--disamb_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--cf_path", type=str, default=str(ROOT / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(ROOT / "data" / "coherence.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.out_csv)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    disamb_path = Path(args.disamb_path)
    cf_path = Path(args.cf_path)
    coh_path = Path(args.coh_path)

    # Incrementally build a single CSV so partial progress is preserved if a model OOMs/crashes.
    if bool(args.fresh) and out_csv.exists():
        out_csv.unlink()
    all_rows: List[Dict[str, Any]] = _read_csv_rows(out_csv)

    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    for model in args.models:
        model_tag = _sanitize_filename(model)
        tmp_csv = out_csv.with_suffix(f".{model_tag}.tmp.csv")
        log_path = log_dir / f"{model_tag}.log"

        patch_layers: Optional[str]
        if str(args.patch_layers).strip().lower() == "all":
            patch_layers = None  # let aom_eval default to "all layers"
        elif str(args.patch_layers).strip().lower() == "auto":
            patch_layers = _auto_patch_layers(
                model, local_files_only=bool(args.local_files_only), trust_remote_code=bool(args.trust_remote_code)
            )
        else:
            patch_layers = str(args.patch_layers).strip()

        cmd = _build_aom_eval_cmd(
            model_name_or_path=model,
            seeds=list(args.seeds),
            out_csv=tmp_csv,
            disamb_path=disamb_path,
            cf_path=cf_path,
            coh_path=coh_path,
            device=str(args.device),
            torch_dtype=str(args.torch_dtype),
            attn_implementation=str(args.attn_implementation),
            logprobs_dtype=str(args.logprobs_dtype),
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            ci=float(args.ci),
            run_patching=not bool(args.no_patching),
            patch_layers=patch_layers,
            run_patching_specificity=not bool(args.no_specificity),
            patch_specificity_depth_frac=float(args.patch_specificity_depth_frac),
        )

        print(f"[run] model={model!r} seeds={list(args.seeds)} patch_layers={patch_layers!r}", flush=True)
        print(f"[run] cmd={' '.join(cmd)}", flush=True)

        try:
            with open(log_path, "w", encoding="utf-8") as log_f:
                subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] model={model!r} exited with code {e.returncode}; log={log_path}", flush=True)
            continue

        new_rows = _read_csv_rows(tmp_csv)
        all_rows.extend(new_rows)
        _write_csv_rows(all_rows, out_csv)

        try:
            tmp_csv.unlink()
        except Exception:
            pass

    print(f"[done] wrote {len(all_rows)} rows to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
