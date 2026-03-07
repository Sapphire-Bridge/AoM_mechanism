from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, List


ROOT = Path(__file__).resolve().parents[1]

MANDATORY_FIRST_MODELS: list[str] = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "google/gemma-2-2b",
]

DEFAULT_FOLLOWUP_MODELS: list[str] = [
    "Qwen/Qwen2.5-3B",
    "allenai/OLMo-7B-0724-hf",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(s: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(s).lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _shlex_join(argv: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in argv)


def _run(argv: List[str], *, dry_run: bool) -> None:
    print(_shlex_join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(ROOT), check=True)


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {str(path)}")
    return obj


def _dedupe_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run canonical MoM flagship evals with frozen protocol hash and verify run manifests."
    )
    p.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_FOLLOWUP_MODELS),
        help=(
            "Comma-separated model ids to run after mandatory first-stage models "
            "(Pythia-70M, Pythia-160M, Gemma-2-2B)."
        ),
    )
    p.add_argument(
        "--protocol_path",
        type=str,
        default=str(ROOT / "configs" / "mom_flagship_protocol.yaml"),
        help="Frozen protocol file path.",
    )
    p.add_argument("--disamb_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--cf_path", type=str, default=str(ROOT / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(ROOT / "data" / "coherence.jsonl"))
    p.add_argument("--dataset_manifest_path", type=str, default="", help="Optional DATASET_MANIFEST.json path.")
    p.add_argument("--results_dir", type=str, default=str(ROOT / "results" / "mom_flagship"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--revision", type=str, default="", help="Optional HF model revision (commit/tag/branch).")
    p.add_argument("--tokenizer_revision", type=str, default="", help="Optional HF tokenizer revision.")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(str(args.protocol_path)).expanduser().resolve()
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol_path not found: {str(protocol_path)}")
    protocol_sha256 = _sha256_file(protocol_path)

    results_dir = Path(str(args.results_dir)).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    models = _dedupe_preserve(MANDATORY_FIRST_MODELS + requested_models)
    if not models:
        raise ValueError("No models provided after expansion")
    print(
        "[master] model execution order: " + ", ".join(models),
        flush=True,
    )

    run_meta: dict[str, Any] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": str(protocol_sha256),
        "mandatory_first_models": list(MANDATORY_FIRST_MODELS),
        "requested_followup_models": requested_models,
        "models": models,
        "runs": [],
    }

    for model_name in models:
        slug = _slug(model_name)
        csv_path = results_dir / f"{slug}.csv"
        manifest_path = csv_path.with_suffix(".manifest.json")

        cmd: List[str] = [
            sys.executable,
            str(ROOT / "aom_eval.py"),
            "--model_name_or_path",
            str(model_name),
            "--device",
            str(args.device),
            "--attn_implementation",
            str(args.attn_implementation),
            "--disamb_path",
            str(args.disamb_path),
            "--cf_path",
            str(args.cf_path),
            "--coh_path",
            str(args.coh_path),
            "--bootstrap_n",
            str(int(args.bootstrap_n)),
            "--bootstrap_seed",
            str(int(args.bootstrap_seed)),
            "--ci",
            str(float(args.ci)),
            "--seed",
            str(int(args.seed)),
            "--run_patching",
            "--run_patching_specificity",
            "--strict_finite",
            "--strict_metrics",
            "--require_git",
            "--protocol_path",
            str(protocol_path),
            "--protocol_sha256",
            str(protocol_sha256),
            "--csv_path",
            str(csv_path),
        ]
        if str(args.dataset_manifest_path).strip():
            cmd += ["--dataset_manifest_path", str(args.dataset_manifest_path).strip()]
        if bool(args.local_files_only):
            cmd.append("--local_files_only")
        if bool(args.trust_remote_code):
            cmd.append("--trust_remote_code")
        if str(args.revision).strip():
            cmd += ["--revision", str(args.revision).strip()]
        if str(args.tokenizer_revision).strip():
            cmd += ["--tokenizer_revision", str(args.tokenizer_revision).strip()]
        # Qwen typically prefers bf16; leave other models default unless overridden at CLI.
        if "qwen" in model_name.lower():
            cmd += ["--torch_dtype", "bfloat16"]

        _run(cmd, dry_run=bool(args.dry_run))

        run_info: dict[str, Any] = {
            "model": str(model_name),
            "csv_path": str(csv_path),
            "manifest_path": str(manifest_path),
        }
        if not bool(args.dry_run):
            manifest = _load_json(manifest_path)
            msha = str(manifest.get("protocol_sha256", "") or "")
            if msha != protocol_sha256:
                raise ValueError(
                    f"Protocol hash mismatch for {model_name}: manifest={msha!r} expected={protocol_sha256!r}"
                )
            verified = bool(manifest.get("protocol_sha256_verified", False))
            if not verified:
                raise ValueError(f"Manifest did not mark protocol_sha256 as verified for {model_name}")
            run_info["verified_protocol_sha256"] = str(msha)
            run_info["protocol_sha256_verified"] = bool(verified)
        run_meta["runs"].append(run_info)

    meta_path = results_dir / "mom_flagship_runs.json"
    if not bool(args.dry_run):
        meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(run_meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
