#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _shlex_join(argv: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in argv)


def _run(argv: List[str], *, dry_run: bool) -> None:
    print(_shlex_join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(ROOT), check=True)


def _find_hf_snapshot_dir(*, hub_dir: Path, model_key: str) -> Path | None:
    """
    Find a single snapshot directory under ~/.cache/huggingface/hub/ for a model.

    model_key example: "models--google--gemma-2-2b-it"
    """
    snapshots = hub_dir / model_key / "snapshots"
    if not snapshots.exists():
        return None
    # Prefer lexicographically last snapshot id as a stable heuristic.
    cands = sorted([p for p in snapshots.iterdir() if p.is_dir()])
    return cands[-1] if cands else None


def _default_hf_hub_dir() -> Path:
    # HF uses $HF_HOME by default, but the common path is ~/.cache/huggingface.
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        return Path(hf_home).expanduser().resolve() / "hub"
    return Path.home().resolve() / ".cache" / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end safety DISAMB run (chat-primary): convert raw safety pairs to naturalistic continuations, "
            "render chat prompts, split CSD/IH dev/test, and run aom_eval baselines (+ optional CLT patching) "
            "for Gemma base + Gemma-IT."
        )
    )
    p.add_argument(
        "--raw_pairs",
        type=str,
        default=str(ROOT / "data" / "safety_disamb_pairs_raw_v2_csd_ih.jsonl"),
        help="Input safety raw pairs JSONL (id/prompt_a/prompt_b/target_a/target_b/meta).",
    )
    p.add_argument(
        "--choice_profile",
        type=str,
        default="naturalistic_a",
        choices=["rubric", "naturalistic", "naturalistic_a", "naturalistic_b"],
        help="Continuation profile for safety labels (help/refuse/caution).",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Output directory (defaults to results/safety_chat_<timestamp> if unset).",
    )
    p.add_argument(
        "--include_combined",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the combined CSD+IH split (default: off; for speed, CSD and IH runs are sufficient).",
    )
    p.add_argument(
        "--also_run_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If --run_clt_patching is enabled, also run separate baseline CSVs (default: off; CLT CSV already includes baseline metrics).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run up to this many aom_eval subprocesses concurrently (default: 1).",
    )

    # Models: default to HF cache snapshots if present.
    p.add_argument("--base_model_name_or_path", type=str, default="", help="Gemma base model path/id (optional).")
    p.add_argument("--it_model_name_or_path", type=str, default="", help="Gemma-IT model path/id (optional).")
    p.add_argument(
        "--chat_tokenizer_name_or_path",
        type=str,
        default="",
        help=(
            "Tokenizer used for chat prerendering (defaults to --it_model_name_or_path if unset). "
            "Must support apply_chat_template."
        ),
    )
    p.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disallow downloads (offline).",
    )

    # Split + eval knobs.
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--dev_n", type=int, default=10, help="Dev pairs per group (csd, ih).")
    p.add_argument("--seed", type=int, default=42, help="aom_eval random seed.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--score_batch_size", type=int, default=8)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--torch_dtype", type=str, default="auto")
    p.add_argument("--logprobs_dtype", type=str, default="float32")

    # CLT patching sweep (optional).
    p.add_argument("--run_clt_patching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--clt_repo",
        type=str,
        default=str(ROOT / "clt_bundles" / "gemma-scope-2b-pt-res_sweep_smoke"),
        help="Local path (or HF repo id) for a CLT bundle.",
    )
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_layers", type=str, default="4,8,12,16,20,24")
    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--clt_decode_strategy", type=str, default="delta_1decode", choices=["safe_2decode", "delta_1decode"])
    p.add_argument("--clt_dtype_policy", type=str, default="clt", choices=["clt", "model"])
    p.add_argument("--clt_eps_active", type=float, default=1e-6)

    p.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.jobs) < 1:
        raise ValueError("--jobs must be >= 1")

    raw_pairs = Path(str(args.raw_pairs)).expanduser().resolve()
    if not raw_pairs.exists():
        raise FileNotFoundError(f"--raw_pairs not found: {str(raw_pairs)}")

    # Resolve default model snapshot paths when unspecified.
    hub_dir = _default_hf_hub_dir()
    base_model = str(args.base_model_name_or_path).strip()
    it_model = str(args.it_model_name_or_path).strip()
    if not base_model:
        snap = _find_hf_snapshot_dir(hub_dir=hub_dir, model_key="models--google--gemma-2-2b")
        if snap is None:
            raise FileNotFoundError(
                "Could not auto-detect Gemma base snapshot in HF cache. "
                "Pass --base_model_name_or_path explicitly."
            )
        base_model = str(snap)
    if not it_model:
        snap = _find_hf_snapshot_dir(hub_dir=hub_dir, model_key="models--google--gemma-2-2b-it")
        if snap is None:
            raise FileNotFoundError(
                "Could not auto-detect Gemma-IT snapshot in HF cache. "
                "Pass --it_model_name_or_path explicitly."
            )
        it_model = str(snap)

    chat_tok = str(args.chat_tokenizer_name_or_path).strip() or it_model

    # Results dir.
    results_dir = str(args.results_dir).strip()
    if not results_dir:
        ts = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_dir = str(ROOT / "results" / f"safety_chat_{ts}")
    results_dir_path = Path(results_dir).expanduser().resolve()
    datasets_dir = results_dir_path / "datasets"
    splits_dir = results_dir_path / "splits"
    results_dir_path.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1) Convert raw -> DisambPair with desired continuations.
    converted = datasets_dir / f"safety_disamb_pairs_{str(args.choice_profile)}.jsonl"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "convert_safety_pairs_to_disamb.py"),
            "--in_jsonl",
            str(raw_pairs),
            "--out_jsonl",
            str(converted),
            "--safety_choice_profile",
            str(args.choice_profile),
        ],
        dry_run=bool(args.dry_run),
    )

    # 2) Render prompts: user_only (strip cloze suffix), and chat_prerendered for the IT chat template.
    user_only = datasets_dir / f"{converted.stem}.user_only.jsonl"
    chat = datasets_dir / f"{converted.stem}.chat_prerendered.jsonl"

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_disamb_chat_prompts.py"),
            "--in_jsonl",
            str(converted),
            "--out_jsonl",
            str(user_only),
            "--mode",
            "user_only",
        ],
        dry_run=bool(args.dry_run),
    )

    render_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "render_disamb_chat_prompts.py"),
        "--in_jsonl",
        str(converted),
        "--out_jsonl",
        str(chat),
        "--mode",
        "chat_prerendered",
        "--tokenizer_name_or_path",
        str(chat_tok),
        "--add_generation_prompt",
    ]
    if bool(args.local_files_only):
        render_cmd.append("--local_files_only")
    _run(render_cmd, dry_run=bool(args.dry_run))

    # 3) Deterministic dev/test split stratified by csd/ih prefix.
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_disamb_dev_test.py"),
            "--in_jsonl",
            str(chat),
            "--out_dir",
            str(splits_dir),
            "--seed",
            str(int(args.split_seed)),
            "--dev_n",
            str(int(args.dev_n)),
            "--groups",
            "csd,ih",
        ],
        dry_run=bool(args.dry_run),
    )

    stem = chat.stem
    combined_test = splits_dir / f"{stem}_test.jsonl"
    csd_test = splits_dir / f"{stem}_csd_test.jsonl"
    ih_test = splits_dir / f"{stem}_ih_test.jsonl"

    # 4) Run baselines (+ optional CLT patching) on CSD and IH separately (test only).
    models = [
        ("gemma2b_base", base_model),
        ("gemma2b_it", it_model),
    ]

    common_eval: List[str] = []
    if bool(args.local_files_only):
        common_eval.append("--local_files_only")
    common_eval.extend(
        [
            "--device",
            str(args.device),
            "--attn_implementation",
            str(args.attn_implementation),
            "--score_batch_size",
            str(int(args.score_batch_size)),
            "--seed",
            str(int(args.seed)),
            "--bootstrap_n",
            str(int(args.bootstrap_n)),
            "--bootstrap_seed",
            str(int(args.bootstrap_seed)),
            "--ci",
            str(float(args.ci)),
            "--torch_dtype",
            str(args.torch_dtype),
            "--logprobs_dtype",
            str(args.logprobs_dtype),
            # Safety run is DISAMB-only; skip CF/COH to keep results interpretable and runtime reasonable.
            "--cf_path",
            "",
            "--coh_path",
            "",
            "--prompt_mode",
            "raw",
            "--prompt_input",
            "full_prompt",
            # Ensure continuation scoring never merges with the last prompt token.
            "--normalize_boundaries",
            # We intentionally run DISAMB-only.
            "--composite_missing_policy",
            "ignore",
        ]
    )

    def _eval_cmd(
        *,
        model_path: str,
        disamb_path: Path,
        out_csv: Path,
        run_clt: bool,
    ) -> List[str]:
        cmd: List[str] = [
            sys.executable,
            str(ROOT / "aom_eval.py"),
            "--model_name_or_path",
            str(model_path),
            "--disamb_path",
            str(disamb_path),
            "--csv_path",
            str(out_csv),
        ]
        if run_clt:
            cmd.extend(
                [
                    "--run_clt_patching",
                    "--clt_repo",
                    str(args.clt_repo),
                    "--clt_width",
                    str(args.clt_width),
                    "--clt_layers",
                    str(args.clt_layers),
                    "--clt_scale",
                    str(float(args.clt_scale)),
                    "--clt_dtype",
                    str(args.clt_dtype),
                    "--clt_decode_strategy",
                    str(args.clt_decode_strategy),
                    "--clt_dtype_policy",
                    str(args.clt_dtype_policy),
                    "--clt_eps_active",
                    str(float(args.clt_eps_active)),
                ]
            )
        cmd.extend(common_eval)
        return cmd

    datasets: list[tuple[str, Path]] = [
        ("safety_chat_csd_test", csd_test),
        ("safety_chat_ih_test", ih_test),
    ]
    if bool(args.include_combined):
        datasets.insert(0, ("safety_chat_test", combined_test))

    jobs: list[list[str]] = []
    for model_slug, model_path in models:
        for ds_slug, ds_path in datasets:
            if bool(args.run_clt_patching):
                if bool(args.also_run_baseline):
                    out_csv = results_dir_path / f"{model_slug}_{ds_slug}_baseline_seed{int(args.seed)}.csv"
                    jobs.append(_eval_cmd(model_path=str(model_path), disamb_path=ds_path, out_csv=out_csv, run_clt=False))
                out_csv_clt = results_dir_path / f"{model_slug}_{ds_slug}_clt_seed{int(args.seed)}.csv"
                jobs.append(_eval_cmd(model_path=str(model_path), disamb_path=ds_path, out_csv=out_csv_clt, run_clt=True))
            else:
                out_csv = results_dir_path / f"{model_slug}_{ds_slug}_baseline_seed{int(args.seed)}.csv"
                jobs.append(_eval_cmd(model_path=str(model_path), disamb_path=ds_path, out_csv=out_csv, run_clt=False))

    if bool(args.dry_run):
        for cmd in jobs:
            _run(cmd, dry_run=True)
    else:
        procs: list[subprocess.Popen] = []
        pending = list(jobs)
        while pending or procs:
            while pending and len(procs) < int(args.jobs):
                cmd = pending.pop(0)
                print(_shlex_join(cmd), flush=True)
                procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
            # Poll for completion.
            alive: list[subprocess.Popen] = []
            for p in procs:
                rc = p.poll()
                if rc is None:
                    alive.append(p)
                    continue
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, p.args)
            procs = alive
            time.sleep(0.25)

    print(f"\nWrote run outputs under: {str(results_dir_path)}", flush=True)


if __name__ == "__main__":
    main()
