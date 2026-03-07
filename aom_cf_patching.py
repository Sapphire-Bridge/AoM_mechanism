from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aom.config import load_config, resolve_relative_paths, sha256_text as sha256_text_config, validate_config_keys
from aom.data.bundle_manifest import validate_bundle_manifest
from aom.data.dataset_manifest import DatasetLoadError
from aom.data.loaders import load_counterfactual_pairs_with_manifest
from aom.provenance.protocol import (
    ProtocolArgBinding,
    coerce_bool as _coerce_bool,
    enforce_protocol_bindings,
    resolve_protocol_provenance as _resolve_protocol_provenance_shared,
)
from aom.repro import ReproConfig, collect_versions, get_git_commit_hash, seed_everything
from aom.run_manifest import build_run_manifest, redact_argv, write_run_manifest
from aom.run_summary import ErrorPolicy, RunSummary, normalize_error_thresholds


def _parse_int_list(s: str) -> Optional[List[int]]:
    s = str(s or "").strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_protocol_provenance(*, protocol_path_raw: str, protocol_sha256_raw: str) -> tuple[str, str]:
    prov = _resolve_protocol_provenance_shared(
        protocol_path_raw=str(protocol_path_raw or ""),
        protocol_sha256_raw=str(protocol_sha256_raw or ""),
        require_path_for_sha=False,
    )
    return str(prov.protocol_path), str(prov.protocol_sha256)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="AoM-CF causal patching runner (context/intervention-swap).")
    p.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Optional YAML/JSON config file. Values set here become argparse defaults; explicit CLI flags still win. "
            "Relative paths inside configs are resolved relative to the repo root (directory containing aom_cf_patching.py)."
        ),
    )

    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--models", nargs="*", type=str, default=None, help="Optional list of models to sweep.")
    p.add_argument("--revision", type=str, default=None, help="Optional HF model revision (branch/tag/commit SHA).")
    p.add_argument("--tokenizer_revision", type=str, default=None, help="Optional HF tokenizer revision.")
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading models that require custom code from the Hugging Face repo (use with care).",
    )
    p.add_argument("--local_files_only", action="store_true", help="Disallow downloads (offline mode).")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        help="Dtype: float32, float16, bfloat16, auto (default: float16 on MPS, bfloat16 for Qwen, float32 otherwise).",
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation. Use 'eager' for activation patching.",
    )
    p.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU (e.g., 'auto').")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--score_batch_size", type=int, default=8, help="Batch size for continuation scoring (default: 8).")
    p.add_argument(
        "--use_prefix_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse prompt KV cache across continuations when supported (default: True).",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep_seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
        help="Determinism mode: strict/best_effort/off (default: best_effort).",
    )

    p.add_argument("--cf_path", type=str, default=str(root / "data" / "counterfactual.jsonl"))
    p.add_argument(
        "--dataset_manifest_path",
        type=str,
        default="",
        help="Optional paper DATASET_MANIFEST.json path. If set, SHA256-gate datasets against it.",
    )
    p.add_argument(
        "--protocol_path",
        type=str,
        default="",
        help="Optional frozen protocol file path; SHA256 is recorded in rows and run manifest.",
    )
    p.add_argument(
        "--protocol_sha256",
        type=str,
        default="",
        help="Optional explicit protocol SHA256 (64 hex). If --protocol_path is set and this is empty, hash is computed.",
    )
    p.add_argument(
        "--require_frozen_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require protocol.{name,version,prereg_tag,status=frozen} when resolving --protocol_path.",
    )

    p.add_argument("--patch_layers", type=str, default="", help="Comma-separated layers to patch (default: all).")
    p.add_argument(
        "--max_total_len_delta",
        type=int,
        default=5,
        help="Skip items where abs(len(base_ids)-len(cf_ids)) exceeds this (default: 5).",
    )
    p.add_argument(
        "--span_mode",
        type=str,
        default="divergent_only",
        choices=["divergent_only", "divergent_plus_downstream", "left_aligned_truncated"],
        help="Patch span selection policy (default: divergent_only).",
    )
    p.add_argument(
        "--include_expected_effects",
        type=str,
        default="shift,invariant",
        help="Comma-separated expected_effect values to include (default: 'shift,invariant'). Add 'graded' to include graded items.",
    )

    p.add_argument("--no_length_norm", action="store_true", help="Use sum logprob instead of mean logprob.")
    p.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap replicates for CIs.")
    p.add_argument("--bootstrap_seed", type=int, default=42, help="RNG seed for bootstrap CIs.")
    p.add_argument("--ci", type=float, default=0.95, help="Confidence level for bootstrap CIs.")

    p.add_argument(
        "--error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip", "skip_silent"],
        help="How to handle per-run exceptions in a model/seed sweep (default: warn_skip).",
    )
    p.add_argument(
        "--data_error_policy",
        type=str,
        default="warn_skip",
        choices=["raise", "warn_skip"],
        help="How to handle invalid dataset rows (default: warn_skip).",
    )
    p.add_argument(
        "--strict_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any dataset rows are invalid (equivalent to --data_error_policy raise).",
    )
    p.add_argument(
        "--strict_errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any run failures/skips occurred (equivalent to --max_fail_rate 0 --max_skips 0).",
    )
    p.add_argument(
        "--max_fail_rate",
        type=float,
        default=1.0,
        help="Fail if failed/attempted exceeds this (default: 1.0 disables).",
    )
    p.add_argument("--max_skips", type=int, default=-1, help="Fail if skipped exceeds this (default: -1 disables).")
    p.add_argument(
        "--require_git",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a non-empty git commit hash for provenance (default: True). Disable with --no-require_git.",
    )

    p.add_argument("--csv_path", type=str, default="")
    p.add_argument("--results_dir", type=str, default="results", help="Directory for results artifacts (CSV + manifest).")
    p.add_argument("--run_name", type=str, default="", help="Base name for outputs when --csv_path is unset.")

    # Apply config file as defaults (CLI flags override).
    cfg_ns, _unknown = p.parse_known_args()
    cfg_path = str(getattr(cfg_ns, "config", "") or "").strip()
    cfg_sha256 = ""
    if cfg_path:
        cfg_p = Path(cfg_path)
        cfg_raw = cfg_p.read_text(encoding="utf-8")
        cfg_sha256 = sha256_text_config(cfg_raw)
        cfg = load_config(cfg_p)
        cfg = resolve_relative_paths(cfg, base_dir=root)
        allowed = {a.dest for a in p._actions if getattr(a, "dest", None)}
        validate_config_keys(config=cfg, allowed=allowed)
        p.set_defaults(**cfg)

    args = p.parse_args()
    setattr(args, "config_path", cfg_path)
    setattr(args, "config_sha256", cfg_sha256)
    return args


def _infer_input_device(model) -> "torch.device":
    import torch

    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            dev = emb.weight.device
            if dev.type != "meta":
                return dev
    except Exception:
        pass
    try:
        dev = next(model.parameters()).device
        if dev.type != "meta":
            return dev
    except Exception:
        pass
    return torch.device("cpu")


def _model_param_dtype(model) -> str:
    try:
        p = next(model.parameters())
        return str(p.dtype)
    except Exception:
        return ""


def main() -> None:
    run_t0 = time.perf_counter()
    run_started_at_utc = _utc_now_iso()
    try:
        args = parse_args()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(2) from e
    try:
        protocol_prov = _resolve_protocol_provenance_shared(
            protocol_path_raw=str(getattr(args, "protocol_path", "") or ""),
            protocol_sha256_raw=str(getattr(args, "protocol_sha256", "") or ""),
            require_path_for_sha=False,
            require_frozen=bool(getattr(args, "require_frozen_protocol", False)),
        )
        setattr(args, "protocol_path", str(protocol_prov.protocol_path))
        setattr(args, "protocol_sha256", str(protocol_prov.protocol_sha256))
        setattr(args, "protocol_sha256_verified", bool(protocol_prov.protocol_sha256_verified))
        setattr(args, "protocol_sha256_source", str(protocol_prov.protocol_sha256_source))
        setattr(args, "protocol_name", str(protocol_prov.protocol_name))
        setattr(args, "protocol_version", str(protocol_prov.protocol_version))
        setattr(args, "protocol_prereg_tag", str(protocol_prov.protocol_prereg_tag))
        if str(protocol_prov.protocol_path):
            enforce_protocol_bindings(
                args=args,
                protocol_config=protocol_prov.protocol_config,
                bindings=[
                    ProtocolArgBinding("bootstrap_n", ("bootstrap", "n"), int),
                    ProtocolArgBinding("bootstrap_seed", ("bootstrap", "seed"), int),
                    ProtocolArgBinding("ci", ("bootstrap", "ci"), float),
                    ProtocolArgBinding("seed", ("splits", "random_seed"), int, required=False),
                    ProtocolArgBinding("require_git", ("repro", "require_git"), _coerce_bool),
                ],
                context="aom_cf_patching",
                protocol_path=str(protocol_prov.protocol_path),
                protocol_sha256=str(protocol_prov.protocol_sha256),
            )
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(2) from e
    error_policy: ErrorPolicy = str(getattr(args, "error_policy", "warn_skip"))  # type: ignore[assignment]
    strict_data = bool(getattr(args, "strict_data", False))
    data_error_policy = "raise" if strict_data else str(getattr(args, "data_error_policy", "warn_skip"))

    summary = RunSummary()
    fatal_error: Exception | None = None
    rows: List[Dict[str, Any]] = []
    attempted_expected: int | None = None
    n_models: int | None = None
    n_seeds: int | None = None
    repro: dict[str, Any] | None = None
    versions: dict[str, str] | None = None
    device_backend: str | None = None
    datasets: dict[str, Any] | None = None
    dataset_warnings: list[str] = []

    def _warn(msg: str) -> None:
        if error_policy != "skip_silent":
            print(str(msg), file=sys.stderr, flush=True)

    csv_path = str(getattr(args, "csv_path", "") or "").strip()
    if not csv_path:
        results_dir = Path(str(getattr(args, "results_dir", "results")))
        run_name = str(getattr(args, "run_name", "") or "").strip() or "cf_patching"
        csv_path = str(results_dir / f"{run_name}.csv")
    csv_p = Path(csv_path)
    manifest_path = str(csv_p.with_suffix(".manifest.json"))

    try:
        from aom.interventions.patching.base import run_activation_patching
        from aom.interventions.patching.cf_protocol import CFPatchingConfig, CFInterventionSwapProtocol
        from aom.models.loader import load_causal_lm
        from aom.utils import configure_logprob_computation, configure_scoring_performance, get_best_device, set_seed

        import torch

        configure_logprob_computation(
            logprobs_dtype=getattr(torch, "float32"),
            strict_finite=True,
        )
        configure_scoring_performance(
            score_batch_size=int(getattr(args, "score_batch_size", 1)),
            use_prefix_cache=bool(getattr(args, "use_prefix_cache", False)),
        )
        if str(getattr(args, "attn_implementation", "eager")) != "eager":
            raise ValueError("Activation patching requires --attn_implementation eager")

        models = args.models if args.models is not None and len(args.models) > 0 else [args.model_name_or_path]
        seeds = args.sweep_seeds if args.sweep_seeds is not None and len(args.sweep_seeds) > 0 else [args.seed]
        n_models = int(len(models))
        n_seeds = int(len(seeds))
        attempted_expected = int(n_models * n_seeds)

        datasets = {}
        cf_items_raw, cf_manifest = load_counterfactual_pairs_with_manifest(
            str(args.cf_path),
            role="cf",
            error_policy=data_error_policy,
        )
        datasets["cf"] = cf_manifest.as_dict()
        invalid = int(datasets["cf"].get("n_rows_invalid", 0) or 0)
        total = int(datasets["cf"].get("n_rows_total", 0) or 0)
        valid = int(datasets["cf"].get("n_rows_valid", 0) or 0)
        if total > 0 and valid == 0:
            raise ValueError(f"Dataset cf has {invalid}/{total} invalid rows and 0 valid rows")
        if invalid > 0:
            msg = f"dataset cf: invalid_rows {invalid} of {total}"
            dataset_warnings.append(msg)
            _warn(f"[WARN] {msg}")

        repo_root = Path(__file__).resolve().parent
        git_commit_hash = get_git_commit_hash(repo_root=repo_root, required=bool(getattr(args, "require_git", True)))
        argv_redacted_list = redact_argv(sys.argv)
        argv_redacted_json = json.dumps(argv_redacted_list, ensure_ascii=False)
        argv_sha256 = hashlib.sha256(argv_redacted_json.encode("utf-8")).hexdigest()

        dataset_bundle_info: dict[str, str] | None = None
        dataset_manifest_path_raw = str(getattr(args, "dataset_manifest_path", "") or "").strip()
        if dataset_manifest_path_raw:
            dataset_bundle_info = validate_bundle_manifest(
                dataset_manifest_path_raw,
                cf_path=str(getattr(args, "cf_path", "") or ""),
            )
        setattr(
            args,
            "dataset_bundle_manifest_sha256",
            "" if dataset_bundle_info is None else dataset_bundle_info["dataset_bundle_manifest_sha256"],
        )
        setattr(
            args,
            "dataset_bundle_manifest_name",
            "" if dataset_bundle_info is None else dataset_bundle_info["dataset_bundle_manifest_name"],
        )
        setattr(
            args,
            "dataset_bundle_id",
            "" if dataset_bundle_info is None else dataset_bundle_info["dataset_bundle_id"],
        )

        # Resolve base device once (unless using device_map).
        if args.device == "auto":
            base_device = get_best_device()
        else:
            base_device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])
        device_backend = str(getattr(base_device, "type", str(base_device)))

        determinism = str(getattr(args, "determinism", "best_effort"))
        repro = seed_everything(ReproConfig(seed=int(seeds[0]), determinism=determinism), device=base_device)
        repro["seeds"] = [int(s) for s in seeds]
        versions = collect_versions()

        layer_list = _parse_int_list(str(getattr(args, "patch_layers", "") or ""))
        include_effects_raw = str(getattr(args, "include_expected_effects", "shift,invariant"))
        include_effects = tuple(s.strip() for s in include_effects_raw.split(",") if s.strip())
        protocol = CFInterventionSwapProtocol(
            config=CFPatchingConfig(
                max_total_len_delta=int(getattr(args, "max_total_len_delta", 5)),
                span_mode=str(getattr(args, "span_mode", "divergent_only")),
                include_expected_effects=include_effects,
            )
        )

        for model_name in models:
            print(
                "Loading model "
                f"model={model_name!r} "
                f"local_files_only={bool(args.local_files_only)} "
                f"torch_dtype={args.torch_dtype!r} "
                f"attn_implementation={args.attn_implementation!r} "
                f"device_map={args.device_map!r}",
                flush=True,
            )
            try:
                loaded = load_causal_lm(
                    model_name,
                    device=base_device,
                    torch_dtype=args.torch_dtype,
                    revision=getattr(args, "revision", None),
                    tokenizer_revision=getattr(args, "tokenizer_revision", None),
                    local_files_only=args.local_files_only,
                    trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
                    attn_implementation=args.attn_implementation,
                    device_map=args.device_map,
                )
            except Exception as e:
                for _s in seeds:
                    summary.record_failure(e)
                _warn(f"[fail] model={model_name!r}: {type(e).__name__}: {e}")
                if error_policy == "raise":
                    raise
                continue

            model = loaded.model
            tokenizer = loaded.tokenizer
            print(f"Loaded {model_name} (arch={loaded.architecture})", flush=True)

            tensor_device = _infer_input_device(model) if args.device_map is not None else base_device

            for s in seeds:
                eval_t0 = time.perf_counter()
                eval_started_at_utc = _utc_now_iso()
                try:
                    set_seed(int(s))
                    patch_res = run_activation_patching(
                        model=model,
                        tokenizer=tokenizer,
                        protocol=protocol,
                        items=cf_items_raw,
                        device=tensor_device,
                        layers=layer_list,
                        normalize_by_length=not bool(getattr(args, "no_length_norm", False)),
                        ci=float(getattr(args, "ci", 0.95)),
                        bootstrap_n=int(getattr(args, "bootstrap_n", 1000)),
                        bootstrap_seed=int(getattr(args, "bootstrap_seed", 42)),
                    )
                except Exception as e:
                    summary.record_failure(e)
                    _warn(f"[fail] model={model_name!r} seed={int(s)}: {type(e).__name__}: {e}")
                    if error_policy == "raise":
                        raise
                    continue
                eval_ended_at_utc = _utc_now_iso()
                eval_wall_time_sec = float(time.perf_counter() - eval_t0)

                row: Dict[str, Any] = {
                    "model": str(model_name),
                    "arch": str(loaded.architecture),
                    "seed": int(s),
                    "torch_version": "" if versions is None else str(versions.get("torch", "")),
                    "transformers_version": "" if versions is None else str(versions.get("transformers", "")),
                    "tokenizers_version": "" if versions is None else str(versions.get("tokenizers", "")),
                    "python_version": "" if versions is None else str(versions.get("python", "")),
                    "config_path": str(getattr(args, "config_path", "") or ""),
                    "config_sha256": str(getattr(args, "config_sha256", "") or ""),
                    "protocol_path": str(getattr(args, "protocol_path", "") or ""),
                    "protocol_sha256": str(getattr(args, "protocol_sha256", "") or ""),
                    "protocol_sha256_verified": bool(getattr(args, "protocol_sha256_verified", False)),
                    "protocol_sha256_source": str(getattr(args, "protocol_sha256_source", "") or ""),
                    "protocol_name": str(getattr(args, "protocol_name", "") or ""),
                    "protocol_version": str(getattr(args, "protocol_version", "") or ""),
                    "protocol_prereg_tag": str(getattr(args, "protocol_prereg_tag", "") or ""),
                    "dataset_manifest_path": str(getattr(args, "dataset_manifest_path", "") or ""),
                    "dataset_bundle_manifest_sha256": str(getattr(args, "dataset_bundle_manifest_sha256", "") or ""),
                    "dataset_bundle_manifest_name": str(getattr(args, "dataset_bundle_manifest_name", "") or ""),
                    "dataset_bundle_id": str(getattr(args, "dataset_bundle_id", "") or ""),
                    "cf_path": str(getattr(args, "cf_path", "")),
                    "cf_sha256": _sha256_file(Path(str(getattr(args, "cf_path", "")))),
                    "device": str(tensor_device),
                    "requested_device": str(getattr(args, "device", "")),
                    "device_map": str(getattr(args, "device_map", "")),
                    "attn_implementation": str(getattr(args, "attn_implementation", "")),
                    "torch_dtype_requested": str(getattr(args, "torch_dtype", "")),
                    "model_param_dtype": _model_param_dtype(model),
                    "hf_revision_requested": str(getattr(args, "revision", "") or ""),
                    "hf_tokenizer_revision_requested": str(getattr(args, "tokenizer_revision", "") or ""),
                    "hf_tokenizer_revision_effective": str(loaded.tokenizer_revision_effective or ""),
                    "hf_local_files_only": bool(getattr(args, "local_files_only", False)),
                    "hf_trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
                    "hf_model_commit_hash": str(loaded.model_commit_hash or ""),
                    "git_commit": str(git_commit_hash),
                    "argv_redacted_json": str(argv_redacted_json),
                    "argv_sha256": str(argv_sha256),
                    "eval_started_at_utc": str(eval_started_at_utc),
                    "eval_ended_at_utc": str(eval_ended_at_utc),
                    "eval_wall_time_sec": float(eval_wall_time_sec),
                    "bootstrap_n": int(getattr(args, "bootstrap_n", 0)),
                    "bootstrap_seed": int(getattr(args, "bootstrap_seed", 0)),
                    "ci": float(getattr(args, "ci", 0.0)),
                    "max_total_len_delta": int(getattr(args, "max_total_len_delta", 5)),
                    "span_mode": str(getattr(args, "span_mode", "divergent_only")),
                }
                row.update({f"cf_patch_{k}": v for k, v in patch_res.items()})
                rows.append(row)
                summary.record_success()
                print(
                    f"model={row['model']} seed={row['seed']} cf_patch_mean_max_effect={row.get('cf_patch_mean_max_effect', float('nan'))}",
                    flush=True,
                )

            # Free model before next one (important for large Qwen variants).
            del model
            del tokenizer
            del loaded

    except Exception as e:
        fatal_error = e
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[FATAL] Aborting; run manifest will be written to {manifest_path}", file=sys.stderr, flush=True)

    run_ended_at_utc = _utc_now_iso()
    run_wall_time_sec = float(time.perf_counter() - run_t0)
    for r in rows:
        r["started_at_utc"] = str(run_started_at_utc)
        r["ended_at_utc"] = str(run_ended_at_utc)
        r["wall_time_sec"] = float(run_wall_time_sec)

    # Always write artifacts (CSV if we have rows; manifest always).
    csv_sha256 = None
    if rows:
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, str(csv_p))
        if csv_p.exists():
            csv_sha256 = _sha256_file(csv_p)

    max_skips_raw = int(getattr(args, "max_skips", -1))
    strict_errors = bool(getattr(args, "strict_errors", False))
    max_fail_rate, max_skips = normalize_error_thresholds(
        max_fail_rate=float(getattr(args, "max_fail_rate", 1.0)),
        max_skips=max_skips_raw,
        strict_errors=strict_errors,
    )
    max_skips_manifest = int(max_skips_raw if max_skips is None else max_skips)
    if fatal_error is not None:
        run_status = "FAIL"
        run_status_reasons = [f"fatal: {type(fatal_error).__name__}"]
    else:
        run_status, run_status_reasons = summary.evaluate(
            max_fail_rate=max_fail_rate,
            max_skips=max_skips,
        )
        if dataset_warnings:
            if run_status == "PASS":
                run_status = "WARN"
            for msg in dataset_warnings:
                if msg not in run_status_reasons:
                    run_status_reasons.append(str(msg))

    if fatal_error is None and attempted_expected is not None and int(summary.attempted) != int(attempted_expected):
        msg = f"attempted {int(summary.attempted)} != attempted_expected {int(attempted_expected)}"
        if run_status == "PASS":
            run_status = "WARN"
        if msg not in run_status_reasons:
            run_status_reasons.append(msg)

    if rows:
        manifest_row: Dict[str, Any] = dict(rows[0])
        if len(rows) != 1:
            manifest_row["_manifest_note"] = f"CSV contains {len(rows)} rows; manifest stores the first row only."
    else:
        manifest_row = {"_manifest_note": "No successful results rows were produced."}
        manifest_row["error_policy"] = str(error_policy)
        manifest_row["strict_errors"] = bool(strict_errors)
        manifest_row["max_fail_rate"] = float(max_fail_rate)
        manifest_row["max_skips"] = int(max_skips_manifest)
        if fatal_error is not None:
            manifest_row["fatal_error_type"] = str(type(fatal_error).__name__)
            manifest_row["fatal_error"] = str(fatal_error)

    dataset_manifest_path = str(getattr(args, "dataset_manifest_path", "") or "").strip() or None
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=manifest_row,
        dataset_manifest_path=dataset_manifest_path,
        csv_path=str(csv_p) if csv_p.exists() else None,
        csv_sha256=str(csv_sha256) if csv_sha256 is not None else None,
        csv_n_rows=int(len(rows)),
    )
    manifest["started_at_utc"] = str(run_started_at_utc)
    manifest["ended_at_utc"] = str(run_ended_at_utc)
    manifest["wall_time_sec"] = float(run_wall_time_sec)
    if dataset_manifest_path is not None:
        manifest["dataset_bundle_manifest_sha256"] = str(getattr(args, "dataset_bundle_manifest_sha256", "") or "")
        manifest["dataset_bundle_manifest_name"] = str(getattr(args, "dataset_bundle_manifest_name", "") or "")
        manifest["dataset_bundle_id"] = str(getattr(args, "dataset_bundle_id", "") or "")
    config_path = str(getattr(args, "config_path", "") or "").strip()
    if config_path:
        manifest["config_path"] = str(config_path)
        manifest["config_sha256"] = str(getattr(args, "config_sha256", "") or "")
    protocol_path = str(getattr(args, "protocol_path", "") or "").strip()
    protocol_sha256 = str(getattr(args, "protocol_sha256", "") or "").strip()
    protocol_sha256_verified = bool(getattr(args, "protocol_sha256_verified", False))
    protocol_sha256_source = str(getattr(args, "protocol_sha256_source", "") or "").strip()
    protocol_name = str(getattr(args, "protocol_name", "") or "").strip()
    protocol_version = str(getattr(args, "protocol_version", "") or "").strip()
    protocol_prereg_tag = str(getattr(args, "protocol_prereg_tag", "") or "").strip()
    if protocol_path:
        manifest["protocol_path"] = str(protocol_path)
    if protocol_sha256:
        manifest["protocol_sha256"] = str(protocol_sha256)
    manifest["protocol_sha256_verified"] = bool(protocol_sha256_verified)
    if protocol_sha256_source:
        manifest["protocol_sha256_source"] = str(protocol_sha256_source)
    if protocol_name:
        manifest["protocol_name"] = str(protocol_name)
    if protocol_version:
        manifest["protocol_version"] = str(protocol_version)
    if protocol_prereg_tag:
        manifest["protocol_prereg_tag"] = str(protocol_prereg_tag)
    manifest["error_policy"] = str(error_policy)
    manifest["strict_errors"] = bool(strict_errors)
    manifest["max_fail_rate"] = float(max_fail_rate)
    manifest["max_skips"] = int(max_skips_manifest)
    manifest["data_error_policy"] = str(data_error_policy)
    manifest["strict_data"] = bool(strict_data)
    manifest["attempt_unit"] = "model_seed"
    manifest["attempted_expected"] = attempted_expected
    manifest["n_models"] = n_models
    manifest["n_seeds"] = n_seeds
    if repro is not None:
        manifest["repro"] = dict(repro)
    if versions is not None:
        manifest["versions"] = dict(versions)
    if device_backend is not None:
        manifest["device_backend"] = str(device_backend)
    if datasets is not None:
        manifest["datasets"] = dict(datasets)
    manifest["run_status"] = str(run_status)
    manifest["run_status_reasons"] = list(run_status_reasons)
    manifest["run_summary"] = summary.as_dict()
    write_run_manifest(manifest_path, manifest)

    if run_status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
