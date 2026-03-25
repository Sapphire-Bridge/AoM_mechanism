from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aom.data.loaders import (
    load_coherence_items_with_manifest,
    load_counterfactual_pairs_with_manifest,
    load_disamb_pairs_with_manifest,
)
from aom.data.bundle_manifest import validate_bundle_manifest
from aom.data.dataset_manifest import DatasetLoadError
from aom.config import load_config, resolve_relative_paths, sha256_text as sha256_text_config, validate_config_keys
from aom.prompting import (
    PromptRenderConfig,
    normalize_prompt_boundary,
    render_prompt,
    sha256_chat_template,
    sha256_text,
    validate_prompt_boundary,
)
from aom.provenance.protocol import (
    ProtocolArgBinding,
    coerce_bool as _coerce_bool,
    enforce_protocol_bindings,
    resolve_protocol_provenance as _resolve_protocol_provenance_shared,
)
from aom.repro import ReproConfig, collect_versions, get_git_commit_hash, seed_everything
from aom.models.loader import load_causal_lm
from aom.run_manifest import build_run_manifest, redact_argv, write_run_manifest
from aom.run_summary import ErrorPolicy, RunSummary, normalize_error_thresholds
from aom.utils import get_best_device


def _parse_int_list(s: str) -> Optional[List[int]]:
    s = s.strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _patching_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "run_patching", False)
        or getattr(args, "run_sae_patching", False)
        or getattr(args, "run_clt_patching", False)
        or getattr(args, "run_patching_specificity", False)
    )


def _enforce_device_map_patching_guard(args: argparse.Namespace) -> None:
    if getattr(args, "device_map", None) is None:
        return
    if not _patching_requested(args):
        return
    raise ValueError(
        "Patching runs do not support --device_map. Disable --device_map or disable patching "
        "(--run_patching/--run_sae_patching/--run_clt_patching/--run_patching_specificity)."
    )


def _resolve_protocol_provenance(*, protocol_path_raw: str, protocol_sha256_raw: str) -> tuple[str, str]:
    prov = _resolve_protocol_provenance_shared(
        protocol_path_raw=str(protocol_path_raw or ""),
        protocol_sha256_raw=str(protocol_sha256_raw or ""),
        require_path_for_sha=False,
    )
    return str(prov.protocol_path), str(prov.protocol_sha256)


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


def _validate_prompt_rendering_args(args: argparse.Namespace) -> None:
    prompt_input = str(getattr(args, "prompt_input", "full_prompt"))
    prompt_mode = str(getattr(args, "prompt_mode", "raw"))
    if prompt_input == "full_prompt" and prompt_mode == "chat_template":
        raise ValueError(
            "chat_template mode expects user_message inputs; "
            "use --prompt_input user_message or set --prompt_mode raw"
        )


def _load_system_prompt(args: argparse.Namespace) -> str | None:
    prompt_mode = str(getattr(args, "prompt_mode", "raw"))
    system_prompt_file = str(getattr(args, "system_prompt_file", "") or "").strip()
    system_prompt_inline = str(getattr(args, "system_prompt", "") or "")

    if system_prompt_file and system_prompt_inline.strip():
        raise ValueError("Use at most one of --system_prompt_file or --system_prompt")

    if prompt_mode != "chat_template" and (system_prompt_file or system_prompt_inline.strip()):
        raise ValueError("--system_prompt and --system_prompt_file are only valid with --prompt_mode chat_template")

    if system_prompt_file:
        return Path(system_prompt_file).read_text(encoding="utf-8")
    if system_prompt_inline.strip():
        return system_prompt_inline
    return None


def _apply_prompt_rendering(
    *,
    tokenizer,
    cfg: PromptRenderConfig,
    prompt_input: str,
    prompt_mode: str,
    disamb_items,
    cf_items,
    coh_items,
):
    if prompt_mode == "raw":
        return disamb_items, cf_items, coh_items
    if prompt_mode != "chat_template":
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")
    if prompt_input != "user_message":
        raise ValueError(
            "chat_template mode expects user_message inputs; "
            "use --prompt_input user_message or set --prompt_mode raw"
        )

    def _render_one(user_text: str) -> str:
        rendered, _meta = render_prompt(tokenizer, user_text, cfg)
        return rendered

    out_disamb = []
    for it in disamb_items:
        a_text = _render_one(it.a.prompt)
        b_text = _render_one(it.b.prompt)
        out_disamb.append(replace(it, a=replace(it.a, prompt=a_text), b=replace(it.b, prompt=b_text)))

    out_cf = []
    for it in cf_items:
        base_text = _render_one(it.base.prompt)
        cf_text = _render_one(it.cf.prompt)
        out_cf.append(replace(it, base=replace(it.base, prompt=base_text), cf=replace(it.cf, prompt=cf_text)))

    out_coh = []
    for it in coh_items:
        ctx = _render_one(it.context)
        out_coh.append(replace(it, context=ctx))

    return out_disamb, out_cf, out_coh


def _apply_prompt_boundary_policies(
    *,
    boundary_check: str,
    normalize_boundaries: bool,
    prompt_mode: str,
    disamb_items,
    cf_items,
    coh_items,
):
    if boundary_check not in {"off", "warn", "error"}:
        raise ValueError(f"Invalid boundary_check: {boundary_check!r}")

    if boundary_check == "off" and not normalize_boundaries:
        return disamb_items, cf_items, coh_items

    def _apply_one(prompt: str) -> str:
        updated = normalize_prompt_boundary(prompt) if normalize_boundaries else prompt
        if boundary_check == "warn":
            validate_prompt_boundary(updated, mode=str(prompt_mode))
        elif boundary_check == "error":
            if updated and not updated[-1].isspace():
                raise ValueError(
                    f"Prompt boundary may be unstable (mode={prompt_mode}): prompt does not end with whitespace."
                )
        return updated

    out_disamb = []
    for it in disamb_items:
        a_prompt = _apply_one(it.a.prompt)
        b_prompt = _apply_one(it.b.prompt)
        if normalize_boundaries:
            out_disamb.append(replace(it, a=replace(it.a, prompt=a_prompt), b=replace(it.b, prompt=b_prompt)))
        else:
            out_disamb.append(it)

    out_cf = []
    for it in cf_items:
        base_prompt = _apply_one(it.base.prompt)
        cf_prompt = _apply_one(it.cf.prompt)
        if normalize_boundaries:
            out_cf.append(replace(it, base=replace(it.base, prompt=base_prompt), cf=replace(it.cf, prompt=cf_prompt)))
        else:
            out_cf.append(it)

    out_coh = []
    for it in coh_items:
        context = _apply_one(it.context)
        if normalize_boundaries:
            out_coh.append(replace(it, context=context))
        else:
            out_coh.append(it)

    return out_disamb, out_cf, out_coh


def _prepare_datasets_for_eval(
    *,
    args: argparse.Namespace,
    tokenizer,
    disamb_items_raw,
    cf_items_raw,
    coh_items_raw,
):
    _validate_prompt_rendering_args(args)
    prompt_input = str(getattr(args, "prompt_input", "full_prompt"))
    prompt_mode = str(getattr(args, "prompt_mode", "raw"))
    add_generation_prompt = bool(getattr(args, "add_generation_prompt", True))
    system_prompt = _load_system_prompt(args)

    render_cfg = PromptRenderConfig(
        prompt_mode=prompt_mode,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )
    disamb_items, cf_items, coh_items = _apply_prompt_rendering(
        tokenizer=tokenizer,
        cfg=render_cfg,
        prompt_input=prompt_input,
        prompt_mode=prompt_mode,
        disamb_items=disamb_items_raw,
        cf_items=cf_items_raw,
        coh_items=coh_items_raw,
    )

    disamb_items, cf_items, coh_items = _apply_prompt_boundary_policies(
        boundary_check=str(getattr(args, "boundary_check", "warn")),
        normalize_boundaries=bool(getattr(args, "normalize_boundaries", False)),
        prompt_mode=prompt_mode,
        disamb_items=disamb_items,
        cf_items=cf_items,
        coh_items=coh_items,
    )
    system_prompt_sha256 = sha256_text(system_prompt)
    chat_template_sha256 = sha256_chat_template(tokenizer)
    return disamb_items, cf_items, coh_items, system_prompt_sha256, chat_template_sha256


def run_eval_with_loaded(
    *,
    model_name: str,
    model,
    tokenizer,
    architecture: str,
    hf_model_commit_hash: str | None,
    hf_tokenizer_revision_effective: str | None,
    system_prompt_sha256: str | None,
    chat_template_sha256: str | None,
    args: argparse.Namespace,
    seed: int,
    git_commit_hash: str,
    argv_redacted_json: str,
    argv_sha256: str,
    tensor_device,
    disamb_items,
    cf_items,
    coh_items,
) -> Dict[str, Any]:
    from aom.baselines import compute_disamb_keyword_baseline
    from aom.metrics.coherence import compute_aom_coh
    from aom.metrics.clt_cpt import compute_clt_cpt_context_swap_patching
    from aom.metrics.composite import compute_composite_metric
    from aom.metrics.counterfactual import compute_aom_cf
    from aom.metrics.disamb import compute_aom_disamb, compute_cpt_context_swap_patching, compute_cpt_target_specificity_control
    from aom.metrics.sae_patching import compute_sae_cpt_context_swap_patching
    from aom.utils import set_seed

    set_seed(seed)

    # Provenance / audit fields (helps compare across backends).
    import platform
    import sys

    import torch
    import transformers
    import tokenizers

    eval_t0 = time.perf_counter()
    eval_started_at_utc = datetime.now(timezone.utc).isoformat()

    def _sha256_path(p: str) -> str:
        p = str(p or "")
        if not p:
            return ""
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def _model_param_dtype(model) -> str:
        try:
            p = next(model.parameters())
            return str(p.dtype)
        except Exception:
            return ""

    t_disamb0 = time.perf_counter()
    disamb_res = compute_aom_disamb(
        model,
        tokenizer,
        disamb_items,
        tensor_device,
        normalize_by_length=not args.no_length_norm,
        ci=args.ci,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )
    eval_time_disamb_sec = float(time.perf_counter() - t_disamb0)

    t_cf0 = time.perf_counter()
    cf_res = compute_aom_cf(
        model,
        tokenizer,
        cf_items,
        tensor_device,
        normalize_by_length=not args.no_length_norm,
        ci=args.ci,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )
    eval_time_cf_sec = float(time.perf_counter() - t_cf0)

    t_coh0 = time.perf_counter()
    coh_res = compute_aom_coh(
        model,
        tokenizer,
        coh_items,
        tensor_device,
        normalize_by_length=not args.no_length_norm,
        ci=args.ci,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )
    eval_time_coh_sec = float(time.perf_counter() - t_coh0)

    out: Dict[str, Any] = {
        "model": model_name,
        "arch": architecture,
        "seed": seed,
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
        "disamb_path": str(getattr(args, "disamb_path", "")),
        "cf_path": str(getattr(args, "cf_path", "")),
        "coh_path": str(getattr(args, "coh_path", "")),
        "disamb_sha256": _sha256_path(str(getattr(args, "disamb_path", ""))),
        "cf_sha256": _sha256_path(str(getattr(args, "cf_path", ""))),
        "coh_sha256": _sha256_path(str(getattr(args, "coh_path", ""))),
        "device": str(tensor_device),
        "requested_device": str(getattr(args, "device", "")),
        "device_map": str(getattr(args, "device_map", "")),
        "attn_implementation": str(getattr(args, "attn_implementation", "")),
        "torch_dtype_requested": str(getattr(args, "torch_dtype", "")),
        "model_param_dtype": _model_param_dtype(model),
        "logprobs_dtype": str(getattr(args, "logprobs_dtype", "")),
        "score_batch_size": int(getattr(args, "score_batch_size", 1)),
        "use_prefix_cache": bool(getattr(args, "use_prefix_cache", False)),
        "strict_finite": bool(getattr(args, "strict_finite", True)),
        "strict_metrics": bool(getattr(args, "strict_metrics", False)),
        "composite_missing_policy": str(getattr(args, "composite_missing_policy", "nan")),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "tokenizers_version": str(getattr(tokenizers, "__version__", "")),
        "python_version": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "git_commit": str(git_commit_hash),
        "argv_redacted_json": str(argv_redacted_json),
        "argv_sha256": str(argv_sha256),
        "hf_revision_requested": str(getattr(args, "revision", "") or ""),
        "hf_tokenizer_revision_requested": str(getattr(args, "tokenizer_revision", "") or ""),
        "hf_tokenizer_revision_effective": str(hf_tokenizer_revision_effective or ""),
        "hf_local_files_only": bool(getattr(args, "local_files_only", False)),
        "hf_trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "hf_model_commit_hash": str(hf_model_commit_hash or ""),
        "prompt_input": str(getattr(args, "prompt_input", "full_prompt")),
        "prompt_mode": str(getattr(args, "prompt_mode", "raw")),
        "system_prompt_sha256": str(system_prompt_sha256 or ""),
        "chat_template_sha256": str(chat_template_sha256 or ""),
        "add_generation_prompt": bool(getattr(args, "add_generation_prompt", True)),
        "no_length_norm": bool(getattr(args, "no_length_norm", False)),
        "bootstrap_n": int(getattr(args, "bootstrap_n", 0)),
        "bootstrap_seed": int(getattr(args, "bootstrap_seed", 0)),
        "ci": float(getattr(args, "ci", 0.0)),
        **{f"disamb_{k}": v for k, v in disamb_res.items()},
        **{f"cf_{k}": v for k, v in cf_res.items()},
        **{f"coh_{k}": v for k, v in coh_res.items()},
    }
    composite_missing_policy = "fail" if bool(getattr(args, "strict_metrics", False)) else str(
        getattr(args, "composite_missing_policy", "nan")
    )
    composite = compute_composite_metric(
        disamb_res,
        cf_res,
        coh_res,
        missing_policy=composite_missing_policy,
    )
    out["composite_missing_policy"] = str(composite_missing_policy)
    out["aom_composite"] = float(composite.value)
    out["aom_composite_valid"] = bool(composite.valid)
    out["aom_composite_n"] = int(composite.n)
    if composite.reason is not None:
        out["aom_composite_reason"] = str(composite.reason)

    if not composite.valid:
        print(f"[WARN] Composite invalid: {composite.reason}", file=sys.stderr, flush=True)

    if disamb_items:
        baseline = compute_disamb_keyword_baseline(
            disamb_items,
            ci=args.ci,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
        )
        out.update({f"baseline_disamb_{k}": v for k, v in baseline.items()})

    if args.run_patching and disamb_items:
        t_patch0 = time.perf_counter()
        cpt_res = compute_cpt_context_swap_patching(
            model,
            tokenizer,
            disamb_items,
            tensor_device,
            layers=_parse_int_list(args.patch_layers),
            normalize_by_length=not args.no_length_norm,
            require_token_id_match=not args.patch_allow_token_id_mismatch,
            ci=args.ci,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
        )
        out.update({f"cpt_{k}": v for k, v in cpt_res.items()})
        out["eval_time_cpt_context_swap_sec"] = float(time.perf_counter() - t_patch0)

    if getattr(args, "run_sae_patching", False) and disamb_items:
        t_sae0 = time.perf_counter()
        if not str(getattr(args, "sae_repo", "")).strip():
            raise ValueError("--run_sae_patching requires --sae_repo")
        sae_layers = _parse_int_list(str(getattr(args, "sae_layers", "") or ""))
        if sae_layers is None:
            # Defaulting to "all layers" would trigger large downloads; require explicit selection.
            raise ValueError("--run_sae_patching requires --sae_layers (comma-separated)")

        from aom.interventions.sae_adapter import SAEInputTransform, SAEPatchConfig
        from aom.interventions.sae_loader import load_gemma_scope_sae

        sae_by_layer = {}
        transform_by_layer = {}
        for l in sae_layers:
            sae, _meta = load_gemma_scope_sae(
                str(getattr(args, "sae_repo")),
                layer=int(l),
                width=str(getattr(args, "sae_width")),
                run_name=getattr(args, "sae_run_name", None),
                l0_target=getattr(args, "sae_l0_target", None),
                device=str(tensor_device),
                dtype=str(getattr(args, "sae_dtype", "float32")),
                revision=getattr(args, "sae_revision", None),
                local_files_only=bool(getattr(args, "local_files_only", False)),
            )
            sae_by_layer[int(l)] = sae
            transform_by_layer[int(l)] = SAEInputTransform(scale=float(getattr(args, "sae_scale", 1.0)))

        sae_cfg = SAEPatchConfig(
            decode_strategy=str(getattr(args, "sae_decode_strategy", "delta_1decode")),
            eps_active=float(getattr(args, "sae_eps_active", 1e-6)),
            dtype_policy=str(getattr(args, "sae_dtype_policy", "sae")),
        )
        sae_cpt_res = compute_sae_cpt_context_swap_patching(
            model,
            tokenizer,
            disamb_items,
            tensor_device,
            sae_by_layer=sae_by_layer,
            transform_by_layer=transform_by_layer,
            layers=list(sae_layers),
            config=sae_cfg,
            normalize_by_length=not args.no_length_norm,
            require_token_id_match=not args.patch_allow_token_id_mismatch,
            ci=args.ci,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
        )
        out.update({f"sae_cpt_{k}": v for k, v in sae_cpt_res.items()})
        out["sae_repo"] = str(getattr(args, "sae_repo"))
        out["sae_width"] = str(getattr(args, "sae_width"))
        out["sae_layers"] = ",".join(str(int(x)) for x in sae_layers)
        out["sae_run_name"] = str(getattr(args, "sae_run_name", ""))
        out["sae_l0_target"] = str(getattr(args, "sae_l0_target", ""))
        out["sae_scale"] = float(getattr(args, "sae_scale", 1.0))
        out["sae_decode_strategy"] = str(getattr(args, "sae_decode_strategy", ""))
        out["eval_time_sae_cpt_sec"] = float(time.perf_counter() - t_sae0)

    if getattr(args, "run_clt_patching", False) and disamb_items:
        t_clt0 = time.perf_counter()
        if not str(getattr(args, "clt_repo", "")).strip():
            raise ValueError("--run_clt_patching requires --clt_repo")
        clt_layers = _parse_int_list(str(getattr(args, "clt_layers", "") or ""))
        if clt_layers is None:
            # Defaulting to "all layers" would trigger large downloads; require explicit selection.
            raise ValueError("--run_clt_patching requires --clt_layers (comma-separated)")

        from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
        from aom.interventions.clt_loader import load_clt

        clt_by_layer = {}
        transform_by_layer = {}
        meta_by_layer = {}
        for l in clt_layers:
            clt, meta = load_clt(
                str(getattr(args, "clt_repo")),
                layer=int(l),
                width=str(getattr(args, "clt_width")),
                run_name=getattr(args, "clt_run_name", None),
                l0_target=getattr(args, "clt_l0_target", None),
                device=str(tensor_device),
                dtype=str(getattr(args, "clt_dtype", "float32")),
                local_files_only=bool(getattr(args, "local_files_only", False)),
            )
            clt_by_layer[int(l)] = clt
            transform_by_layer[int(l)] = CLTInputTransform(scale=float(getattr(args, "clt_scale", 1.0)))
            meta_by_layer[int(l)] = meta

        clt_cfg = CLTPatchConfig(
            decode_strategy=str(getattr(args, "clt_decode_strategy", "delta_1decode")),
            eps_active=float(getattr(args, "clt_eps_active", 1e-6)),
            dtype_policy=str(getattr(args, "clt_dtype_policy", "clt")),
        )
        clt_cpt_res = compute_clt_cpt_context_swap_patching(
            model,
            tokenizer,
            disamb_items,
            tensor_device,
            clt_by_layer=clt_by_layer,
            transform_by_layer=transform_by_layer,
            layers=list(clt_layers),
            config=clt_cfg,
            normalize_by_length=not args.no_length_norm,
            require_token_id_match=not args.patch_allow_token_id_mismatch,
            ci=args.ci,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
        )
        out.update({f"clt_cpt_{k}": v for k, v in clt_cpt_res.items()})
        out["clt_repo"] = str(getattr(args, "clt_repo"))
        out["clt_width"] = str(getattr(args, "clt_width"))
        out["clt_layers"] = ",".join(str(int(x)) for x in clt_layers)
        out["clt_run_name"] = str(getattr(args, "clt_run_name", ""))
        out["clt_l0_target"] = str(getattr(args, "clt_l0_target", ""))
        out["clt_scale"] = float(getattr(args, "clt_scale", 1.0))
        out["clt_decode_strategy"] = str(getattr(args, "clt_decode_strategy", ""))
        out["clt_dtype_policy"] = str(getattr(args, "clt_dtype_policy", "clt"))
        out["clt_eps_active"] = float(getattr(args, "clt_eps_active", 1e-6))
        out["clt_site_mode"] = "|".join(sorted({str(meta_by_layer[int(l)].site_mode) for l in clt_layers}))
        out["clt_encode_site"] = "|".join(sorted({str(meta_by_layer[int(l)].encode_site) for l in clt_layers}))
        out["clt_decode_site"] = "|".join(sorted({str(meta_by_layer[int(l)].decode_site) for l in clt_layers}))
        out["clt_writeback_site"] = "|".join(sorted({str(meta_by_layer[int(l)].writeback_site) for l in clt_layers}))
        params_hash_pairs: list[str] = []
        cfg_hash_pairs: list[str] = []
        for l in sorted(int(x) for x in clt_layers):
            meta = meta_by_layer[int(l)]
            params_path = str(getattr(meta, "params_path", "") or "")
            cfg_path = str(getattr(meta, "cfg_path", "") or "")
            params_sha = _sha256_path(params_path)
            cfg_sha = _sha256_path(cfg_path)
            out[f"clt_params_path_layer_{int(l)}"] = params_path
            out[f"clt_params_sha256_layer_{int(l)}"] = params_sha
            out[f"clt_cfg_path_layer_{int(l)}"] = cfg_path
            out[f"clt_cfg_sha256_layer_{int(l)}"] = cfg_sha
            params_hash_pairs.append(f"{int(l)}:{params_sha}")
            cfg_hash_pairs.append(f"{int(l)}:{cfg_sha}")
        out["clt_params_sha256_layers"] = "|".join(params_hash_pairs)
        out["clt_cfg_sha256_layers"] = "|".join(cfg_hash_pairs)
        out["eval_time_clt_cpt_sec"] = float(time.perf_counter() - t_clt0)

    if getattr(args, "run_patching_specificity", False) and disamb_items:
        t_spec0 = time.perf_counter()
        spec_res = compute_cpt_target_specificity_control(
            model,
            tokenizer,
            disamb_items,
            tensor_device,
            layer=getattr(args, "patch_specificity_layer", None),
            depth_frac=float(getattr(args, "patch_specificity_depth_frac", 0.25)),
            buffer=int(getattr(args, "patch_specificity_buffer", 2)),
            donor_buffer=getattr(args, "patch_specificity_donor_buffer", None),
            position_window=int(getattr(args, "patch_specificity_position_window", 8)),
            selection_seed=int(getattr(args, "patch_specificity_seed", 0)),
            normalize_by_length=not args.no_length_norm,
            require_token_id_match=not args.patch_allow_token_id_mismatch,
            ci=args.ci,
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
        )
        out.update({f"cpt_spec_{k}": v for k, v in spec_res.items()})
        out["eval_time_cpt_specificity_sec"] = float(time.perf_counter() - t_spec0)

    eval_ended_at_utc = datetime.now(timezone.utc).isoformat()
    out["eval_started_at_utc"] = str(eval_started_at_utc)
    out["eval_ended_at_utc"] = str(eval_ended_at_utc)
    out["eval_wall_time_sec"] = float(time.perf_counter() - eval_t0)
    out["eval_time_disamb_sec"] = float(eval_time_disamb_sec)
    out["eval_time_cf_sec"] = float(eval_time_cf_sec)
    out["eval_time_coh_sec"] = float(eval_time_coh_sec)

    return out


def run_once(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    from aom.models.loader import load_causal_lm
    from aom.utils import configure_scoring_performance, get_best_device, set_seed

    t0 = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    _enforce_device_map_patching_guard(args)
    set_seed(seed)
    configure_scoring_performance(
        score_batch_size=int(getattr(args, "score_batch_size", 1)),
        use_prefix_cache=bool(getattr(args, "use_prefix_cache", False)),
    )
    if args.device == "auto":
        device = get_best_device()
    else:
        device = {"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device]
        import torch

        device = torch.device(device)

    print(
        "Loading model "
        f"model={args.model_name_or_path!r} "
        f"local_files_only={bool(args.local_files_only)} "
        f"torch_dtype={args.torch_dtype!r} "
        f"attn_implementation={args.attn_implementation!r} "
        f"device_map={args.device_map!r}",
        flush=True,
    )
    loaded = load_causal_lm(
        args.model_name_or_path,
        device=device,
        torch_dtype=args.torch_dtype,
        revision=getattr(args, "revision", None),
        tokenizer_revision=getattr(args, "tokenizer_revision", None),
        local_files_only=args.local_files_only,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    print(f"Loaded {args.model_name_or_path} (arch={loaded.architecture})")

    # If we use `device_map`, let Accelerate/Transformers handle device placement.
    # In that case keep input tensors on CPU (common HF pattern).
    import torch

    tensor_device = _infer_input_device(model) if args.device_map is not None else device

    if args.disamb_path:
        disamb_items, disamb_manifest = load_disamb_pairs_with_manifest(
            str(args.disamb_path),
            role="disamb",
            error_policy="warn_skip",
        )
        if int(disamb_manifest.n_rows_invalid) > 0:
            raise ValueError(
                f"Dataset disamb has {int(disamb_manifest.n_rows_invalid)}/{int(disamb_manifest.n_rows_total)} invalid rows"
            )
    else:
        disamb_items = []

    if args.cf_path:
        cf_items, cf_manifest = load_counterfactual_pairs_with_manifest(
            str(args.cf_path),
            role="cf",
            error_policy="warn_skip",
        )
        if int(cf_manifest.n_rows_invalid) > 0:
            raise ValueError(
                f"Dataset cf has {int(cf_manifest.n_rows_invalid)}/{int(cf_manifest.n_rows_total)} invalid rows"
            )
    else:
        cf_items = []

    if args.coh_path:
        coh_items, coh_manifest = load_coherence_items_with_manifest(
            str(args.coh_path),
            role="coh",
            error_policy="warn_skip",
        )
        if int(coh_manifest.n_rows_invalid) > 0:
            raise ValueError(
                f"Dataset coh has {int(coh_manifest.n_rows_invalid)}/{int(coh_manifest.n_rows_total)} invalid rows"
            )
    else:
        coh_items = []

    disamb_items, cf_items, coh_items, system_prompt_sha256, chat_template_sha256 = _prepare_datasets_for_eval(
        args=args,
        tokenizer=tokenizer,
        disamb_items_raw=disamb_items,
        cf_items_raw=cf_items,
        coh_items_raw=coh_items,
    )
    print(
        f"Loaded datasets disamb={len(disamb_items)} cf={len(cf_items)} coh={len(coh_items)}",
        flush=True,
    )

    repo_root = Path(__file__).resolve().parent
    git_commit_hash = get_git_commit_hash(
        repo_root=repo_root,
        required=bool(getattr(args, "require_git", True)),
    )
    argv_redacted_list = redact_argv(sys.argv)
    argv_redacted_json = json.dumps(argv_redacted_list, ensure_ascii=False)
    argv_sha256 = hashlib.sha256(argv_redacted_json.encode("utf-8")).hexdigest()

    dataset_bundle_info: dict[str, str] | None = None
    dataset_manifest_path_raw = str(getattr(args, "dataset_manifest_path", "") or "").strip()
    if dataset_manifest_path_raw:
        dataset_bundle_info = validate_bundle_manifest(
            dataset_manifest_path_raw,
            disamb_path=str(getattr(args, "disamb_path", "") or ""),
            cf_path=str(getattr(args, "cf_path", "") or ""),
            coh_path=str(getattr(args, "coh_path", "") or ""),
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

    out = run_eval_with_loaded(
        model_name=args.model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        architecture=loaded.architecture,
        hf_model_commit_hash=loaded.model_commit_hash,
        hf_tokenizer_revision_effective=loaded.tokenizer_revision_effective,
        system_prompt_sha256=system_prompt_sha256,
        chat_template_sha256=chat_template_sha256,
        args=args,
        seed=seed,
        git_commit_hash=git_commit_hash,
        argv_redacted_json=argv_redacted_json,
        argv_sha256=argv_sha256,
        tensor_device=tensor_device,
        disamb_items=disamb_items,
        cf_items=cf_items,
        coh_items=coh_items,
    )
    out["started_at_utc"] = str(started_at_utc)
    out["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["wall_time_sec"] = float(time.perf_counter() - t0)
    return out


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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Optional YAML/JSON config file. Values set here become argparse defaults; explicit CLI flags still win. "
            "Relative paths inside configs are resolved relative to the repo root (directory containing aom_eval.py)."
        ),
    )
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--models", nargs="*", type=str, default=None, help="Optional list of models to sweep.")
    p.add_argument("--revision", type=str, default=None, help="Optional HF model revision (branch/tag/commit SHA).")
    p.add_argument(
        "--tokenizer_revision",
        type=str,
        default=None,
        help="Optional HF tokenizer revision (defaults to --revision when unset).",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading models that require custom code from the Hugging Face repo (use with care).",
    )
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
    p.add_argument(
        "--logprobs_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16", "float64"],
        help="Dtype for log-softmax / logprob scoring (default: float32 for stability).",
    )
    p.add_argument(
        "--score_batch_size",
        type=int,
        default=8,
        help="Batch size for continuation scoring (default: 8).",
    )
    p.add_argument(
        "--use_prefix_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse prompt KV cache across continuations when supported (default: True).",
    )
    p.add_argument(
        "--strict_finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast on NaN/Inf during logprob scoring (recommended for paper runs).",
    )
    p.add_argument(
        "--strict_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast if any primary metric is missing/invalid (default: False).",
    )
    p.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU (e.g., 'auto').")
    p.add_argument("--local_files_only", action="store_true", help="Disallow downloads (offline mode).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep_seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
        help="Determinism mode: strict/best_effort/off (default: best_effort).",
    )
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

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--cf_path", type=str, default=str(root / "data" / "counterfactual.jsonl"))
    p.add_argument("--coh_path", type=str, default=str(root / "data" / "coherence.jsonl"))

    p.add_argument(
        "--prompt_input",
        type=str,
        default="full_prompt",
        choices=["full_prompt", "user_message"],
        help="Interpret dataset prompt fields as full prompt strings or as user messages (default: full_prompt).",
    )
    p.add_argument(
        "--prompt_mode",
        type=str,
        default="raw",
        choices=["raw", "chat_template"],
        help="Prompt rendering mode: raw (identity) or chat_template (render via tokenizer.apply_chat_template).",
    )
    p.add_argument("--system_prompt_file", type=str, default="", help="System prompt file path (chat_template mode).")
    p.add_argument("--system_prompt", type=str, default="", help="System prompt text (chat_template mode; redacted in manifest).")
    p.add_argument(
        "--add_generation_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to add the model-specific generation prompt when rendering chat templates (default: True).",
    )

    p.add_argument(
        "--boundary_check",
        type=str,
        default="warn",
        choices=["off", "warn", "error"],
        help="Prompt boundary policy: warn/error if a prompt doesn't end with whitespace/newline (default: warn).",
    )
    p.add_argument(
        "--normalize_boundaries",
        action="store_true",
        help="Append a single space to prompts that don't end with whitespace (opt-in).",
    )

    p.add_argument("--run_patching", action="store_true", help="Run CPT-style context-swap activation patching.")
    p.add_argument("--patch_layers", type=str, default="", help="Comma-separated layers to patch (default: all).")
    p.add_argument(
        "--patch_allow_token_id_mismatch",
        action="store_true",
        help="Allow patching even if target span token IDs differ (not recommended).",
    )

    p.add_argument(
        "--run_sae_patching",
        action="store_true",
        help="Run SAE feature-space CPT-style patching (requires Gemma Scope SAEs).",
    )
    p.add_argument("--sae_repo", type=str, default="", help="HF repo id or local path for a Gemma Scope SAE bundle.")
    p.add_argument("--sae_revision", type=str, default=None, help="Optional HF revision for the SAE repo.")
    p.add_argument("--sae_width", type=str, default="16k", help="SAE width (matches directory width_*).")
    p.add_argument("--sae_run_name", type=str, default=None, help="Explicit run subdir name (e.g. 'average_l0_71').")
    p.add_argument("--sae_l0_target", type=int, default=None, help="Select run dir by average_l0_* tag.")
    p.add_argument(
        "--sae_layers",
        type=str,
        default="",
        help="Comma-separated SAE layers to run (required; avoids implicit 'all layers' downloads).",
    )
    p.add_argument("--sae_scale", type=float, default=1.0, help="SAE input scale factor (transform.forward scale).")
    p.add_argument(
        "--sae_dtype",
        type=str,
        default="auto",
        help="Dtype for SAE weights/compute (e.g. auto/float16/bfloat16/float32).",
    )
    p.add_argument(
        "--sae_decode_strategy",
        type=str,
        default="delta_1decode",
        choices=["safe_2decode", "delta_1decode"],
        help="Decode strategy for SAE delta injection.",
    )
    p.add_argument(
        "--sae_dtype_policy",
        type=str,
        default="sae",
        choices=["sae", "model"],
        help="Dtype policy inside SAE hook.",
    )
    p.add_argument("--sae_eps_active", type=float, default=1e-6, help="Threshold for 'active feature' counting.")

    p.add_argument(
        "--run_clt_patching",
        action="store_true",
        help="Run CLT latent-space CPT-style patching.",
    )
    p.add_argument("--clt_repo", type=str, default="", help="HF repo id or local path for a CLT bundle.")
    p.add_argument("--clt_width", type=str, default="16k", help="CLT width (matches directory width_*).")
    p.add_argument("--clt_run_name", type=str, default=None, help="Explicit CLT run subdir name (e.g. 'average_l0_71').")
    p.add_argument("--clt_l0_target", type=int, default=None, help="Select CLT run dir by average_l0_* tag.")
    p.add_argument(
        "--clt_layers",
        type=str,
        default="",
        help="Comma-separated CLT layers to run (required; avoids implicit 'all layers' downloads).",
    )
    p.add_argument("--clt_scale", type=float, default=1.0, help="CLT input scale factor (transform.forward scale).")
    p.add_argument("--clt_dtype", type=str, default="float32", help="Dtype for CLT weights/compute (e.g. float32).")
    p.add_argument(
        "--clt_decode_strategy",
        type=str,
        default="delta_1decode",
        choices=["safe_2decode", "delta_1decode"],
        help="Decode strategy for CLT delta injection.",
    )
    p.add_argument(
        "--clt_dtype_policy",
        type=str,
        default="clt",
        choices=["clt", "model"],
        help="Dtype policy inside CLT hook.",
    )
    p.add_argument("--clt_eps_active", type=float, default=1e-6, help="Threshold for 'active latent' counting.")

    p.add_argument(
        "--run_patching_specificity",
        action="store_true",
        help="Run a target-specificity control: compare target-span patch vs matched-token off-target patch at a fixed layer.",
    )
    p.add_argument(
        "--patch_specificity_layer",
        type=int,
        default=None,
        help="Fixed layer index for specificity control (default: derived from --patch_specificity_depth_frac).",
    )
    p.add_argument(
        "--patch_specificity_depth_frac",
        type=float,
        default=0.25,
        help="If --patch_specificity_layer is unset, use round(depth_frac*(L-1)) to pick a fixed layer.",
    )
    p.add_argument(
        "--patch_specificity_buffer",
        type=int,
        default=2,
        help="Exclude receiver control positions within +/- buffer of the target span (default: 2).",
    )
    p.add_argument(
        "--patch_specificity_donor_buffer",
        type=int,
        default=None,
        help="Exclude donor control positions within +/- donor_buffer of the target span (default: same as buffer).",
    )
    p.add_argument(
        "--patch_specificity_seed",
        type=int,
        default=0,
        help="Seed for deterministic off-target control pair selection (not optimized for effect size).",
    )
    p.add_argument(
        "--patch_specificity_position_window",
        type=int,
        default=8,
        help="Position-match control receiver indices to the target span by sampling within an adaptive +/- window (default: 8).",
    )
    p.add_argument("--no_length_norm", action="store_true", help="Use sum logprob instead of mean logprob.")
    p.add_argument("--bootstrap_n", type=int, default=1000, help="Bootstrap replicates for metric CIs.")
    p.add_argument("--bootstrap_seed", type=int, default=42, help="RNG seed for bootstrap CIs.")
    p.add_argument("--ci", type=float, default=0.95, help="Confidence level for bootstrap CIs.")
    p.add_argument(
        "--composite_missing_policy",
        type=str,
        default="nan",
        choices=["fail", "nan", "ignore"],
        help="Composite policy when any primary metric is missing/invalid (default: nan).",
    )
    p.add_argument("--csv_path", type=str, default="")
    p.add_argument("--results_dir", type=str, default="results", help="Directory for results artifacts (CSV + manifest).")
    p.add_argument("--run_name", type=str, default="", help="Base name for outputs when --csv_path is unset.")
    p.add_argument(
        "--dataset_manifest_path",
        type=str,
        default="",
        help="Optional path to a dataset manifest to record in the run manifest.",
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
    p.add_argument(
        "--require_git",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a non-empty git commit hash for provenance (default: True). Disable with --no-require_git.",
    )

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


def main() -> None:
    run_t0 = time.perf_counter()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
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
                    ProtocolArgBinding("strict_finite", ("repro", "strict_finite"), _coerce_bool),
                ],
                context="aom_eval",
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
    attempt_unit = "model_seed"
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

    def _primary_metrics_invalid_reason(row: Dict[str, Any]) -> str | None:
        specs = [
            ("disamb_accuracy", "disamb_accuracy_valid", "disamb_accuracy_reason"),
            ("cf_shift_direction_accuracy", "cf_shift_direction_accuracy_valid", "cf_shift_direction_accuracy_reason"),
            ("coh_constraint_accuracy", "coh_constraint_accuracy_valid", "coh_constraint_accuracy_reason"),
        ]
        bad: list[str] = []
        for value_key, valid_key, reason_key in specs:
            valid = row.get(valid_key, None)
            val = row.get(value_key, None)
            ok = (
                isinstance(valid, bool)
                and valid
                and isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(float(val))
            )
            if ok:
                continue
            reason = row.get(reason_key, None)
            if isinstance(reason, str) and reason.strip():
                bad.append(f"{value_key} ({reason.strip()})")
            else:
                bad.append(str(value_key))
        if not bad:
            return None
        return "invalid primary metrics: " + ", ".join(bad)

    csv_path = str(getattr(args, "csv_path", "") or "").strip()
    if not csv_path:
        results_dir = Path(str(getattr(args, "results_dir", "results")))
        run_name = str(getattr(args, "run_name", "") or "").strip() or "aom_eval"
        csv_path = str(results_dir / f"{run_name}.csv")
    csv_p = Path(csv_path)
    manifest_path = str(csv_p.with_suffix(".manifest.json"))

    try:
        from aom.utils import configure_logprob_computation, configure_scoring_performance

        import torch

        configure_logprob_computation(
            logprobs_dtype=getattr(torch, str(args.logprobs_dtype)),
            strict_finite=bool(args.strict_finite),
        )
        configure_scoring_performance(
            score_batch_size=int(getattr(args, "score_batch_size", 1)),
            use_prefix_cache=bool(getattr(args, "use_prefix_cache", False)),
        )
        _enforce_device_map_patching_guard(args)

        models = args.models if args.models is not None and len(args.models) > 0 else [args.model_name_or_path]
        seeds = args.sweep_seeds if args.sweep_seeds is not None and len(args.sweep_seeds) > 0 else [args.seed]
        n_models = int(len(models))
        n_seeds = int(len(seeds))
        attempted_expected = int(n_models * n_seeds)

        datasets = {}
        if args.disamb_path:
            try:
                disamb_items_raw, disamb_manifest = load_disamb_pairs_with_manifest(
                    str(args.disamb_path),
                    role="disamb",
                    error_policy=data_error_policy,
                )
                datasets["disamb"] = disamb_manifest.as_dict()
            except DatasetLoadError as e:
                datasets["disamb"] = e.manifest.as_dict()
                raise
        else:
            disamb_items_raw = []
        if args.cf_path:
            try:
                cf_items_raw, cf_manifest = load_counterfactual_pairs_with_manifest(
                    str(args.cf_path),
                    role="cf",
                    error_policy=data_error_policy,
                )
                datasets["cf"] = cf_manifest.as_dict()
            except DatasetLoadError as e:
                datasets["cf"] = e.manifest.as_dict()
                raise
        else:
            cf_items_raw = []
        if args.coh_path:
            try:
                coh_items_raw, coh_manifest = load_coherence_items_with_manifest(
                    str(args.coh_path),
                    role="coh",
                    error_policy=data_error_policy,
                )
                datasets["coh"] = coh_manifest.as_dict()
            except DatasetLoadError as e:
                datasets["coh"] = e.manifest.as_dict()
                raise
        else:
            coh_items_raw = []

        for role, dm in datasets.items():
            invalid = int(dm.get("n_rows_invalid", 0) or 0)
            total = int(dm.get("n_rows_total", 0) or 0)
            valid = int(dm.get("n_rows_valid", 0) or 0)
            if total > 0 and valid == 0:
                raise ValueError(f"Dataset {role} has {invalid}/{total} invalid rows and 0 valid rows")
            if invalid > 0:
                msg = f"dataset {role}: invalid_rows {invalid} of {total}"
                dataset_warnings.append(msg)
                _warn(f"[WARN] {msg}")

        print(f"Datasets disamb={len(disamb_items_raw)} cf={len(cf_items_raw)} coh={len(coh_items_raw)}", flush=True)

        repo_root = Path(__file__).resolve().parent
        git_commit_hash = get_git_commit_hash(
            repo_root=repo_root,
            required=bool(getattr(args, "require_git", True)),
        )
        argv_redacted_list = redact_argv(sys.argv)
        argv_redacted_json = json.dumps(argv_redacted_list, ensure_ascii=False)
        argv_sha256 = hashlib.sha256(argv_redacted_json.encode("utf-8")).hexdigest()

        dataset_bundle_info: dict[str, str] | None = None
        dataset_manifest_path_raw = str(getattr(args, "dataset_manifest_path", "") or "").strip()
        if dataset_manifest_path_raw:
            dataset_bundle_info = validate_bundle_manifest(
                dataset_manifest_path_raw,
                disamb_path=str(getattr(args, "disamb_path", "") or ""),
                cf_path=str(getattr(args, "cf_path", "") or ""),
                coh_path=str(getattr(args, "coh_path", "") or ""),
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
        if determinism == "strict" and not bool(repro.get("determinism_enforced", False)):
            raise ValueError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")

        import gc

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

            disamb_items, cf_items, coh_items, system_prompt_sha256, chat_template_sha256 = _prepare_datasets_for_eval(
                args=args,
                tokenizer=tokenizer,
                disamb_items_raw=disamb_items_raw,
                cf_items_raw=cf_items_raw,
                coh_items_raw=coh_items_raw,
            )

            for s in seeds:
                try:
                    r = run_eval_with_loaded(
                        model_name=model_name,
                        model=model,
                        tokenizer=tokenizer,
                        architecture=loaded.architecture,
                        hf_model_commit_hash=loaded.model_commit_hash,
                        hf_tokenizer_revision_effective=loaded.tokenizer_revision_effective,
                        system_prompt_sha256=system_prompt_sha256,
                        chat_template_sha256=chat_template_sha256,
                        args=args,
                        seed=int(s),
                        git_commit_hash=git_commit_hash,
                        argv_redacted_json=argv_redacted_json,
                        argv_sha256=argv_sha256,
                        tensor_device=tensor_device,
                        disamb_items=disamb_items,
                        cf_items=cf_items,
                        coh_items=coh_items,
                    )
                except Exception as e:
                    summary.record_failure(e)
                    _warn(f"[fail] model={model_name!r} seed={int(s)}: {type(e).__name__}: {e}")
                    if error_policy == "raise":
                        raise
                    continue

                invalid_reason = _primary_metrics_invalid_reason(r)
                if invalid_reason is not None:
                    summary.record_invalid(invalid_reason)
                    _warn(f"[WARN] model={model_name!r} seed={int(s)}: {invalid_reason}")

                rows.append(r)
                summary.record_success()
                print(
                    f"model={r.get('model','')} seed={r['seed']} aom={r['aom_composite']:.3f} "
                    f"disamb_acc={r.get('disamb_accuracy', 0.0):.3f} "
                    f"cf_dir={r.get('cf_shift_direction_accuracy', 0.0):.3f} "
                    f"cf_label_acc={r.get('cf_label_accuracy_shift_items', 0.0):.3f} "
                    f"coh_acc={r.get('coh_constraint_accuracy', 0.0):.3f}",
                    flush=True,
                )

            # Free model before next one (important for large Qwen variants).
            del model
            del tokenizer
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass

    except Exception as e:
        fatal_error = e
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[FATAL] Aborting; run manifest will be written to {manifest_path}", file=sys.stderr, flush=True)

    run_ended_at_utc = datetime.now(timezone.utc).isoformat()
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
    manifest["attempt_unit"] = str(attempt_unit)
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
