from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from aom.provenance.protocol import resolve_protocol_provenance


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _u32_seed(*parts: str, base_seed: int) -> int:
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], byteorder="little", signed=False)


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_int_list(raw: str) -> List[int]:
    s = str(raw or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _validate_sha256_hex(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError("SHA256 values must be 64 lowercase hex characters")
    return s


def _item_event_check(
    *,
    model: Any,
    prompt: str,
    token_text: str,
    token_id: int,
) -> Dict[str, Any]:
    tok_prompt = _to_tokens_no_bos(model, str(prompt))[0].tolist()
    tok_text = _to_tokens_no_bos(model, str(token_text))[0].tolist()
    tok_prompt_plus = _to_tokens_no_bos(model, str(prompt) + str(token_text))[0].tolist()

    reasons: List[str] = []
    if len(tok_text) != 1:
        reasons.append(f"token_text_not_single_token(len={len(tok_text)})")
    if len(tok_text) >= 1 and int(tok_text[0]) != int(token_id):
        reasons.append(f"token_id_mismatch(tokenizer={int(tok_text[0])},manifest={int(token_id)})")
    if len(tok_prompt_plus) < len(tok_prompt):
        reasons.append("prompt_plus_shorter_than_prompt")
    else:
        if tok_prompt_plus[: len(tok_prompt)] != tok_prompt:
            reasons.append("prefix_instability(prompt_prefix_changed)")
        if len(tok_prompt_plus) != len(tok_prompt) + 1:
            reasons.append(f"prompt_plus_len_delta_not_one(delta={len(tok_prompt_plus) - len(tok_prompt)})")
        elif int(tok_prompt_plus[len(tok_prompt)]) != int(token_id):
            reasons.append(
                f"next_token_id_mismatch(next={int(tok_prompt_plus[len(tok_prompt)])},manifest={int(token_id)})"
            )
    return {
        "valid": len(reasons) == 0,
        "reason": "" if not reasons else ";".join(reasons),
        "prompt_len": int(len(tok_prompt)),
        "token_text_len": int(len(tok_text)),
        "prompt_plus_len": int(len(tok_prompt_plus)),
    }


def _load_tlens() -> tuple[Any, Any]:
    try:
        import transformer_lens  # type: ignore[import-not-found]
        from transformer_lens import HookedTransformer  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "TransformerLens not available. Install with `pip install transformer-lens` and retry."
        ) from e
    return transformer_lens, HookedTransformer


def _load_hooked_transformer(
    hooked_transformer_cls: Any,
    *,
    model_name_or_path: str,
    device: Any,
    torch_dtype: str | None,
    revision: str | None,
    tokenizer_revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    import torch

    sig = inspect.signature(hooked_transformer_cls.from_pretrained)
    kwargs: Dict[str, Any] = {}
    if "device" in sig.parameters:
        kwargs["device"] = str(device)
    if "default_prepend_bos" in sig.parameters:
        kwargs["default_prepend_bos"] = False

    hf_model_kwargs: Dict[str, Any] = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    hf_tokenizer_kwargs: Dict[str, Any] = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "use_fast": True,
    }
    if str(revision or "").strip():
        hf_model_kwargs["revision"] = str(revision).strip()
    tok_rev = str(tokenizer_revision or "").strip() or str(revision or "").strip()
    if tok_rev:
        hf_tokenizer_kwargs["revision"] = str(tok_rev)
    if torch_dtype is not None:
        dtype = getattr(torch, str(torch_dtype))
        hf_model_kwargs["torch_dtype"] = dtype
    if "hf_model_kwargs" in sig.parameters:
        kwargs["hf_model_kwargs"] = hf_model_kwargs
    elif "model_kwargs" in sig.parameters:
        kwargs["model_kwargs"] = hf_model_kwargs
    if "hf_tokenizer_kwargs" in sig.parameters:
        kwargs["hf_tokenizer_kwargs"] = hf_tokenizer_kwargs
    elif "tokenizer_kwargs" in sig.parameters:
        kwargs["tokenizer_kwargs"] = hf_tokenizer_kwargs

    try:
        model = hooked_transformer_cls.from_pretrained(str(model_name_or_path), **kwargs)
    except TypeError as e:
        raise RuntimeError(
            "Could not call HookedTransformer.from_pretrained with current TransformerLens version."
        ) from e
    model.eval()
    return model


def _resolve_hook_name(model: Any, *, layer: int, suffix: str) -> str:
    preferred = f"blocks.{int(layer)}.{suffix}"
    hook_dict = getattr(model, "hook_dict", None)
    if isinstance(hook_dict, Mapping) and preferred in hook_dict:
        return preferred
    if isinstance(hook_dict, Mapping):
        cands = [k for k in hook_dict.keys() if f"blocks.{int(layer)}." in k and str(k).endswith(str(suffix))]
        if len(cands) == 1:
            return str(cands[0])
        if len(cands) > 1:
            raise RuntimeError(f"Ambiguous hooks for layer={layer} suffix={suffix!r}: {sorted(cands)!r}")
    raise RuntimeError(f"Missing hook for layer={layer} suffix={suffix!r}")


def _to_tokens_no_bos(model: Any, text: str) -> Any:
    import torch

    to_tokens = getattr(model, "to_tokens", None)
    if not callable(to_tokens):
        raise RuntimeError("TransformerLens model missing to_tokens()")
    try:
        tokens = to_tokens(str(text), prepend_bos=False)
    except TypeError:
        tokens = to_tokens(str(text))
    if not isinstance(tokens, torch.Tensor):
        raise RuntimeError("to_tokens() did not return a tensor")
    if tokens.ndim != 2 or int(tokens.shape[0]) != 1:
        raise RuntimeError(f"Unexpected token shape from to_tokens: {tuple(tokens.shape)}")
    return tokens


def _load_manifest(path: Path) -> tuple[dict[str, Any], List[Dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        items = [dict(x) for x in obj if isinstance(x, Mapping)]
        return {"manifest_version": "legacy-list"}, items
    if isinstance(obj, Mapping):
        items_raw = obj.get("items", None)
        if not isinstance(items_raw, list):
            raise ValueError("Manifest object must contain an `items` list")
        items = [dict(x) for x in items_raw if isinstance(x, Mapping)]
        return dict(obj), items
    raise ValueError("Manifest must be a JSON object or list")


def _resolve_model_name(meta: Mapping[str, Any], items: Sequence[Mapping[str, Any]], override: str) -> str:
    if str(override or "").strip():
        return str(override).strip()
    names: set[str] = set()
    root_name = meta.get("model_name_or_path", None)
    if isinstance(root_name, str) and root_name.strip():
        names.add(root_name.strip())
    for it in items:
        n = it.get("model_name_or_path", None)
        if isinstance(n, str) and n.strip():
            names.add(n.strip())
    if len(names) == 1:
        return str(next(iter(names)))
    if not names:
        raise ValueError("Could not resolve model_name_or_path from manifest; pass --model_name_or_path")
    raise ValueError(f"Manifest contains multiple model names {sorted(names)!r}; pass --model_name_or_path")


def _parse_selected_heads(item: Mapping[str, Any], layer: int) -> List[int]:
    raw = item.get("selected_heads", None)
    if not isinstance(raw, Mapping):
        return []
    vals = raw.get(str(layer), None)
    if not isinstance(vals, list):
        vals = raw.get(int(layer), None)  # type: ignore[arg-type]
    if not isinstance(vals, list):
        return []
    out: List[int] = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"selected_heads[{layer!r}] must be list[int]")
        out.append(int(v))
    return sorted(set(out))


def _sample_random_heads(
    *,
    n_heads: int,
    k: int,
    excluded: Sequence[int],
    seed: int,
) -> List[int]:
    ex = {int(x) for x in excluded}
    pool = [h for h in range(int(n_heads)) if h not in ex]
    if k <= 0 or not pool:
        return []
    k_eff = min(int(k), len(pool))
    rng = random.Random(int(seed))
    return sorted(rng.sample(pool, k_eff))


def _delta_g_from_logits(logits: Any, *, target_pos: int, refusal_token_id: int, guidance_token_id: int) -> float:
    import torch

    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if target_pos < 0 or target_pos >= int(logits.shape[1]):
        raise ValueError(f"target_pos out of range: {target_pos} for seq_len={int(logits.shape[1])}")
    target = logits[0, int(target_pos)]
    return float((target[int(refusal_token_id)] - target[int(guidance_token_id)]).item())


def _head_contribs_for_layer(
    *,
    cache: Any,
    hook_name: str,
    target_pos: int,
    logit_dir: Any,
) -> Any:
    import torch

    head_out = cache[hook_name]
    if not isinstance(head_out, torch.Tensor):
        raise ValueError(f"cache[{hook_name!r}] is not a tensor")
    if head_out.ndim == 4:
        vecs = head_out[0, int(target_pos), :, :]
    elif head_out.ndim == 3:
        vecs = head_out[int(target_pos), :, :]
    else:
        raise ValueError(f"Unexpected head output shape for {hook_name}: {tuple(head_out.shape)}")
    if vecs.ndim != 2:
        raise ValueError(f"Unexpected vector shape for {hook_name}: {tuple(vecs.shape)}")
    dir_vec = logit_dir.to(device=vecs.device, dtype=vecs.dtype)
    contrib = vecs @ dir_vec
    return contrib.detach().float().cpu()


def _accumulate_head_means(
    *,
    cache: Any,
    hook_name: str,
    running_sum: Dict[int, Any],
    running_count: Dict[int, int],
    layer: int,
) -> None:
    import torch

    head_out = cache[hook_name]
    if not isinstance(head_out, torch.Tensor):
        raise ValueError(f"cache[{hook_name!r}] is not a tensor")
    if head_out.ndim == 4:
        vals = head_out[0]  # [pos, head, d_model]
    elif head_out.ndim == 3:
        vals = head_out  # [pos, head, d_model]
    else:
        raise ValueError(f"Unexpected head output shape for {hook_name}: {tuple(head_out.shape)}")
    if vals.ndim != 3:
        raise ValueError(f"Unexpected head output shape for {hook_name}: {tuple(vals.shape)}")
    summed = vals.sum(dim=0).detach().float().cpu()
    cnt = int(vals.shape[0])
    if int(layer) not in running_sum:
        running_sum[int(layer)] = summed
        running_count[int(layer)] = cnt
    else:
        running_sum[int(layer)] = running_sum[int(layer)] + summed
        running_count[int(layer)] = int(running_count[int(layer)]) + cnt


def _run_layer_head_ablation(
    *,
    model: Any,
    tokens: Any,
    hook_name: str,
    head_indices: Sequence[int],
    layer_mean: Any,
    target_pos: int,
    ablate_positions: str,
) -> Any:
    import torch

    idx = [int(h) for h in head_indices]
    if not idx:
        return model(tokens, return_type="logits")

    def _hook(act: torch.Tensor, _hook_obj) -> torch.Tensor:
        if act.ndim != 4:
            raise ValueError(f"Expected act rank-4 [batch,pos,head,d_model], got {tuple(act.shape)}")
        out = act.clone()
        mean = layer_mean.to(device=act.device, dtype=act.dtype)
        if int(mean.shape[0]) != int(act.shape[2]):
            raise ValueError(
                f"Mean-head shape mismatch for hook {hook_name}: mean_n_heads={int(mean.shape[0])} "
                f"act_n_heads={int(act.shape[2])}"
            )
        if ablate_positions == "all":
            out[:, :, idx, :] = mean[idx].unsqueeze(0).unsqueeze(0)
        elif ablate_positions == "target":
            out[:, int(target_pos), idx, :] = mean[idx].unsqueeze(0)
        else:
            raise ValueError(f"Unknown ablate_positions={ablate_positions!r}")
        return out

    return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=[(hook_name, _hook)])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="TransformerLens head-level analysis for MoM flagship manifests.")
    p.add_argument("--manifest_path", type=str, required=True, help="Manifest from scripts/export_mom_flagship_manifest.py")
    p.add_argument("--results_path", type=str, default=str(root / "results" / "mom_flagship_tlens" / "circuits.json"))
    p.add_argument("--model_name_or_path", type=str, default="", help="Override model from manifest.")
    p.add_argument("--revision", type=str, default="", help="Optional HF model revision (commit/tag/branch).")
    p.add_argument("--tokenizer_revision", type=str, default="", help="Optional HF tokenizer revision.")
    p.add_argument("--layers", type=str, default="", help="Optional comma-separated layer override.")
    p.add_argument(
        "--head_selection_source",
        type=str,
        default="manifest",
        choices=["manifest", "rank_abs_contrib"],
        help="Use prereg-selected heads from manifest or pick top-|C_l_h| heads per item.",
    )
    p.add_argument("--top_h", type=int, default=-1, help="Top-H heads per layer. -1 uses protocol/default.")
    p.add_argument("--ablate_positions", type=str, default="all", choices=["all", "target"])
    p.add_argument("--run_per_head_ablation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_items", type=int, default=0, help="0 means all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--determinism",
        type=str,
        default="best_effort",
        choices=["strict", "best_effort", "off"],
        help="Determinism policy for torch/random seeds.",
    )
    p.add_argument("--torch_dtype", type=str, default="", choices=["", "float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--fail_on_token_event_mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed when prompt+label tokenization does not preserve the next-token event.",
    )
    p.add_argument(
        "--allow_protocol_deviation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow exploratory overrides that differ from protocol defaults (recorded as deviations).",
    )
    p.add_argument("--protocol_path", type=str, default="", help="Optional frozen protocol path for hash verification.")
    p.add_argument("--protocol_sha256", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from aom.repro import ReproConfig, collect_versions, get_git_commit_hash, seed_everything
    from aom.run_manifest import build_run_manifest, write_run_manifest
    from aom.utils import bootstrap_ci

    t0 = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()

    protocol_prov = resolve_protocol_provenance(
        protocol_path_raw=str(getattr(args, "protocol_path", "") or ""),
        protocol_sha256_raw=str(getattr(args, "protocol_sha256", "") or ""),
        require_path_for_sha=False,
        require_frozen=True,
    )
    protocol_cfg = dict(protocol_prov.protocol_config)

    manifest_path = Path(str(args.manifest_path)).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"--manifest_path not found: {str(manifest_path)}")
    manifest_sha256 = _sha256_file(manifest_path)
    manifest_meta, items = _load_manifest(manifest_path)
    if int(getattr(args, "max_items", 0) or 0) > 0:
        items = items[: int(args.max_items)]
    if not items:
        raise ValueError("Manifest has no items to analyze")

    manifest_protocol_sha = _validate_sha256_hex(str(manifest_meta.get("protocol_sha256", "") or ""))
    cli_protocol_sha = str(protocol_prov.protocol_sha256 or "")
    if cli_protocol_sha and manifest_protocol_sha and cli_protocol_sha != manifest_protocol_sha:
        raise ValueError(
            f"Protocol hash mismatch: --protocol_sha256={cli_protocol_sha} manifest.protocol_sha256={manifest_protocol_sha}"
        )
    if cli_protocol_sha:
        effective_protocol_sha = str(cli_protocol_sha)
        protocol_sha_source = str(protocol_prov.protocol_sha256_source)
    elif manifest_protocol_sha:
        effective_protocol_sha = str(manifest_protocol_sha)
        protocol_sha_source = "copied_from_manifest"
    else:
        effective_protocol_sha = ""
        protocol_sha_source = ""

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))

    repro = seed_everything(ReproConfig(seed=int(args.seed), determinism=str(args.determinism)), device=device)
    if str(args.determinism) == "strict" and not bool(repro.get("determinism_enforced", False)):
        raise RuntimeError(f"Strict determinism requested but not enforced: {repro.get('determinism_reason')}")
    versions = collect_versions()
    transformers_version = str(versions.get("transformers", ""))

    _transformer_lens, HookedTransformer = _load_tlens()
    model_name = _resolve_model_name(manifest_meta, items, str(args.model_name_or_path))
    revision = str(getattr(args, "revision", "") or "").strip()
    tokenizer_revision = str(getattr(args, "tokenizer_revision", "") or "").strip()
    model = _load_hooked_transformer(
        HookedTransformer,
        model_name_or_path=str(model_name),
        device=device,
        torch_dtype=None if not str(args.torch_dtype or "").strip() else str(args.torch_dtype),
        revision=revision or None,
        tokenizer_revision=tokenizer_revision or None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )

    n_layers = int(getattr(getattr(model, "cfg", None), "n_layers", 0))
    if n_layers < 1:
        raise RuntimeError("Could not infer n_layers from TransformerLens model")

    layers_override = _parse_int_list(str(getattr(args, "layers", "") or ""))
    for l in layers_override:
        if l < 0 or l >= n_layers:
            raise ValueError(f"--layers includes out-of-range layer {l}; valid range [0,{n_layers-1}]")

    primary_h = int(getattr(args, "top_h", -1))
    if primary_h <= 0:
        multiplicity = protocol_cfg.get("multiplicity", {}) if isinstance(protocol_cfg, Mapping) else {}
        if isinstance(multiplicity, Mapping):
            primary_h = int(multiplicity.get("primary_head_set_size", 10) or 10)
        else:
            primary_h = 10
    if primary_h < 1:
        raise ValueError("Resolved top_h must be >= 1")

    protocol_deviations: List[str] = []
    if str(protocol_prov.protocol_path):
        multiplicity = protocol_cfg.get("multiplicity", {}) if isinstance(protocol_cfg, Mapping) else {}
        if isinstance(multiplicity, Mapping):
            expected_h = int(multiplicity.get("primary_head_set_size", primary_h) or primary_h)
            if int(primary_h) != int(expected_h):
                protocol_deviations.append(f"top_h={int(primary_h)} (protocol={int(expected_h)})")
        if str(args.head_selection_source) != "manifest":
            protocol_deviations.append(f"head_selection_source={str(args.head_selection_source)!r} (protocol='manifest')")
        if layers_override:
            protocol_deviations.append("layers_override_set")
        if protocol_deviations and not bool(getattr(args, "allow_protocol_deviation", False)):
            raise ValueError(
                "Protocol deviations detected; rerun with protocol-conformant settings or pass --allow_protocol_deviation: "
                + "; ".join(protocol_deviations)
            )

    # Pass 1: baseline + attribution contributions + mean head output tensors.
    per_item: List[Dict[str, Any]] = []
    head_sums: Dict[int, torch.Tensor] = {}
    head_counts: Dict[int, int] = {}

    for idx, item in enumerate(items):
        prompt = str(item.get("prompt_text", "") or "")
        if not prompt:
            raise ValueError(f"items[{idx}] missing prompt_text")
        target_pos = int(item.get("target_pos"))
        decision_token_pos = item.get("decision_token_pos", None)
        refusal_token_id = int(item.get("refusal_token_id"))
        guidance_token_id = int(item.get("guidance_token_id"))
        refusal_token_text = str(item.get("refusal_token_text", "") or "")
        guidance_token_text = str(item.get("guidance_token_text", "") or "")
        if not refusal_token_text:
            raise ValueError(f"items[{idx}] missing refusal_token_text")
        if not guidance_token_text:
            raise ValueError(f"items[{idx}] missing guidance_token_text")

        tokens = _to_tokens_no_bos(model, prompt)
        seq_len = int(tokens.shape[1])
        if isinstance(decision_token_pos, int) and int(decision_token_pos) != int(seq_len):
            raise ValueError(
                f"items[{idx}] tokenization mismatch: decision_token_pos={int(decision_token_pos)} "
                f"but TL prompt seq_len={seq_len}. Check BOS/chat-template/prompt boundary settings."
            )
        if target_pos < 0 or target_pos >= seq_len:
            raise ValueError(
                f"items[{idx}] target_pos out of range for tokenized prompt: target_pos={target_pos} seq_len={seq_len}"
            )
        if int(target_pos) != int(seq_len - 1):
            raise ValueError(
                f"items[{idx}] target_pos mismatch: target_pos={int(target_pos)} expected_last_index={int(seq_len - 1)}"
            )

        refusal_event = _item_event_check(
            model=model,
            prompt=str(prompt),
            token_text=str(refusal_token_text),
            token_id=int(refusal_token_id),
        )
        guidance_event = _item_event_check(
            model=model,
            prompt=str(prompt),
            token_text=str(guidance_token_text),
            token_id=int(guidance_token_id),
        )
        token_event_valid = bool(refusal_event["valid"]) and bool(guidance_event["valid"])
        if not token_event_valid:
            msg = (
                f"items[{idx}] token-event mismatch (refusal={refusal_event['reason']!r}, "
                f"guidance={guidance_event['reason']!r})"
            )
            if bool(getattr(args, "fail_on_token_event_mismatch", True)):
                raise ValueError(msg)
            continue

        logits, cache = model.run_with_cache(tokens, return_type="logits")
        baseline_delta_g = _delta_g_from_logits(
            logits,
            target_pos=int(target_pos),
            refusal_token_id=int(refusal_token_id),
            guidance_token_id=int(guidance_token_id),
        )

        w_u = getattr(model, "W_U", None)
        if not isinstance(w_u, torch.Tensor) or w_u.ndim != 2:
            raise RuntimeError("TransformerLens model missing expected W_U matrix")
        if int(refusal_token_id) < 0 or int(refusal_token_id) >= int(w_u.shape[1]):
            raise ValueError(f"refusal_token_id out of range: {refusal_token_id}")
        if int(guidance_token_id) < 0 or int(guidance_token_id) >= int(w_u.shape[1]):
            raise ValueError(f"guidance_token_id out of range: {guidance_token_id}")
        logit_dir = (w_u[:, int(refusal_token_id)] - w_u[:, int(guidance_token_id)]).detach()

        selected_layers = [int(x) for x in item.get("selected_layers", []) if isinstance(x, int)]
        if layers_override:
            selected_layers = list(layers_override)
        if not selected_layers:
            raise ValueError(
                f"items[{idx}] has no selected_layers; provide --layers or populate selected_layers in manifest"
            )

        layer_data: Dict[str, Any] = {}
        for layer in selected_layers:
            hook_name = _resolve_hook_name(model, layer=int(layer), suffix="attn.hook_result")
            contrib = _head_contribs_for_layer(
                cache=cache,
                hook_name=str(hook_name),
                target_pos=int(target_pos),
                logit_dir=logit_dir,
            )
            _accumulate_head_means(
                cache=cache,
                hook_name=str(hook_name),
                running_sum=head_sums,
                running_count=head_counts,
                layer=int(layer),
            )
            contrib_list = [float(x) for x in contrib.tolist()]
            layer_data[str(layer)] = {
                "hook_name": str(hook_name),
                "head_contributions": contrib_list,
                "n_heads": int(len(contrib_list)),
            }

        per_item.append(
            {
                "pair_id": str(item.get("pair_id", "")),
                "side": str(item.get("side", "")),
                "direction_id": str(item.get("direction_id", "")),
                "template_id": str(item.get("template_id", "")),
                "risk_domain": str(item.get("risk_domain", "")),
                "target_pos": int(target_pos),
                "refusal_token_id": int(refusal_token_id),
                "guidance_token_id": int(guidance_token_id),
                "selected_layers": [int(x) for x in selected_layers],
                "selected_heads_manifest": item.get("selected_heads", {}),
                "baseline_delta_g": float(baseline_delta_g),
                "token_event_valid": bool(token_event_valid),
                "token_event_refusal_valid": bool(refusal_event["valid"]),
                "token_event_refusal_reason": str(refusal_event["reason"]),
                "token_event_guidance_valid": bool(guidance_event["valid"]),
                "token_event_guidance_reason": str(guidance_event["reason"]),
                "layers": layer_data,
            }
        )

    if not per_item:
        raise ValueError("No valid items remained after token-event validation")

    head_means: Dict[int, torch.Tensor] = {}
    for layer, summed in head_sums.items():
        cnt = int(head_counts[layer])
        if cnt <= 0:
            raise RuntimeError(f"Internal error: non-positive head mean count for layer={layer}")
        head_means[int(layer)] = (summed / float(cnt)).detach().clone()

    # Pass 2: top-H + random-H ablations (+ optional per-head ablations).
    for item, manifest_item in zip(per_item, items):
        prompt = str(manifest_item.get("prompt_text", "") or "")
        tokens = _to_tokens_no_bos(model, prompt)
        target_pos = int(item["target_pos"])
        baseline = float(item["baseline_delta_g"])
        for layer in [int(x) for x in item["selected_layers"]]:
            layer_key = str(layer)
            layer_info = dict(item["layers"][layer_key])
            contrib = [float(x) for x in layer_info["head_contributions"]]
            n_heads = int(layer_info["n_heads"])
            if n_heads < 1:
                raise RuntimeError(f"Invalid n_heads for layer={layer}")

            if str(args.head_selection_source) == "manifest":
                top_heads = _parse_selected_heads(manifest_item, int(layer))
                if not top_heads:
                    raise ValueError(
                        f"Missing selected_heads for layer={layer} in item pair_id={item['pair_id']!r} side={item['side']!r}"
                    )
                top_heads = [h for h in top_heads if 0 <= h < n_heads]
                top_heads = sorted(set(top_heads))[: int(primary_h)]
            else:
                ranked = sorted(range(n_heads), key=lambda h: abs(float(contrib[h])), reverse=True)
                top_heads = ranked[: int(primary_h)]

            if not top_heads:
                raise ValueError(
                    f"Empty top-head set for layer={layer} pair_id={item['pair_id']!r} side={item['side']!r}"
                )

            rand_seed = _u32_seed(
                str(item.get("pair_id", "")),
                str(item.get("side", "")),
                str(layer),
                base_seed=int(args.seed),
            )
            random_heads = _sample_random_heads(
                n_heads=int(n_heads),
                k=int(len(top_heads)),
                excluded=list(top_heads),
                seed=int(rand_seed),
            )

            logits_top = _run_layer_head_ablation(
                model=model,
                tokens=tokens,
                hook_name=str(layer_info["hook_name"]),
                head_indices=list(top_heads),
                layer_mean=head_means[int(layer)],
                target_pos=int(target_pos),
                ablate_positions=str(args.ablate_positions),
            )
            delta_top = _delta_g_from_logits(
                logits_top,
                target_pos=int(target_pos),
                refusal_token_id=int(item["refusal_token_id"]),
                guidance_token_id=int(item["guidance_token_id"]),
            )

            delta_rand: float | None = None
            if random_heads:
                logits_rand = _run_layer_head_ablation(
                    model=model,
                    tokens=tokens,
                    hook_name=str(layer_info["hook_name"]),
                    head_indices=list(random_heads),
                    layer_mean=head_means[int(layer)],
                    target_pos=int(target_pos),
                    ablate_positions=str(args.ablate_positions),
                )
                delta_rand = _delta_g_from_logits(
                    logits_rand,
                    target_pos=int(target_pos),
                    refusal_token_id=int(item["refusal_token_id"]),
                    guidance_token_id=int(item["guidance_token_id"]),
                )

            per_head_effects: Dict[str, float] = {}
            if bool(args.run_per_head_ablation):
                for h in top_heads:
                    logits_h = _run_layer_head_ablation(
                        model=model,
                        tokens=tokens,
                        hook_name=str(layer_info["hook_name"]),
                        head_indices=[int(h)],
                        layer_mean=head_means[int(layer)],
                        target_pos=int(target_pos),
                        ablate_positions=str(args.ablate_positions),
                    )
                    delta_h = _delta_g_from_logits(
                        logits_h,
                        target_pos=int(target_pos),
                        refusal_token_id=int(item["refusal_token_id"]),
                        guidance_token_id=int(item["guidance_token_id"]),
                    )
                    per_head_effects[str(int(h))] = float(baseline - float(delta_h))

            layer_info["top_heads"] = [int(h) for h in top_heads]
            layer_info["random_heads"] = [int(h) for h in random_heads]
            layer_info["delta_g_top_h_ablation"] = float(delta_top)
            layer_info["effect_top_h_ablation"] = float(baseline - float(delta_top))
            if delta_rand is not None:
                layer_info["delta_g_random_h_ablation"] = float(delta_rand)
                layer_info["effect_random_h_ablation"] = float(baseline - float(delta_rand))
            layer_info["per_head_effects"] = {str(k): float(v) for k, v in per_head_effects.items()}
            item["layers"][layer_key] = layer_info

    # Summaries
    bootstrap_cfg = protocol_cfg.get("bootstrap", {}) if isinstance(protocol_cfg, Mapping) else {}
    if isinstance(bootstrap_cfg, Mapping):
        summary_bootstrap_n = int(bootstrap_cfg.get("n", 1000) or 1000)
        summary_bootstrap_ci = float(bootstrap_cfg.get("ci", 0.95) or 0.95)
        summary_bootstrap_seed = int(bootstrap_cfg.get("seed", int(args.seed)) or int(args.seed))
    else:
        summary_bootstrap_n = 1000
        summary_bootstrap_ci = 0.95
        summary_bootstrap_seed = int(args.seed)

    per_layer_effects_top: Dict[int, List[float]] = {}
    per_layer_effects_rand: Dict[int, List[float]] = {}
    for item in per_item:
        for layer_s, layer_info in item["layers"].items():
            layer = int(layer_s)
            et = layer_info.get("effect_top_h_ablation", None)
            er = layer_info.get("effect_random_h_ablation", None)
            if isinstance(et, (int, float)):
                per_layer_effects_top.setdefault(layer, []).append(float(et))
            if isinstance(er, (int, float)):
                per_layer_effects_rand.setdefault(layer, []).append(float(er))

    summary_layers: Dict[str, Any] = {}
    for layer in sorted(set(list(per_layer_effects_top.keys()) + list(per_layer_effects_rand.keys()))):
        layer_sum: Dict[str, Any] = {}
        xs_top = per_layer_effects_top.get(layer, [])
        xs_rand = per_layer_effects_rand.get(layer, [])
        if xs_top:
            mean, lo, hi = bootstrap_ci(
                xs_top,
                n_bootstrap=int(summary_bootstrap_n),
                ci=float(summary_bootstrap_ci),
                seed=int(summary_bootstrap_seed),
            )
            layer_sum["effect_top_h_mean"] = float(mean)
            layer_sum["effect_top_h_ci_low"] = float(lo)
            layer_sum["effect_top_h_ci_high"] = float(hi)
            layer_sum["effect_top_h_n"] = int(len(xs_top))
        if xs_rand:
            mean, lo, hi = bootstrap_ci(
                xs_rand,
                n_bootstrap=int(summary_bootstrap_n),
                ci=float(summary_bootstrap_ci),
                seed=int(summary_bootstrap_seed),
            )
            layer_sum["effect_random_h_mean"] = float(mean)
            layer_sum["effect_random_h_ci_low"] = float(lo)
            layer_sum["effect_random_h_ci_high"] = float(hi)
            layer_sum["effect_random_h_n"] = int(len(xs_rand))
        summary_layers[str(layer)] = layer_sum

    ended_at_utc = datetime.now(timezone.utc).isoformat()
    out: Dict[str, Any] = {
        "artifact_version": "1.0",
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": str(ended_at_utc),
        "wall_time_sec": float(time.perf_counter() - t0),
        "model_name_or_path": str(model_name),
        "model_revision": str(revision),
        "tokenizer_revision": str(tokenizer_revision or revision),
        "device": str(device),
        "torch_dtype": str(args.torch_dtype),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
        "transformers_version": str(transformers_version),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(manifest_sha256),
        "manifest_protocol_sha256": str(manifest_protocol_sha),
        "protocol_path": str(protocol_prov.protocol_path),
        "protocol_path_resolved": str(protocol_prov.protocol_path),
        "protocol_sha256": str(effective_protocol_sha),
        "protocol_sha256_source": str(protocol_sha_source),
        "protocol_sha256_verified": bool(protocol_prov.protocol_sha256_verified),
        "protocol_name": str(protocol_prov.protocol_name),
        "protocol_version": str(protocol_prov.protocol_version),
        "protocol_prereg_tag": str(protocol_prov.protocol_prereg_tag),
        "head_selection_source": str(args.head_selection_source),
        "top_h": int(primary_h),
        "ablate_positions": str(args.ablate_positions),
        "protocol_deviations": list(protocol_deviations),
        "run_per_head_ablation": bool(args.run_per_head_ablation),
        "n_items": int(len(per_item)),
        "summary_ci_method": "iid_item_bootstrap_exploratory",
        "summary_bootstrap_n": int(summary_bootstrap_n),
        "summary_bootstrap_ci": float(summary_bootstrap_ci),
        "summary_bootstrap_seed": int(summary_bootstrap_seed),
        "summary_by_layer": summary_layers,
        "items": per_item,
        "repro": dict(repro),
        "versions": dict(versions),
        "git_commit": str(get_git_commit_hash(repo_root=Path(__file__).resolve().parent, required=False)),
        "argv": list(sys.argv),
    }

    results_path = Path(str(args.results_path)).expanduser().resolve()
    _write_json(results_path, out)
    print(f"Wrote circuits artifact to {str(results_path)}", flush=True)

    # Sidecar run manifest for repository-standard provenance scanning.
    manifest_row: Dict[str, Any] = {
        "model": str(model_name),
        "model_revision": str(revision),
        "tokenizer_revision": str(tokenizer_revision or revision),
        "local_files_only": bool(args.local_files_only),
        "trust_remote_code": bool(args.trust_remote_code),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(manifest_sha256),
        "protocol_sha256": str(effective_protocol_sha),
        "protocol_sha256_source": str(protocol_sha_source),
        "protocol_sha256_verified": bool(protocol_prov.protocol_sha256_verified),
        "n_items": int(len(per_item)),
        "top_h": int(primary_h),
        "head_selection_source": str(args.head_selection_source),
        "ablate_positions": str(args.ablate_positions),
        "run_per_head_ablation": bool(args.run_per_head_ablation),
    }
    run_manifest = build_run_manifest(
        argv=sys.argv,
        results_row=manifest_row,
        dataset_manifest_path=None,
        csv_path=None,
        csv_sha256=None,
        csv_n_rows=None,
    )
    run_manifest["run_status"] = "PASS"
    run_manifest["run_status_reasons"] = []
    run_manifest["run_summary"] = {
        "attempted": int(len(per_item)),
        "succeeded": int(len(per_item)),
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
    run_manifest["started_at_utc"] = str(started_at_utc)
    run_manifest["ended_at_utc"] = str(ended_at_utc)
    run_manifest["wall_time_sec"] = float(time.perf_counter() - t0)
    run_manifest["repro"] = dict(repro)
    run_manifest["versions"] = dict(versions)
    run_manifest["device_backend"] = str(device.type)
    run_manifest["protocol_path"] = str(protocol_prov.protocol_path)
    run_manifest["protocol_path_resolved"] = str(protocol_prov.protocol_path)
    run_manifest["protocol_sha256"] = str(effective_protocol_sha)
    run_manifest["protocol_sha256_source"] = str(protocol_sha_source)
    run_manifest["protocol_sha256_verified"] = bool(protocol_prov.protocol_sha256_verified)
    run_manifest["protocol_name"] = str(protocol_prov.protocol_name)
    run_manifest["protocol_version"] = str(protocol_prov.protocol_version)
    run_manifest["protocol_prereg_tag"] = str(protocol_prov.protocol_prereg_tag)
    run_manifest["protocol_deviations"] = list(protocol_deviations)
    write_run_manifest(results_path.with_suffix(".manifest.json"), run_manifest)


if __name__ == "__main__":
    main()
