from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import transformers

from aom.interventions.activation_patching import get_num_layers
from aom.mechanistic.attention_recorder import AttentionPatternRecorder
from aom.mechanistic.synthetic import build_allowed_ids, generate_induction_batch, permute_first_half
from aom.models.loader import load_causal_lm

from .base import BackendLoadResult, BackendRunResult, InductionCollector, InductionBackend
from .hooks import CachedRun, HookSpec, HookableBackend


def _max_position_embeddings(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


@dataclass(frozen=True)
class HFEagerBackend(InductionBackend):
    name: str = "hf"

    def load(
        self,
        *,
        model_name_or_path: str,
        device: torch.device,
        torch_dtype: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: str,
    ) -> BackendLoadResult:
        loaded = load_causal_lm(
            model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
            attn_implementation=attn_implementation,  # recorder forces eager if needed
            device_map=None,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer

        exclude_ids: Set[int] = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_id, int):
            exclude_ids.add(int(pad_id))

        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None or int(vocab_size) < 1:
            raise ValueError("could not determine vocab_size for synthetic generation")

        n_layers = int(get_num_layers(model))
        max_seq_len = _max_position_embeddings(model)

        param0 = next(model.parameters(), None)
        model_param_dtype = str(param0.dtype) if param0 is not None else ""
        model_param_device = str(param0.device) if param0 is not None else ""

        num_attention_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
        num_key_value_heads = getattr(getattr(model, "config", None), "num_key_value_heads", None)

        return BackendLoadResult(
            model=model,
            tokenizer=tokenizer,
            architecture=str(loaded.architecture),
            device=device,
            n_layers=n_layers,
            vocab_size=int(vocab_size),
            exclude_token_ids=frozenset(exclude_ids),
            max_seq_len=max_seq_len,
            model_param_dtype=model_param_dtype,
            model_param_device=model_param_device,
            num_attention_heads=int(num_attention_heads) if isinstance(num_attention_heads, int) else None,
            num_key_value_heads=int(num_key_value_heads) if isinstance(num_key_value_heads, int) else None,
            backend_version=str(transformers.__version__),
        )

    def run_batches(
        self,
        *,
        loaded: BackendLoadResult,
        layers: List[int],
        base_len: int,
        repeats: int,
        batch_size: int,
        n_batches: int,
        seed: int,
        baseline: str,
        debug_store_attn: bool,
        validate_attn: str,
        tl_crosscheck: bool,
        bootstrap_n: int,
        bootstrap_seed: int,
        ci: float,
    ) -> BackendRunResult:
        model = loaded.model
        device = loaded.device
        _ = bool(tl_crosscheck)
        _ = int(bootstrap_n)
        _ = int(bootstrap_seed)
        _ = float(ci)

        allowed_ids = build_allowed_ids(int(loaded.vocab_size), set(loaded.exclude_token_ids))
        recorder = AttentionPatternRecorder(
            model,
            layers=layers,
            base_len=int(base_len),
            repeats=int(repeats),
            baseline_mode=str(baseline),
            debug_store_attn=bool(debug_store_attn),
            validate_attn=str(validate_attn),
        )
        model.eval()

        with torch.no_grad(), recorder:
            for batch_idx in range(int(n_batches)):
                batch_seed = int(seed) + int(batch_idx)
                base_ids = generate_induction_batch(
                    vocab_size=int(loaded.vocab_size),
                    base_len=int(base_len),
                    repeats=int(repeats),
                    batch_size=int(batch_size),
                    seed=int(batch_seed),
                    allowed_ids=allowed_ids,
                )
                input_ids = base_ids.to(device)
                attention_mask = torch.ones_like(input_ids)

                recorder.set_mode("repeat")
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                if str(baseline) == "shuffle":
                    permuted = permute_first_half(
                        base_ids, base_len=int(base_len), seed=int(batch_seed), resample_identity=True
                    )
                    permuted = permuted.to(device)
                    recorder.set_mode("control")
                    _ = model(input_ids=permuted, attention_mask=attention_mask, use_cache=False)

        return BackendRunResult(collector=recorder, extras={})


def _parse_normalized_capture_name(name: str) -> tuple[str, Optional[int]]:
    raw = str(name).strip()
    if not raw:
        raise ValueError("empty capture name")
    if "." not in raw:
        return raw, None
    base, idx = raw.rsplit(".", 1)
    if idx.isdigit():
        return str(base), int(idx)
    return raw, None


def _normalized_capture_names(capture: Sequence[str], *, n_layers: int) -> List[str]:
    layer_scoped = {"pattern", "value", "head_result", "attn_out", "mlp_out", "resid_pre", "resid_post"}
    global_scoped = {"embed", "pos_embed", "resid_final"}
    out: List[str] = []
    for item in capture:
        base, layer = _parse_normalized_capture_name(str(item))
        if layer is not None:
            if layer < 0 or layer >= int(n_layers):
                raise ValueError(f"capture layer out of range: {item!r} for n_layers={int(n_layers)}")
            out.append(f"{base}.{int(layer)}")
            continue
        if base in layer_scoped:
            out.extend([f"{base}.{int(l)}" for l in range(int(n_layers))])
            continue
        if base in global_scoped:
            out.append(str(base))
            continue
        raise ValueError(f"unknown capture key: {item!r}")
    seen: set[str] = set()
    uniq: List[str] = []
    for key in out:
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


@dataclass(frozen=True)
class HFEagerHookableBackend(HookableBackend):
    """
    Capture-first fallback for HF eager models.

    Supports logits and residual captures. Pattern/value patching is intentionally
    unsupported in this backend to keep QK/OV analyses TL-only in v1.
    """

    model: Any
    tokenizer: Any

    def _resolve_tokens(self, prompt: str, *, tokens: Optional[torch.Tensor], add_special_tokens: bool) -> torch.Tensor:
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 2:
                raise ValueError(f"tokens must be rank-2 [batch, pos], got shape={tuple(tokens.shape)}")
            return tokens.to(next(self.model.parameters()).device)
        enc = self.tokenizer(str(prompt), return_tensors="pt", add_special_tokens=bool(add_special_tokens))
        ids = enc["input_ids"]
        if not isinstance(ids, torch.Tensor):
            raise ValueError("tokenizer did not produce tensor input_ids")
        return ids.to(next(self.model.parameters()).device)

    def run_with_cache(self, prompt: str, *, capture: List[str], **kwargs: Any) -> CachedRun:
        tokens = self._resolve_tokens(
            prompt,
            tokens=kwargs.pop("tokens", None),
            add_special_tokens=bool(kwargs.pop("add_special_tokens", False)),
        )
        # We request attentions only when needed, because many HF kernels disable them.
        wants_pattern = any(str(c).split(".", 1)[0] == "pattern" for c in capture)
        out = self.model(
            input_ids=tokens,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=bool(wants_pattern),
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Model did not return hidden states.")
        n_layers = int(len(hs) - 1)
        capture_keys = _normalized_capture_names(capture, n_layers=n_layers)
        cache: Dict[str, Any] = {}
        missing: List[str] = []
        for key in capture_keys:
            base, layer = _parse_normalized_capture_name(str(key))
            if layer is None:
                if base == "embed":
                    cache[str(key)] = hs[0].detach().clone()
                    continue
                if base == "pos_embed":
                    # Not generally exposed in HF outputs.
                    missing.append(str(key))
                    continue
                if base == "resid_final":
                    cache[str(key)] = hs[-1].detach().clone()
                    continue
                missing.append(str(key))
                continue

            l = int(layer)
            if base == "resid_pre":
                cache[str(key)] = hs[l].detach().clone()
                continue
            if base == "resid_post":
                cache[str(key)] = hs[l + 1].detach().clone()
                continue
            if base == "pattern":
                atts = out.attentions
                if atts is None or l >= len(atts) or atts[l] is None:
                    missing.append(str(key))
                    continue
                cache[str(key)] = atts[l].detach().clone()
                continue
            # HF eager fallback does not expose these consistently.
            if base in {"value", "head_result", "attn_out", "mlp_out"}:
                missing.append(str(key))
                continue
            missing.append(str(key))

        return CachedRun(
            logits=out.logits,
            cache=cache,
            meta={
                "tokens": tokens.detach().clone(),
                "n_layers": n_layers,
                "missing_capture_keys": missing,
            },
        )

    def run_with_hooks(self, prompt: str, *, hooks: List[HookSpec], **kwargs: Any) -> Any:
        if hooks:
            raise NotImplementedError(
                "HFEagerHookableBackend does not support pattern/value hook patching. "
                "Use TransformerLensHookableBackend for QK/OV analyses."
            )
        tokens = self._resolve_tokens(
            prompt,
            tokens=kwargs.pop("tokens", None),
            add_special_tokens=bool(kwargs.pop("add_special_tokens", False)),
        )
        out = self.model(input_ids=tokens, use_cache=False, return_dict=True)
        return out.logits
