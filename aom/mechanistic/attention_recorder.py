from __future__ import annotations

import inspect
from typing import Dict, Iterable, List, Optional

import torch

from aom.interventions.activation_patching import detect_architecture, get_decoder_blocks
from .induction import (
    calculate_first_half_mass,
    calculate_induction_control,
    calculate_induction_diag_fraction,
    calculate_induction_score,
    validate_attention_weights,
)


class AttentionPatternRecorder:
    """
    Architecture-aware attention recorder that computes induction statistics in-hook.
    """

    def __init__(
        self,
        model,
        *,
        layers: Iterable[int],
        base_len: int,
        repeats: int = 2,
        baseline_mode: str = "shuffle",
        debug_store_attn: bool = False,
        validate_attn: str = "none",
    ) -> None:
        self.model = model
        self.architecture = detect_architecture(model)
        self.layers = [int(x) for x in layers]
        self.base_len = int(base_len)
        self.repeats = int(repeats)
        self.seq_len = int(self.base_len * self.repeats)
        if self.repeats != 2:
            raise ValueError("repeats must be 2 for V1 induction metric")
        if baseline_mode not in {"shuffle", "offset0"}:
            raise ValueError(f"invalid baseline_mode={baseline_mode!r}")
        self.baseline_mode = baseline_mode
        self.debug_store_attn = bool(debug_store_attn)
        if validate_attn not in {"none", "first", "always"}:
            raise ValueError(f"invalid validate_attn={validate_attn!r}")
        self.validate_attn = validate_attn
        self._validated_layers: set[int] = set()
        self.mode = "repeat"
        self._prev_attn_impl: Optional[str] = None

        self._originals: List[tuple[object, str, object, bool]] = []
        self._samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._mass_samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._fraction_samples: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "repeat": {int(l): [] for l in self.layers},
            "control": {int(l): [] for l in self.layers},
        }
        self._debug_attn: Dict[int, Optional[torch.Tensor]] = {int(l): None for l in self.layers}

    def set_mode(self, mode: str) -> None:
        if mode not in {"repeat", "control"}:
            raise ValueError(f"invalid mode={mode!r}")
        self.mode = mode

    def __enter__(self) -> "AttentionPatternRecorder":
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "_attn_implementation"):
                self._prev_attn_impl = str(self.model.config._attn_implementation)
                if self._prev_attn_impl != "eager":
                    self.model.config._attn_implementation = "eager"
            if self.architecture == "gpt2":
                self._install_gpt2_hooks()
            elif self.architecture in {"llama", "qwen2", "qwen3"}:
                self._install_llama_like_hooks()
            else:
                raise ValueError(f"unsupported architecture for recorder: {self.architecture}")
        except Exception:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore()

    def _restore(self) -> None:
        for module, attr, original, had_instance_attr in reversed(self._originals):
            if had_instance_attr:
                setattr(module, attr, original)
            else:
                if hasattr(module, "__dict__") and attr in module.__dict__:
                    delattr(module, attr)
        self._originals = []
        if self._prev_attn_impl is not None and hasattr(self.model, "config"):
            self.model.config._attn_implementation = self._prev_attn_impl
        self._prev_attn_impl = None

    def _record_original(self, module: object, attr: str, original: object) -> None:
        had_instance_attr = hasattr(module, "__dict__") and attr in module.__dict__
        self._originals.append((module, attr, original, had_instance_attr))

    def _install_gpt2_hooks(self) -> None:
        blocks = get_decoder_blocks(self.model)
        for layer in self.layers:
            attn = blocks[layer].attn
            original = attn.forward
            sig = inspect.signature(original)
            if "output_attentions" not in sig.parameters:
                raise ValueError(
                    "GPT-2 attention forward lacks output_attentions parameter; "
                    "use eager backend or update registry."
                )

            def _make_wrapped(orig_forward, layer_idx: int):
                def wrapped(*args, **kwargs):
                    kwargs = dict(kwargs)
                    kwargs["output_attentions"] = True
                    out = orig_forward(*args, **kwargs)
                    if not isinstance(out, tuple):
                        raise ValueError("unexpected attention output structure")
                    attn_weights = self._extract_attn_weights(out)
                    self._capture(layer_idx, attn_weights)
                    return out

                return wrapped

            wrapped = _make_wrapped(original, int(layer))
            attn.forward = wrapped
            self._record_original(attn, "forward", original)

    def _install_llama_like_hooks(self) -> None:
        blocks = get_decoder_blocks(self.model)
        for layer in self.layers:
            attn = blocks[layer].self_attn
            original = attn.forward
            sig = inspect.signature(original)
            has_output_attn = "output_attentions" in sig.parameters

            def _make_wrapped(orig_forward, layer_idx: int, has_output_attentions: bool):
                def wrapped(*args, **kwargs):
                    if has_output_attentions:
                        want_attn = bool(kwargs.get("output_attentions", False))
                        kwargs = dict(kwargs)
                        kwargs["output_attentions"] = True
                        out = orig_forward(*args, **kwargs)
                        if not isinstance(out, tuple) or len(out) < 2:
                            raise ValueError("unexpected attention output structure")
                        attn_weights = self._extract_attn_weights_llama_like(out)
                        self._capture(layer_idx, attn_weights)
                        if want_attn:
                            return out
                        out_list = list(out)
                        out_list[1] = None
                        return tuple(out_list)
                    else:
                        out = orig_forward(*args, **kwargs)
                        if not isinstance(out, tuple) or len(out) < 2:
                            raise ValueError("unexpected attention output structure")
                        attn_weights = self._extract_attn_weights_llama_like(out)
                        self._capture(layer_idx, attn_weights)
                        return out

                return wrapped

            wrapped = _make_wrapped(original, int(layer), bool(has_output_attn))
            attn.forward = wrapped
            self._record_original(attn, "forward", original)

    def _capture(self, layer: int, attn_weights: Optional[torch.Tensor]) -> None:
        if attn_weights is None:
            raise ValueError("attention weights are None; use eager backend or update registry")
        if not isinstance(attn_weights, torch.Tensor):
            raise ValueError("attention weights must be a torch.Tensor")
        if self.validate_attn != "none":
            if self.validate_attn == "always" or int(layer) not in self._validated_layers:
                validate_attention_weights(attn_weights, self.seq_len)
                if self.validate_attn == "first":
                    self._validated_layers.add(int(layer))

        if self.debug_store_attn:
            self._debug_attn[int(layer)] = attn_weights.detach().float().cpu()

        scores = calculate_induction_score(attn_weights, self.base_len).detach().float().cpu()
        mass = calculate_first_half_mass(attn_weights, self.base_len).detach().float().cpu()
        frac = calculate_induction_diag_fraction(attn_weights, self.base_len).detach().float().cpu()
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if mass.ndim == 1:
            mass = mass.unsqueeze(0)
        if frac.ndim == 1:
            frac = frac.unsqueeze(0)

        if self.baseline_mode == "offset0":
            baseline = calculate_induction_control(attn_weights, self.base_len, mode="offset0").detach().float().cpu()
            if baseline.ndim == 1:
                baseline = baseline.unsqueeze(0)
            self._samples["repeat"][int(layer)].append(scores)
            self._samples["control"][int(layer)].append(baseline)
            self._mass_samples["repeat"][int(layer)].append(mass)
            self._mass_samples["control"][int(layer)].append(mass)
            self._fraction_samples["repeat"][int(layer)].append(frac)
            self._fraction_samples["control"][int(layer)].append(frac)
        else:
            self._samples[self.mode][int(layer)].append(scores)
            self._mass_samples[self.mode][int(layer)].append(mass)
            self._fraction_samples[self.mode][int(layer)].append(frac)

    def get_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        if mode not in self._samples:
            raise ValueError(f"invalid mode={mode!r}")
        return list(self._samples[mode][int(layer)])

    def get_mass_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        if mode not in self._mass_samples:
            raise ValueError(f"invalid mode={mode!r}")
        return list(self._mass_samples[mode][int(layer)])

    def get_fraction_samples(self, layer: int, mode: str) -> List[torch.Tensor]:
        if mode not in self._fraction_samples:
            raise ValueError(f"invalid mode={mode!r}")
        return list(self._fraction_samples[mode][int(layer)])

    def get_last_attn(self, layer: int) -> Optional[torch.Tensor]:
        return self._debug_attn.get(int(layer))

    def _extract_attn_weights(self, out: tuple) -> torch.Tensor:
        if len(out) < 2:
            raise ValueError("unexpected GPT-2 attention output length")
        attn_weights = out[1]
        if attn_weights is None:
            raise ValueError("GPT-2 attention weights are None; ensure _attn_implementation='eager'")
        if not isinstance(attn_weights, torch.Tensor):
            raise ValueError("GPT-2 attention weights are not a tensor; update registry")
        return attn_weights

    def _extract_attn_weights_llama_like(self, out: tuple) -> torch.Tensor:
        # Prefer the second element if it looks like attention weights.
        if len(out) >= 2 and isinstance(out[1], torch.Tensor):
            attn = out[1]
            if attn.ndim == 4:
                return attn
        # Fallback: search for a 4D tensor matching seq_len.
        candidates = []
        for item in out:
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                if item.shape[-1] == self.seq_len and item.shape[-2] == self.seq_len:
                    candidates.append(item)
        if len(candidates) != 1:
            raise ValueError("could not identify attention weights in llama-like output")
        return candidates[0]
