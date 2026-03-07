from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.interventions.activation_patching import get_num_layers
from aom.interventions.patching.coh_protocol import COHConstraintAblationProtocol
from aom.interventions.patching.cf_protocol import CFInterventionSwapProtocol
from aom.interventions.patching.disamb_protocol import DISAMBContextSwapProtocol
from aom.interventions.patching.base import run_activation_patching


@dataclass(frozen=True)
class CompletenessResult:
    task: str
    total_effect: float
    total_effect_ci_low: float
    total_effect_ci_high: float
    explained_effect: float
    explained_effect_ci_low: float
    explained_effect_ci_high: float
    completeness: float
    completeness_ci_low: float
    completeness_ci_high: float
    dark_matter: float
    dark_matter_ci_low: float
    dark_matter_ci_high: float
    total_layers: tuple[int, ...]
    explained_layers: tuple[int, ...]

    def to_row(self) -> Dict[str, Any]:
        return {
            "task": str(self.task),
            "total_effect": float(self.total_effect),
            "total_effect_ci_low": float(self.total_effect_ci_low),
            "total_effect_ci_high": float(self.total_effect_ci_high),
            "explained_effect": float(self.explained_effect),
            "explained_effect_ci_low": float(self.explained_effect_ci_low),
            "explained_effect_ci_high": float(self.explained_effect_ci_high),
            "completeness": float(self.completeness),
            "completeness_ci_low": float(self.completeness_ci_low),
            "completeness_ci_high": float(self.completeness_ci_high),
            "dark_matter": float(self.dark_matter),
            "dark_matter_ci_low": float(self.dark_matter_ci_low),
            "dark_matter_ci_high": float(self.dark_matter_ci_high),
            "total_layers": ",".join(str(int(x)) for x in self.total_layers),
            "explained_layers": ",".join(str(int(x)) for x in self.explained_layers),
        }


def _parse_layers(raw: Optional[Sequence[int]], *, n_layers: int) -> List[int]:
    if raw is None:
        return list(range(int(n_layers)))
    out: List[int] = []
    for x in raw:
        i = int(x)
        if 0 <= i < int(n_layers):
            out.append(int(i))
    out = sorted(set(out))
    if not out:
        raise ValueError("No valid layers selected")
    return out


def _extract_explained_layers(task_expl: Mapping[str, Any]) -> List[int]:
    layers: List[int] = []
    heads = task_expl.get("heads", [])
    if isinstance(heads, list):
        for h in heads:
            if isinstance(h, Mapping) and isinstance(h.get("layer", None), int):
                layers.append(int(h["layer"]))
    fams = task_expl.get("feature_families", [])
    if isinstance(fams, list):
        for f in fams:
            if isinstance(f, Mapping) and isinstance(f.get("layer", None), int):
                layers.append(int(f["layer"]))
    return sorted(set(layers))


def _safe_ratio(num: float, den: float, eps: float = 1e-8) -> float:
    return float(float(num) / max(float(eps), float(den)))


def _stats_from_result(result: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(result.get("mean_max_effect", float("nan"))),
        float(result.get("mean_max_effect_ci_low", float("nan"))),
        float(result.get("mean_max_effect_ci_high", float("nan"))),
    )


def _protocol_for_task(task: str):
    t = str(task)
    if t == "disamb":
        return DISAMBContextSwapProtocol()
    if t == "cf":
        return CFInterventionSwapProtocol()
    if t == "coh":
        return COHConstraintAblationProtocol()
    raise ValueError(f"Unsupported task: {task!r}")


@torch.inference_mode()
def compute_task_completeness(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[Any],
    task: str,
    device: torch.device,
    total_layers: Optional[Sequence[int]] = None,
    explanation: Optional[Mapping[str, Any]] = None,
    normalize_by_length: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> CompletenessResult:
    n_layers = int(get_num_layers(model))
    total = _parse_layers(total_layers, n_layers=n_layers)

    protocol = _protocol_for_task(str(task))
    total_stats = run_activation_patching(
        model=model,
        tokenizer=tokenizer,
        protocol=protocol,
        items=list(items),
        device=device,
        layers=total,
        normalize_by_length=bool(normalize_by_length),
        ci=float(ci),
        bootstrap_n=int(bootstrap_n),
        bootstrap_seed=int(bootstrap_seed),
    )
    t_mean, t_lo, t_hi = _stats_from_result(total_stats)

    explained_layers: List[int] = []
    if isinstance(explanation, Mapping):
        task_expl = explanation.get(str(task), None)
        if isinstance(task_expl, Mapping):
            explained_layers = _extract_explained_layers(task_expl)
    explained_layers = [l for l in explained_layers if l in set(total)]

    if explained_layers:
        expl_stats = run_activation_patching(
            model=model,
            tokenizer=tokenizer,
            protocol=protocol,
            items=list(items),
            device=device,
            layers=explained_layers,
            normalize_by_length=bool(normalize_by_length),
            ci=float(ci),
            bootstrap_n=int(bootstrap_n),
            bootstrap_seed=int(bootstrap_seed),
        )
        e_mean, e_lo, e_hi = _stats_from_result(expl_stats)
    else:
        e_mean, e_lo, e_hi = 0.0, 0.0, 0.0

    comp = _safe_ratio(e_mean, t_mean)
    comp_lo = _safe_ratio(e_lo, t_hi)
    comp_hi = _safe_ratio(e_hi, t_lo if abs(float(t_lo)) > 1e-8 else 1e-8)

    dark = float(t_mean - e_mean)
    dark_lo = float(t_lo - e_hi)
    dark_hi = float(t_hi - e_lo)

    return CompletenessResult(
        task=str(task),
        total_effect=float(t_mean),
        total_effect_ci_low=float(t_lo),
        total_effect_ci_high=float(t_hi),
        explained_effect=float(e_mean),
        explained_effect_ci_low=float(e_lo),
        explained_effect_ci_high=float(e_hi),
        completeness=float(comp),
        completeness_ci_low=float(comp_lo),
        completeness_ci_high=float(comp_hi),
        dark_matter=float(dark),
        dark_matter_ci_low=float(dark_lo),
        dark_matter_ci_high=float(dark_hi),
        total_layers=tuple(int(x) for x in total),
        explained_layers=tuple(int(x) for x in explained_layers),
    )


def parse_explanation_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if not isinstance(cfg, Mapping):
        raise ValueError("explanation config must be a mapping")
    out: Dict[str, Any] = {}
    for task in ("disamb", "cf", "coh"):
        v = cfg.get(task, None)
        if isinstance(v, Mapping):
            out[str(task)] = dict(v)
    return out
