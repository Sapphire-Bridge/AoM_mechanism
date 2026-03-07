from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from aom.mechanistic.backends.hooks import HookSpec
from aom.mechanistic.backends.transformer_lens import make_patch_pattern_hook, make_patch_value_hook
from aom.stats import bootstrap_ci


@dataclass(frozen=True)
class WhyFetchExample:
    task: str
    item_id: str
    prompt_clean: str
    prompt_corrupt: str
    query_pos: int
    evidence_span: Tuple[int, int]
    token_expected_id: int
    token_other_id: int


def _parse_layer_from_key(key: str, *, prefix: str) -> Optional[int]:
    k = str(key)
    if not k.startswith(str(prefix)):
        return None
    try:
        return int(k.rsplit(".", 1)[1])
    except Exception:
        return None


def _idx(pos: int, length: int) -> int:
    p = int(pos)
    if p < 0:
        p = int(length) + p
    if p < 0 or p >= int(length):
        raise ValueError(f"position out of range: {int(pos)} for length={int(length)}")
    return int(p)


def _logit_diff(logits: torch.Tensor, *, query_pos: int, token_expected_id: int, token_other_id: int) -> float:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits rank-3 [batch, pos, vocab], got {tuple(logits.shape)}")
    qp = _idx(int(query_pos), int(logits.size(1)))
    return float((logits[0, qp, int(token_expected_id)] - logits[0, qp, int(token_other_id)]).item())


def compute_fetch_mass(
    cache: Mapping[str, Any],
    *,
    q_pos: int,
    evidence_span: Tuple[int, int],
) -> Dict[int, torch.Tensor]:
    """
    Compute fetch mass per layer/head:
      fetch_mass = sum_{t in evidence_span} pattern[q_pos, t]

    Returns: dict[layer] -> tensor[n_heads]
    """
    s, e = int(evidence_span[0]), int(evidence_span[1])
    if s < 0 or e <= s:
        raise ValueError(f"Invalid evidence_span: {(s, e)!r}")

    out: Dict[int, torch.Tensor] = {}
    for key, val in cache.items():
        layer = _parse_layer_from_key(str(key), prefix="pattern.")
        if layer is None or not torch.is_tensor(val):
            continue
        patt = val
        if patt.ndim != 4:
            continue
        # [batch, head, q, k]
        q_idx = _idx(int(q_pos), int(patt.size(2)))
        start = max(0, min(int(s), int(patt.size(3))))
        end = max(start, min(int(e), int(patt.size(3))))
        if end <= start:
            continue
        mass = patt[0, :, q_idx, start:end].sum(dim=-1).detach().to(dtype=torch.float32).cpu()
        out[int(layer)] = mass
    return out


def compute_qk_ov_effects(
    *,
    hook_backend: Any,
    example: WhyFetchExample,
    clean_cache: Mapping[str, Any],
    corrupt_cache: Mapping[str, Any],
    tokens_corrupt: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    heads_topk: int = 0,
) -> List[Dict[str, Any]]:
    """
    Compute per-head QK/OV causal effects for one clean/corrupted example pair.
    """
    fetch_by_layer = compute_fetch_mass(
        corrupt_cache,
        q_pos=int(example.query_pos),
        evidence_span=tuple(example.evidence_span),
    )
    if not fetch_by_layer:
        return []

    if layers is None:
        layer_ids = sorted(fetch_by_layer.keys())
    else:
        layer_ids = [int(l) for l in layers if int(l) in fetch_by_layer]

    baseline = _logit_diff(
        corrupt_cache["logits"],  # type: ignore[index]
        query_pos=int(example.query_pos),
        token_expected_id=int(example.token_expected_id),
        token_other_id=int(example.token_other_id),
    ) if "logits" in corrupt_cache else None
    if baseline is None:
        raise ValueError("corrupt_cache must include logits under key 'logits'")

    s, e = int(example.evidence_span[0]), int(example.evidence_span[1])
    rows: List[Dict[str, Any]] = []
    for layer in layer_ids:
        masses = fetch_by_layer[int(layer)]
        n_heads = int(masses.numel())
        order = sorted(range(n_heads), key=lambda h: float(masses[h].item()), reverse=True)
        if int(heads_topk) > 0:
            order = order[: int(heads_topk)]
        for head in order:
            qk_hook = make_patch_pattern_hook(
                layer=int(layer),
                head=int(head),
                source_cache=dict(clean_cache),
                q_pos=int(example.query_pos),
            )
            logits_qk = hook_backend.run_with_hooks(
                prompt=str(example.prompt_corrupt),
                hooks=[qk_hook],
                tokens=tokens_corrupt,
                prepend_bos=False,
            )
            y_qk = _logit_diff(
                logits_qk,
                query_pos=int(example.query_pos),
                token_expected_id=int(example.token_expected_id),
                token_other_id=int(example.token_other_id),
            )

            positions = list(range(int(s), int(e)))
            value_key = f"value.{int(layer)}"
            src_v = clean_cache.get(value_key, None)
            dst_v = corrupt_cache.get(value_key, None)
            if not (torch.is_tensor(src_v) and torch.is_tensor(dst_v) and src_v.ndim == 4 and dst_v.ndim == 4):
                continue
            max_pos = min(int(src_v.size(1)), int(dst_v.size(1)))
            valid_positions = [p for p in positions if 0 <= int(p) < max_pos]
            if not valid_positions:
                continue

            ov_hook = make_patch_value_hook(
                layer=int(layer),
                head=int(head),
                source_cache=dict(clean_cache),
                positions=valid_positions,
            )
            logits_ov = hook_backend.run_with_hooks(
                prompt=str(example.prompt_corrupt),
                hooks=[ov_hook],
                tokens=tokens_corrupt,
                prepend_bos=False,
            )
            y_ov = _logit_diff(
                logits_ov,
                query_pos=int(example.query_pos),
                token_expected_id=int(example.token_expected_id),
                token_other_id=int(example.token_other_id),
            )

            rows.append(
                {
                    "task": str(example.task),
                    "item_id": str(example.item_id),
                    "layer": int(layer),
                    "head": int(head),
                    "fetch_mass": float(masses[head].item()),
                    "baseline_corrupt": float(baseline),
                    "y_qk": float(y_qk),
                    "y_ov": float(y_ov),
                    "delta_qk": float(y_qk - baseline),
                    "delta_ov": float(y_ov - baseline),
                    "query_pos": int(example.query_pos),
                    "evidence_start": int(s),
                    "evidence_end": int(e),
                }
            )
    return rows


def aggregate_top_heads(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_n: int = 500,
    bootstrap_seed: int = 42,
    ci: float = 0.95,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int], Dict[str, List[float]]] = {}
    for r in rows:
        key = (str(r.get("task", "")), int(r.get("layer", -1)), int(r.get("head", -1)))
        g = grouped.setdefault(key, {"fetch_mass": [], "delta_qk": [], "delta_ov": []})
        for m in ("fetch_mass", "delta_qk", "delta_ov"):
            v = r.get(m, None)
            if isinstance(v, (int, float)):
                g[m].append(float(v))

    out: List[Dict[str, Any]] = []
    for (task, layer, head), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        fm_mean, fm_lo, fm_hi = bootstrap_ci(
            vals["fetch_mass"],
            n_bootstrap=int(bootstrap_n),
            ci=float(ci),
            seed=int(bootstrap_seed),
        )
        qk_mean, qk_lo, qk_hi = bootstrap_ci(
            vals["delta_qk"],
            n_bootstrap=int(bootstrap_n),
            ci=float(ci),
            seed=int(bootstrap_seed),
        )
        ov_mean, ov_lo, ov_hi = bootstrap_ci(
            vals["delta_ov"],
            n_bootstrap=int(bootstrap_n),
            ci=float(ci),
            seed=int(bootstrap_seed),
        )
        out.append(
            {
                "task": str(task),
                "layer": int(layer),
                "head": int(head),
                "n_samples": int(len(vals["fetch_mass"])),
                "fetch_mass_mean": float(fm_mean),
                "fetch_mass_ci_low": float(fm_lo),
                "fetch_mass_ci_high": float(fm_hi),
                "delta_qk_mean": float(qk_mean),
                "delta_qk_ci_low": float(qk_lo),
                "delta_qk_ci_high": float(qk_hi),
                "delta_ov_mean": float(ov_mean),
                "delta_ov_ci_low": float(ov_lo),
                "delta_ov_ci_high": float(ov_hi),
            }
        )
    out.sort(key=lambda r: float(r["fetch_mass_mean"]), reverse=True)
    return out


def make_uniform_pattern_hook(*, layer: int, head: int) -> HookSpec:
    """
    Utility hook used in tests: force one head's pattern to uniform over keys.
    """

    def _uniform(pattern: torch.Tensor, _hook: Any) -> torch.Tensor:
        if pattern.ndim != 4:
            raise ValueError(f"Expected rank-4 pattern, got {tuple(pattern.shape)}")
        patched = pattern.clone()
        patched[:, int(head), :, :] = 1.0 / float(pattern.size(-1))
        return patched

    return HookSpec(name=f"pattern.{int(layer)}", fn=_uniform)
