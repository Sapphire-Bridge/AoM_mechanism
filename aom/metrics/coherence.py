from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..data.schemas import CoherenceItem
from ..utils import bootstrap_ci_metric
from .disamb import score_labels_next_continuations


@torch.no_grad()
def compute_aom_coh(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[CoherenceItem],
    device: torch.device,
    *,
    normalize_by_length: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    Discourse coherence / constraint tracking.

    Each item provides a context plus a set of valid and invalid continuations. We score each
    continuation via log-probability under the model (optionally length-normalized).

    We report three binary coherence criteria per item:
    - max-pool: max(valid) > max(invalid)
    - mean-pool (primary `constraint_accuracy`): mean(valid) > mean(invalid)
    - strict: min(valid) > max(invalid)

    We also report a pairwise ranking/AUC-style statistic per item:
      AUC = mean_{v in V, u in U}[ 1(v > u) + 0.5*1(v == u) ] in [0, 1].
    """
    def _boot(name: str, values: List[float]) -> Dict[str, Any]:
        mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
        out: Dict[str, Any] = {
            name: float(mv.value),
            f"{name}_ci_low": float(lo),
            f"{name}_ci_high": float(hi),
            f"{name}_n": int(mv.n),
            f"{name}_valid": bool(mv.valid),
        }
        if mv.reason is not None:
            out[f"{name}_reason"] = str(mv.reason)
        return out

    def _base_id(item_id: str) -> str:
        item_id = str(item_id)
        return item_id.split("__", 1)[0] if "__" in item_id else item_id

    # Score all items once; then compute primary/group/paired analyses from cached per-item stats.
    scored: List[Dict[str, float | str]] = []
    for it in items:
        group = str(getattr(it, "group", "main"))
        base_id = _base_id(it.item_id)
        n_constraints = 1
        meta = getattr(it, "metadata", None)
        if isinstance(meta, dict):
            nc = meta.get("n_constraints", None)
            if isinstance(nc, int):
                n_constraints = int(nc)

        # Treat each continuation as its own "label" for scoring.
        valid_map = {f"v{i}": [c] for i, c in enumerate(it.valid_continuations)}
        invalid_map = {f"u{i}": [c] for i, c in enumerate(it.invalid_continuations)}
        valid_scores = score_labels_next_continuations(
            model, tokenizer, it.context, valid_map, device, normalize_by_length=normalize_by_length
        )
        invalid_scores = score_labels_next_continuations(
            model, tokenizer, it.context, invalid_map, device, normalize_by_length=normalize_by_length
        )

        valid_vals = list(valid_scores.by_label.values())
        invalid_vals = list(invalid_scores.by_label.values())
        best_valid = max(valid_vals)
        best_invalid = max(invalid_vals)
        mean_valid = float(sum(valid_vals) / max(1, len(valid_vals)))
        mean_invalid = float(sum(invalid_vals) / max(1, len(invalid_vals)))
        worst_valid = min(valid_vals)

        # AUC-style pairwise ranking statistic.
        denom = float(len(valid_vals) * len(invalid_vals))
        wins = 0.0
        if denom > 0:
            for v in valid_vals:
                for u in invalid_vals:
                    if v > u:
                        wins += 1.0
                    elif v == u:
                        wins += 0.5
        pairwise_auc = wins / denom if denom > 0 else 0.0

        # Context length in tokenizer tokens (useful for ablation length checks).
        context_tokens = float(len(tokenizer.encode(it.context, add_special_tokens=False)))

        scored.append(
            {
                "item_id": str(it.item_id),
                "base_id": str(base_id),
                "group": str(group),
                "n_constraints": str(int(n_constraints)),
                "maxpool_correct": float(best_valid > best_invalid),
                "meanpool_correct": float(mean_valid > mean_invalid),
                "strict_correct": float(worst_valid > best_invalid),
                "pairwise_auc": float(pairwise_auc),
                "max_gap": float(best_valid - best_invalid),
                "mean_gap": float(mean_valid - mean_invalid),
                "strict_gap": float(worst_valid - best_invalid),
                "context_tokens": float(context_tokens),
            }
        )

    primary = [r for r in scored if r["group"] == "main"]
    if not primary:
        primary = list(scored)

    maxpool_acc = [float(r["maxpool_correct"]) for r in primary]
    meanpool_acc = [float(r["meanpool_correct"]) for r in primary]
    strict_acc = [float(r["strict_correct"]) for r in primary]
    maxpool_gaps = [float(r["max_gap"]) for r in primary]
    meanpool_gaps = [float(r["mean_gap"]) for r in primary]
    strict_gaps = [float(r["strict_gap"]) for r in primary]
    pairwise = [float(r["pairwise_auc"]) for r in primary]

    out: Dict[str, Any] = {}
    # Primary metric: mean-pool preference (less "one good continuation hides failures" than max-pool).
    out.update(_boot("constraint_accuracy", meanpool_acc))
    out["violation_rate"] = 1.0 - float(out["constraint_accuracy"])
    out.update(_boot("pairwise_auc", pairwise))
    out.update(_boot("mean_gap", meanpool_gaps))
    out.update(_boot("maxpool_accuracy", maxpool_acc))
    out["maxpool_violation_rate"] = 1.0 - float(out["maxpool_accuracy"])
    out.update(_boot("strict_accuracy", strict_acc))
    out["strict_violation_rate"] = 1.0 - float(out["strict_accuracy"])
    out.update(_boot("strict_gap", strict_gaps))
    out["n_items_total"] = int(len(primary))
    out["n_items_total_all_groups"] = int(len(scored))

    # Stratify by number of constraints when available (defaults to 1).
    n_constraints_values = sorted({int(r.get("n_constraints", "1")) for r in primary})
    if len(n_constraints_values) > 1:
        for nc in n_constraints_values:
            nc_rows = [r for r in primary if int(r.get("n_constraints", "1")) == int(nc)]
            nc_acc = [float(r["meanpool_correct"]) for r in nc_rows]
            nc_gap = [float(r["mean_gap"]) for r in nc_rows]
            nc_auc = [float(r["pairwise_auc"]) for r in nc_rows]
            out.update(_boot(f"n_constraints_{int(nc)}_constraint_accuracy", nc_acc))
            out.update(_boot(f"n_constraints_{int(nc)}_mean_gap", nc_gap))
            out.update(_boot(f"n_constraints_{int(nc)}_pairwise_auc", nc_auc))
            out[f"n_constraints_{int(nc)}_n_items_total"] = int(len(nc_rows))

    # If there are multiple groups, also report group-level mean-pool accuracy.
    groups = sorted({str(r["group"]) for r in scored})
    if len(groups) > 1:
        for g in groups:
            group_rows = [r for r in scored if r["group"] == g]
            group_acc = [float(r["meanpool_correct"]) for r in group_rows]
            group_pair = [float(r["pairwise_auc"]) for r in group_rows]
            group_ctx = [float(r["context_tokens"]) for r in group_rows]

            slug = "".join(ch if ch.isalnum() else "_" for ch in str(g).strip().lower())
            out.update(_boot(f"group_{slug}_constraint_accuracy", group_acc))
            out[f"group_{slug}_n_items_total"] = int(len(group_rows))

            out.update(_boot(f"group_{slug}_pairwise_auc", group_pair))

            out.update(_boot(f"group_{slug}_mean_context_tokens", group_ctx))

        # Paired deltas across matched triplets (bootstrap unit: base_id).
        by_base: Dict[str, Dict[str, Dict[str, float]]] = {}
        for r in scored:
            base_id = str(r["base_id"])
            grp = str(r["group"])
            by_base.setdefault(base_id, {})[grp] = {
                "meanpool_correct": float(r["meanpool_correct"]),
                "mean_gap": float(r["mean_gap"]),
                "pairwise_auc": float(r["pairwise_auc"]),
                "context_tokens": float(r["context_tokens"]),
            }

        def _paired_delta(group_name: str) -> Tuple[List[float], List[float], List[float], List[float]]:
            d_acc: List[float] = []
            d_gap: List[float] = []
            d_auc: List[float] = []
            d_ctx: List[float] = []
            for base_id, gmap in by_base.items():
                if "main" not in gmap or group_name not in gmap:
                    continue
                main = gmap["main"]
                other = gmap[group_name]
                d_acc.append(main["meanpool_correct"] - other["meanpool_correct"])
                d_gap.append(main["mean_gap"] - other["mean_gap"])
                d_auc.append(main["pairwise_auc"] - other["pairwise_auc"])
                d_ctx.append(main["context_tokens"] - other["context_tokens"])
            return d_acc, d_gap, d_auc, d_ctx

        for grp in ("ablate_relevant", "ablate_irrelevant"):
            d_acc, d_gap, d_auc, d_ctx = _paired_delta(grp)
            slug = "".join(ch if ch.isalnum() else "_" for ch in grp.strip().lower())
            out.update(_boot(f"paired_delta_{slug}_accuracy", d_acc))
            out[f"paired_delta_{slug}_n_bases"] = int(len(d_acc))

            out.update(_boot(f"paired_delta_{slug}_mean_gap", d_gap))

            out.update(_boot(f"paired_delta_{slug}_pairwise_auc", d_auc))

            out.update(_boot(f"paired_delta_{slug}_context_tokens", d_ctx))

    return out
