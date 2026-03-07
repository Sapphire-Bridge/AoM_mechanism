from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from aom.data.schemas import CoherenceItem
from aom.token_diff import divergence_span

from .base import ActivationPatchingProtocol, CaseSkip, ComparisonSpec, PatchingCase


@dataclass(frozen=True)
class COHPatchingConfig:
    max_total_len_delta: int = 20


def _base_id(item_id: str) -> str:
    item_id = str(item_id)
    return item_id.split("__", 1)[0] if "__" in item_id else item_id


def _pad_or_truncate(ids: torch.Tensor, *, target_len: int, pad_id: int) -> torch.Tensor:
    if ids.ndim != 1:
        raise ValueError("ids must be 1D")
    L = int(ids.numel())
    if L == int(target_len):
        return ids
    if L > int(target_len):
        return ids[: int(target_len)]
    pad = torch.full((int(target_len) - L,), int(pad_id), dtype=ids.dtype, device=ids.device)
    return torch.cat([ids, pad], dim=0)


class COHConstraintAblationProtocol(ActivationPatchingProtocol):
    """
    Constraint-ablation patching for AoM-COH.

    Receiver: main context
    Donor: pseudo-ablated context built by token-ID replacement (keeps sequence length aligned)
    Patch sites:
      - constraint_span: span that differs between main and ablate_relevant
      - irrelevant_span: span that differs between main and ablate_irrelevant (relevance control)
    Effect: coherence degradation (decrease in valid-vs-invalid margin), so we use effect_sign=-1.0.
    """

    name = "coh_constraint_ablation"

    def __init__(self, *, config: COHPatchingConfig | None = None):
        self.config = COHPatchingConfig() if config is None else config

    def primary_comparisons(self) -> Sequence[ComparisonSpec]:
        return (
            ComparisonSpec(
                name="condition_constraint_vs_irrelevant",
                stratum_key="condition",
                a_value="constraint_span",
                b_value="irrelevant_span",
            ),
        )

    def build_cases(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        items: Sequence[Any],
        device: torch.device,
    ) -> Tuple[List[PatchingCase], List[CaseSkip]]:
        by_base: Dict[str, Dict[str, CoherenceItem]] = {}
        skips: List[CaseSkip] = []
        for raw in items:
            if not isinstance(raw, CoherenceItem):
                skips.append(CaseSkip(case_id=str(getattr(raw, "item_id", "coh_item")), reason="type_mismatch"))
                continue
            it: CoherenceItem = raw
            base = _base_id(it.item_id)
            grp = str(getattr(it, "group", "main"))
            by_base.setdefault(base, {})[grp] = it

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        cases: List[PatchingCase] = []

        for base, gmap in by_base.items():
            if "main" not in gmap or "ablate_relevant" not in gmap or "ablate_irrelevant" not in gmap:
                skips.append(CaseSkip(case_id=str(base), reason="missing_triplet"))
                continue
            main = gmap["main"]
            ab_rel = gmap["ablate_relevant"]
            ab_irr = gmap["ablate_irrelevant"]

            main_text = str(main.context)
            rel_text = str(ab_rel.context)
            irr_text = str(ab_irr.context)

            main_ids = tokenizer(main_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            rel_ids = tokenizer(rel_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            irr_ids = tokenizer(irr_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

            span_rel = divergence_span(main_ids[0].tolist(), rel_ids[0].tolist())
            span_irr = divergence_span(main_ids[0].tolist(), irr_ids[0].tolist())
            if span_rel is None or span_irr is None:
                skips.append(CaseSkip(case_id=str(base), reason="no_divergence"))
                continue

            if abs(int(span_rel.a_len_total) - int(span_rel.b_len_total)) > int(self.config.max_total_len_delta):
                skips.append(CaseSkip(case_id=str(base), reason="total_len_delta_rel_gt_max"))
                continue
            if abs(int(span_irr.a_len_total) - int(span_irr.b_len_total)) > int(self.config.max_total_len_delta):
                skips.append(CaseSkip(case_id=str(base), reason="total_len_delta_irr_gt_max"))
                continue

            # Build pseudo-ablated donor ids aligned to main by replacing the divergent span token IDs.
            rel_recv_span = tuple(range(int(span_rel.a_start), int(span_rel.a_end)))
            irr_recv_span = tuple(range(int(span_irr.a_start), int(span_irr.a_end)))
            if not rel_recv_span or not irr_recv_span:
                skips.append(CaseSkip(case_id=str(base), reason="empty_span"))
                continue

            rel_repl = rel_ids[0, int(span_rel.b_start) : int(span_rel.b_end)].detach()
            irr_repl = irr_ids[0, int(span_irr.b_start) : int(span_irr.b_end)].detach()
            rel_repl = _pad_or_truncate(rel_repl, target_len=len(rel_recv_span), pad_id=int(pad_id))
            irr_repl = _pad_or_truncate(irr_repl, target_len=len(irr_recv_span), pad_id=int(pad_id))

            donor_rel_ids = main_ids.clone()
            donor_rel_ids[0, int(span_rel.a_start) : int(span_rel.a_end)] = rel_repl
            donor_irr_ids = main_ids.clone()
            donor_irr_ids[0, int(span_irr.a_start) : int(span_irr.a_end)] = irr_repl

            meta = getattr(main, "metadata", None)
            n_constraints = 1
            if isinstance(meta, dict):
                nc = meta.get("n_constraints", None)
                if isinstance(nc, int):
                    n_constraints = int(nc)

            choices = {"valid": list(main.valid_continuations), "invalid": list(main.invalid_continuations)}
            strata_common = {
                "constraint_type": str(main.constraint_type),
                "n_constraints": str(int(n_constraints)),
            }

            cases.append(
                PatchingCase(
                    case_id=str(base) + "__constraint_span",
                    receiver_prompt=main_text,
                    donor_prompt="<pseudo_ablate_relevant>",
                    receiver_ids=main_ids,
                    donor_ids=donor_rel_ids,
                    receiver_span=tuple(int(i) for i in rel_recv_span),
                    donor_span=tuple(int(i) for i in rel_recv_span),
                    choices=choices,
                    expected_label="valid",
                    strata={**strata_common, "condition": "constraint_span"},
                    label_aggregation="mean",
                    effect_sign=-1.0,
                )
            )
            cases.append(
                PatchingCase(
                    case_id=str(base) + "__irrelevant_span",
                    receiver_prompt=main_text,
                    donor_prompt="<pseudo_ablate_irrelevant>",
                    receiver_ids=main_ids,
                    donor_ids=donor_irr_ids,
                    receiver_span=tuple(int(i) for i in irr_recv_span),
                    donor_span=tuple(int(i) for i in irr_recv_span),
                    choices=choices,
                    expected_label="valid",
                    strata={**strata_common, "condition": "irrelevant_span"},
                    label_aggregation="mean",
                    effect_sign=-1.0,
                )
            )

        return cases, skips
