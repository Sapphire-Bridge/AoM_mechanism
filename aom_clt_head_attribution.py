from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from aom.data.loaders import load_disamb_pairs
from aom.interventions.clt_adapter import CLTInputTransform
from aom.interventions.clt_loader import load_clt
from aom.mechanistic.backends.hooks import HookSpec
from aom.mechanistic.backends.transformer_lens import TransformerLensBackend, TransformerLensHookableBackend
from aom.metrics.clt_cpt import split_pairs_deterministic
from aom.repro import collect_versions
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.token_spans import token_span_for_substring
from aom.utils import bootstrap_ci, get_best_device


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(k) for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({str(k): r.get(k) for k in keys})


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _logmeanexp(vals: Sequence[float]) -> float:
    if not vals:
        return float("-inf")
    t = torch.tensor(list(vals), dtype=torch.float64)
    m = float(torch.max(t).item())
    return float(m + torch.log(torch.mean(torch.exp(t - m))).item())


def _margin(scores: Mapping[str, float], *, expected: str) -> float:
    exp = float(scores[str(expected)])
    others = [float(v) for k, v in scores.items() if str(k) != str(expected)]
    if not others:
        return float("nan")
    return float(exp - max(others))


def _parse_feature_ids_from_summary(path: Path, *, layer: int, top_n: int) -> List[int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in --topk_summary_path, got {type(obj).__name__}")
    ranking = obj.get("ranking_by_layer", None)
    if not isinstance(ranking, dict):
        raise ValueError("top-k summary missing ranking_by_layer")
    entry = ranking.get(str(int(layer)), None)
    if not isinstance(entry, dict):
        raise ValueError(f"top-k summary missing ranking_by_layer[{int(layer)!r}]")
    top = entry.get("top_features", None)
    if not isinstance(top, list):
        raise ValueError(f"top-k summary missing top_features for layer {int(layer)}")
    out: List[int] = []
    for row in top:
        if not isinstance(row, dict):
            continue
        fid = row.get("feature_id", None)
        if isinstance(fid, int) and int(fid) not in out:
            out.append(int(fid))
        if len(out) >= int(top_n):
            break
    return out


def _score_choices_with_hooks(
    *,
    hook_backend: TransformerLensHookableBackend,
    tokenizer,
    prompt: str,
    choices: Mapping[str, Sequence[str]],
    hooks: Sequence[HookSpec],
    normalize_by_length: bool = True,
) -> Dict[str, float]:
    model_device = next(hook_backend.model.parameters()).device
    prompt_ids = tokenizer(str(prompt), return_tensors="pt", add_special_tokens=False)["input_ids"].to(model_device)
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        vals: List[float] = []
        for cont in continuations:
            cont_ids = tokenizer(str(cont), return_tensors="pt", add_special_tokens=False)["input_ids"].to(model_device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = hook_backend.run_with_hooks(
                prompt="",
                hooks=list(hooks),
                tokens=full_ids,
                prepend_bos=False,
            )
            p = int(prompt_ids.size(1))
            c = int(cont_ids.size(1))
            logits_slice = logits[:, p - 1 : p + c - 1, :].to(dtype=torch.float32)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
            lp = gathered.mean(dim=1) if bool(normalize_by_length) else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return scores


def _group_heads_by_layer(heads: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for layer, head in heads:
        out.setdefault(int(layer), []).append(int(head))
    for layer in list(out.keys()):
        out[int(layer)] = sorted(set(int(h) for h in out[int(layer)]))
    return out


def _make_head_zero_hooks(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    target_positions: Sequence[int],
    ablate_positions: str,
) -> List[HookSpec]:
    hooks: List[HookSpec] = []
    for layer, heads in sorted(heads_by_layer.items(), key=lambda kv: int(kv[0])):
        hs = [int(h) for h in heads]
        if not hs:
            continue
        pos = [int(p) for p in target_positions]

        def _make_fn(head_ids: List[int], pos_ids: List[int], mode: str):
            def _fn(act: torch.Tensor, _hook: Any) -> torch.Tensor:
                if act.ndim != 4:
                    raise ValueError(f"Expected head_result rank-4 [batch,pos,head,d], got {tuple(act.shape)}")
                out = act.clone()
                valid_heads = [h for h in head_ids if 0 <= int(h) < int(act.size(2))]
                if not valid_heads:
                    return out
                if str(mode) == "all":
                    out[:, :, valid_heads, :] = 0.0
                    return out
                valid_pos = [p for p in pos_ids if 0 <= int(p) < int(act.size(1))]
                if not valid_pos:
                    return out
                out[:, valid_pos, valid_heads, :] = 0.0
                return out

            return _fn

        hooks.append(
            HookSpec(
                name=f"head_result.{int(layer)}",
                fn=_make_fn(valid_heads := hs, pos_ids := pos, mode=str(ablate_positions)),
            )
        )
    return hooks


def _sample_random_heads(
    *,
    all_heads: Sequence[Tuple[int, int]],
    excluded: Sequence[Tuple[int, int]],
    k: int,
    seed: int,
) -> List[Tuple[int, int]]:
    pool = [x for x in all_heads if x not in set(excluded)]
    if int(k) <= 0:
        return []
    if len(pool) <= int(k):
        return list(pool)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(len(pool), generator=g)[: int(k)].tolist()
    return [pool[int(i)] for i in perm]


def _pair_context_effect(
    *,
    item,
    hook_backend: TransformerLensHookableBackend,
    tokenizer,
    hooks_a: Sequence[HookSpec],
    hooks_b: Sequence[HookSpec],
    normalize_by_length: bool = True,
) -> float:
    scores_a = _score_choices_with_hooks(
        hook_backend=hook_backend,
        tokenizer=tokenizer,
        prompt=str(item.a.prompt),
        choices=item.choices,
        hooks=hooks_a,
        normalize_by_length=bool(normalize_by_length),
    )
    scores_b = _score_choices_with_hooks(
        hook_backend=hook_backend,
        tokenizer=tokenizer,
        prompt=str(item.b.prompt),
        choices=item.choices,
        hooks=hooks_b,
        normalize_by_length=bool(normalize_by_length),
    )
    ma = _margin(scores_a, expected=str(item.a.expected_label))
    mb = _margin(scores_b, expected=str(item.b.expected_label))
    return float((ma + mb) / 2.0)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="CLT feature-space head attribution + top-vs-random ablation validation.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--clt_repo", type=str, required=True)
    p.add_argument("--clt_layer", type=int, default=4)
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_run_name", type=str, default=None)
    p.add_argument("--clt_l0_target", type=int, default=None)
    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--feature_ids", type=str, default="")
    p.add_argument("--topk_summary_path", type=str, default="")
    p.add_argument("--top_n_features", type=int, default=50)
    p.add_argument("--head_layers", type=str, default="1,2,3")
    p.add_argument("--top_h", type=int, default=5)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--frac_selection", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=0, help="0 = all")
    p.add_argument("--ablate_positions", type=str, default="target", choices=["target", "all"])
    p.add_argument("--qk_patterns", action="store_true")
    p.add_argument("--normalize_by_length", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--out_head_scores_csv", type=str, default=str(root / "results" / "clt_head_scores.csv"))
    p.add_argument("--out_ablation_rows_csv", type=str, default=str(root / "results" / "clt_head_ablation_rows.csv"))
    p.add_argument(
        "--out_ablation_summary_csv",
        type=str,
        default=str(root / "results" / "clt_head_ablation_summary.csv"),
    )
    p.add_argument("--out_qk_csv", type=str, default=str(root / "results" / "clt_head_qk_patterns.csv"))
    p.add_argument("--manifest_path", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.smoke):
        args.top_n_features = min(int(args.top_n_features), 16)
        args.top_h = min(int(args.top_h), 3)
        args.max_pairs = 8 if int(args.max_pairs) <= 0 else min(int(args.max_pairs), 8)
        args.bootstrap_n = min(int(args.bootstrap_n), 100)

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    items = list(load_disamb_pairs(str(args.disamb_path)))
    if int(args.max_pairs) > 0:
        items = items[: int(args.max_pairs)]
    if not items:
        raise ValueError("DISAMB dataset is empty")

    pair_ids_all = sorted({str(it.pair_id) for it in items})
    pair_ids_s, pair_ids_e = split_pairs_deterministic(
        pair_ids_all,
        seed=int(args.split_seed),
        frac_selection=float(args.frac_selection),
    )

    backend = TransformerLensBackend()
    loaded = backend.load(
        model_name_or_path=str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation="eager",
    )
    # Required for head_result.* captures in TransformerLens on Gemma-family models.
    if hasattr(loaded.model, "set_use_attn_result"):
        try:
            loaded.model.set_use_attn_result(True)
        except Exception:
            pass
    hook_backend = TransformerLensHookableBackend(loaded.model, loaded.tokenizer, default_prepend_bos=False)

    clt, _meta = load_clt(
        str(args.clt_repo),
        layer=int(args.clt_layer),
        width=str(args.clt_width),
        run_name=args.clt_run_name,
        l0_target=args.clt_l0_target,
        device=str(device),
        dtype=str(args.clt_dtype),
        local_files_only=bool(args.local_files_only),
    )
    transform = CLTInputTransform(scale=float(args.clt_scale))
    w_enc = getattr(clt, "W_enc", None)
    if not isinstance(w_enc, torch.Tensor) or w_enc.ndim != 2:
        raise ValueError("CLT must expose W_enc tensor with shape (d_model, d_latent)")
    d_latent = int(getattr(clt, "d_latent", int(w_enc.size(1))))

    if str(args.feature_ids).strip():
        feature_ids = _parse_int_csv(str(args.feature_ids))
    elif str(args.topk_summary_path).strip():
        feature_ids = _parse_feature_ids_from_summary(
            Path(str(args.topk_summary_path)),
            layer=int(args.clt_layer),
            top_n=int(args.top_n_features),
        )
    else:
        raise ValueError("Provide --feature_ids or --topk_summary_path")
    feature_ids = sorted({int(fid) for fid in feature_ids if 0 <= int(fid) < int(d_latent)})
    if not feature_ids:
        raise ValueError("No in-range feature IDs resolved")

    head_layers = _parse_int_csv(str(args.head_layers))
    if not head_layers:
        raise ValueError("--head_layers must be non-empty")

    clt_param = next(clt.parameters(), None) if isinstance(clt, torch.nn.Module) else None
    clt_device = clt_param.device if clt_param is not None else torch.device("cpu")
    clt_dtype = clt_param.dtype if clt_param is not None else torch.float32
    w_sel = w_enc[:, feature_ids].detach().to(dtype=torch.float32)

    head_values: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    eval_cases: List[Dict[str, Any]] = []
    qk_rows: List[Dict[str, Any]] = []
    all_heads_seen: set[Tuple[int, int]] = set()

    for it in items:
        span_a, _ = token_span_for_substring(loaded.tokenizer, it.a.prompt, it.target, it.target_occurrence)
        span_b, _ = token_span_for_substring(loaded.tokenizer, it.b.prompt, it.target, it.target_occurrence)
        if not span_a or not span_b:
            continue
        capture = [f"resid_pre.{int(args.clt_layer)}"] + [f"head_result.{int(l)}" for l in head_layers]
        if bool(args.qk_patterns):
            capture += [f"pattern.{int(l)}" for l in head_layers]

        run_a = hook_backend.run_with_cache(str(it.a.prompt), capture=capture, prepend_bos=False)
        run_b = hook_backend.run_with_cache(str(it.b.prompt), capture=capture, prepend_bos=False)
        cache_a = dict(run_a.cache)
        cache_b = dict(run_b.cache)

        resid_a = cache_a.get(f"resid_pre.{int(args.clt_layer)}", None)
        resid_b = cache_b.get(f"resid_pre.{int(args.clt_layer)}", None)
        if not (isinstance(resid_a, torch.Tensor) and isinstance(resid_b, torch.Tensor)):
            continue
        if resid_a.ndim != 3 or resid_b.ndim != 3:
            continue

        lat_a = clt.encode(transform.forward(resid_a.to(device=clt_device, dtype=clt_dtype)))
        lat_b = clt.encode(transform.forward(resid_b.to(device=clt_device, dtype=clt_dtype)))
        mean_feat_a = float(lat_a[0, span_a, :][:, feature_ids].mean().item())
        mean_feat_b = float(lat_b[0, span_b, :][:, feature_ids].mean().item())

        per_pair_scores: Dict[Tuple[int, int], float] = {}
        for layer in head_layers:
            key = f"head_result.{int(layer)}"
            hr_a = cache_a.get(key, None)
            hr_b = cache_b.get(key, None)
            if not (isinstance(hr_a, torch.Tensor) and isinstance(hr_b, torch.Tensor)):
                continue
            if hr_a.ndim != 4 or hr_b.ndim != 4:
                continue
            ta = hr_a[0, span_a, :, :].to(dtype=torch.float32)
            tb = hr_b[0, span_b, :, :].to(dtype=torch.float32)
            contrib_a = torch.einsum("shd,df->shf", ta, w_sel.to(device=ta.device, dtype=ta.dtype)).abs().mean(dim=(0, 2))
            contrib_b = torch.einsum("shd,df->shf", tb, w_sel.to(device=tb.device, dtype=tb.dtype)).abs().mean(dim=(0, 2))
            n_heads = int(min(contrib_a.numel(), contrib_b.numel()))
            for h in range(n_heads):
                all_heads_seen.add((int(layer), int(h)))
                per_pair_scores[(int(layer), int(h))] = float(abs(float(contrib_a[h].item()) - float(contrib_b[h].item())))

        split = "S" if str(it.pair_id) in pair_ids_s else "E"
        for key, val in per_pair_scores.items():
            head_values.setdefault((int(key[0]), int(key[1])), {"S": [], "E": []})
            head_values[(int(key[0]), int(key[1]))][str(split)].append(float(val))

        eval_cases.append(
            {
                "pair_id": str(it.pair_id),
                "item": it,
                "span_a": [int(x) for x in span_a],
                "span_b": [int(x) for x in span_b],
                "mean_feat_a": float(mean_feat_a),
                "mean_feat_b": float(mean_feat_b),
                "split": str(split),
            }
        )

    if not head_values:
        raise ValueError("No head attribution scores computed; check model/backend compatibility.")

    head_score_rows: List[Dict[str, Any]] = []
    for (layer, head), vals in sorted(head_values.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s_vals = list(vals.get("S", []))
        e_vals = list(vals.get("E", []))
        mean_s = float(sum(s_vals) / len(s_vals)) if s_vals else float("nan")
        mean_e = float(sum(e_vals) / len(e_vals)) if e_vals else float("nan")
        head_score_rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "score_selection_mean": float(mean_s),
                "score_selection_n": int(len(s_vals)),
                "score_eval_mean": float(mean_e),
                "score_eval_n": int(len(e_vals)),
            }
        )
    head_score_rows.sort(
        key=lambda r: float(r["score_selection_mean"]) if float(r["score_selection_mean"]) == float(r["score_selection_mean"]) else -1.0,
        reverse=True,
    )
    for rank, row in enumerate(head_score_rows):
        row["rank_selection"] = int(rank + 1)

    top_heads = [(int(r["layer"]), int(r["head"])) for r in head_score_rows[: max(1, int(args.top_h))]]
    all_heads = sorted(all_heads_seen)

    ablation_rows: List[Dict[str, Any]] = []
    eval_only = [x for x in eval_cases if str(x["split"]) == "E"]
    for case in eval_only:
        item = case["item"]
        base_effect = _pair_context_effect(
            item=item,
            hook_backend=hook_backend,
            tokenizer=loaded.tokenizer,
            hooks_a=[],
            hooks_b=[],
            normalize_by_length=bool(args.normalize_by_length),
        )

        top_by_layer = _group_heads_by_layer(top_heads)
        top_hooks_a = _make_head_zero_hooks(
            heads_by_layer=top_by_layer,
            target_positions=case["span_a"],
            ablate_positions=str(args.ablate_positions),
        )
        top_hooks_b = _make_head_zero_hooks(
            heads_by_layer=top_by_layer,
            target_positions=case["span_b"],
            ablate_positions=str(args.ablate_positions),
        )
        top_effect = _pair_context_effect(
            item=item,
            hook_backend=hook_backend,
            tokenizer=loaded.tokenizer,
            hooks_a=top_hooks_a,
            hooks_b=top_hooks_b,
            normalize_by_length=bool(args.normalize_by_length),
        )
        ablation_rows.append(
            {
                "pair_id": str(case["pair_id"]),
                "arm": "topk",
                "baseline_context_effect": float(base_effect),
                "ablated_context_effect": float(top_effect),
                "reduction": float(base_effect - top_effect),
                "n_heads": int(len(top_heads)),
            }
        )

        rand_heads = _sample_random_heads(
            all_heads=all_heads,
            excluded=top_heads,
            k=int(len(top_heads)),
            seed=int(args.seed) + int(abs(hash(str(case["pair_id"]))) % 10_000_000),
        )
        rand_by_layer = _group_heads_by_layer(rand_heads)
        rand_hooks_a = _make_head_zero_hooks(
            heads_by_layer=rand_by_layer,
            target_positions=case["span_a"],
            ablate_positions=str(args.ablate_positions),
        )
        rand_hooks_b = _make_head_zero_hooks(
            heads_by_layer=rand_by_layer,
            target_positions=case["span_b"],
            ablate_positions=str(args.ablate_positions),
        )
        rand_effect = _pair_context_effect(
            item=item,
            hook_backend=hook_backend,
            tokenizer=loaded.tokenizer,
            hooks_a=rand_hooks_a,
            hooks_b=rand_hooks_b,
            normalize_by_length=bool(args.normalize_by_length),
        )
        ablation_rows.append(
            {
                "pair_id": str(case["pair_id"]),
                "arm": "randomk",
                "baseline_context_effect": float(base_effect),
                "ablated_context_effect": float(rand_effect),
                "reduction": float(base_effect - rand_effect),
                "n_heads": int(len(rand_heads)),
            }
        )

    ablation_summary_rows: List[Dict[str, Any]] = []
    for arm in ("topk", "randomk"):
        vals = [float(r["reduction"]) for r in ablation_rows if str(r["arm"]) == arm]
        mean_v, lo, hi = bootstrap_ci(
            vals,
            n_bootstrap=int(args.bootstrap_n),
            ci=float(args.ci),
            seed=int(args.bootstrap_seed),
        )
        ablation_summary_rows.append(
            {
                "arm": str(arm),
                "mean_reduction": float(mean_v),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n": int(len(vals)),
            }
        )
    by_pair_arm: Dict[Tuple[str, str], float] = {}
    for r in ablation_rows:
        by_pair_arm[(str(r["pair_id"]), str(r["arm"]))] = float(r["reduction"])
    diffs = [
        float(by_pair_arm[(pid, "topk")] - by_pair_arm[(pid, "randomk")])
        for pid in sorted({k[0] for k in by_pair_arm.keys()})
        if (pid, "topk") in by_pair_arm and (pid, "randomk") in by_pair_arm
    ]
    if diffs:
        mean_d, lo_d, hi_d = bootstrap_ci(
            diffs,
            n_bootstrap=int(args.bootstrap_n),
            ci=float(args.ci),
            seed=int(args.bootstrap_seed),
        )
        ablation_summary_rows.append(
            {
                "arm": "topk_minus_randomk",
                "mean_reduction": float(mean_d),
                "ci_low": float(lo_d),
                "ci_high": float(hi_d),
                "n": int(len(diffs)),
            }
        )

    if bool(args.qk_patterns):
        for case in eval_only:
            item = case["item"]
            for side_name, prompt, span in (("a", str(item.a.prompt), case["span_a"]), ("b", str(item.b.prompt), case["span_b"])):
                capture = [f"pattern.{int(l)}" for l in head_layers]
                run = hook_backend.run_with_cache(prompt, capture=capture, prepend_bos=False)
                cache = dict(run.cache)
                for layer, head in top_heads:
                    patt = cache.get(f"pattern.{int(layer)}", None)
                    if not isinstance(patt, torch.Tensor) or patt.ndim != 4:
                        continue
                    if int(head) < 0 or int(head) >= int(patt.size(1)):
                        continue
                    valid_q = [int(q) for q in span if 0 <= int(q) < int(patt.size(2))]
                    if not valid_q:
                        continue
                    target_set = set(valid_q)
                    context_idx = [i for i in range(int(patt.size(3))) if i not in target_set]
                    if not context_idx:
                        continue
                    mass = patt[0, int(head), valid_q, :][:, context_idx].sum(dim=-1).mean()
                    qk_rows.append(
                        {
                            "pair_id": str(case["pair_id"]),
                            "side": str(side_name),
                            "layer": int(layer),
                            "head": int(head),
                            "target_to_context_mass": float(mass.item()),
                        }
                    )

    out_head_scores_csv = Path(str(args.out_head_scores_csv))
    out_ablation_rows_csv = Path(str(args.out_ablation_rows_csv))
    out_ablation_summary_csv = Path(str(args.out_ablation_summary_csv))
    out_qk_csv = Path(str(args.out_qk_csv))
    manifest_path = (
        Path(str(args.manifest_path))
        if str(args.manifest_path).strip()
        else out_ablation_summary_csv.with_suffix(".manifest.json")
    )

    _write_csv(out_head_scores_csv, head_score_rows)
    _write_csv(out_ablation_rows_csv, ablation_rows)
    _write_csv(out_ablation_summary_csv, ablation_summary_rows)
    _write_csv(out_qk_csv, qk_rows)

    summary_sha = _sha256_file(out_ablation_summary_csv) if out_ablation_summary_csv.exists() else ""
    result_row = {
        "model_name_or_path": str(args.model_name_or_path),
        "n_features": int(len(feature_ids)),
        "n_heads_ranked": int(len(head_score_rows)),
        "top_h": int(len(top_heads)),
        "n_eval_pairs": int(len(eval_only)),
        "out_head_scores_csv": str(out_head_scores_csv),
        "out_ablation_rows_csv": str(out_ablation_rows_csv),
        "out_ablation_summary_csv": str(out_ablation_summary_csv),
        "out_qk_csv": str(out_qk_csv),
        "ablation_summary_sha256": str(summary_sha),
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path),
        csv_path=str(out_ablation_summary_csv),
        csv_sha256=str(summary_sha),
        csv_n_rows=int(len(ablation_summary_rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(eval_only)),
        "succeeded": int(len(eval_only)),
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
    manifest["versions"] = collect_versions()
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote head score CSV: {out_head_scores_csv}", flush=True)
    print(f"Wrote ablation rows CSV: {out_ablation_rows_csv}", flush=True)
    print(f"Wrote ablation summary CSV: {out_ablation_summary_csv}", flush=True)
    if bool(args.qk_patterns):
        print(f"Wrote QK pattern CSV: {out_qk_csv}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
