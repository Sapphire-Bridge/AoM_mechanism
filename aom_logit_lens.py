from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from aom.data.loaders import load_disamb_pairs
from aom.mechanistic.logit_lens import compute_logit_lens_trace, encode_single_token_id, select_single_token_continuation
from aom.models.loader import load_causal_lm
from aom.utils import get_best_device, set_seed


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
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


def _pick_prompt_and_targets_from_disamb(
    *,
    disamb_path: str,
    pair_id: str,
    side: str,
    label_a: Optional[str],
    label_b: Optional[str],
    tokenizer,
    target_a: Optional[str],
    target_b: Optional[str],
) -> tuple[str, str, int, str, int, str, str, str]:
    items = load_disamb_pairs(disamb_path)
    found = next((it for it in items if it.pair_id == pair_id), None)
    if found is None:
        raise ValueError(f"pair_id {pair_id!r} not found in {disamb_path}")

    if side == "a":
        prompt = found.a.prompt
        expected = found.a.expected_label
    elif side == "b":
        prompt = found.b.prompt
        expected = found.b.expected_label
    else:
        raise ValueError("side must be 'a' or 'b'")

    if target_a is not None and target_b is not None:
        tok_a = encode_single_token_id(tokenizer, target_a)
        tok_b = encode_single_token_id(tokenizer, target_b)
        label_a_out = label_a or ""
        label_b_out = label_b or ""
        return prompt, target_a, tok_a, target_b, tok_b, found.pair_id, side, f"{label_a_out}:{label_b_out}"

    la = label_a or expected
    other_labels = [k for k in found.choices.keys() if str(k) != str(la)]
    if not other_labels:
        raise ValueError(f"No alternative label found in choices for label_a={la!r}")
    lb = label_b or str(sorted(other_labels)[0])

    ca, tok_a = select_single_token_continuation(tokenizer, list(found.choices[str(la)]))
    cb, tok_b = select_single_token_continuation(tokenizer, list(found.choices[str(lb)]))
    return prompt, ca, tok_a, cb, tok_b, found.pair_id, side, f"{la}:{lb}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--target_a", type=str, default="")
    p.add_argument("--target_b", type=str, default="")

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--pair_id", type=str, default="")
    p.add_argument("--side", type=str, default="a", choices=["a", "b"])
    p.add_argument("--label_a", type=str, default="")
    p.add_argument("--label_b", type=str, default="")

    p.add_argument("--position", type=int, default=-1, help="Token position whose residual is unembedded (default: -1).")
    p.add_argument("--lens", type=str, default="auto", choices=["auto", "raw", "final_norm"])

    p.add_argument("--csv_path", type=str, default=str(root / "results" / "logit_lens_trace.csv"))
    p.add_argument("--plot_path", type=str, default=str(root / "results" / "logit_lens_trace.png"))
    p.add_argument("--title", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])

    loaded = load_causal_lm(
        args.model_name_or_path,
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer

    pair_id = ""
    prompt = str(args.prompt or "").strip("\n")
    target_a = str(args.target_a or "")
    target_b = str(args.target_b or "")
    label_pair = ""
    side = ""

    if prompt:
        if not target_a or not target_b:
            raise ValueError("When using --prompt, you must also pass --target_a and --target_b.")
        tok_a = encode_single_token_id(tokenizer, target_a)
        tok_b = encode_single_token_id(tokenizer, target_b)
    else:
        if not args.pair_id:
            raise ValueError("Provide either --prompt (with --target_a/--target_b) or --pair_id (DISAMB JSONL).")
        prompt, target_a, tok_a, target_b, tok_b, pair_id, side, label_pair = _pick_prompt_and_targets_from_disamb(
            disamb_path=str(args.disamb_path),
            pair_id=str(args.pair_id),
            side=str(args.side),
            label_a=str(args.label_a) if args.label_a else None,
            label_b=str(args.label_b) if args.label_b else None,
            tokenizer=tokenizer,
            target_a=str(args.target_a) if args.target_a else None,
            target_b=str(args.target_b) if args.target_b else None,
        )

    trace = compute_logit_lens_trace(
        model,
        tokenizer,
        prompt,
        token_a_id=int(tok_a),
        token_b_id=int(tok_b),
        device=device,
        position=int(args.position),
        lens=str(args.lens),
        compute_logits=True,
    )

    rows: List[Dict[str, Any]] = []
    for p in trace.points:
        rows.append(
            {
                "model": str(args.model_name_or_path),
                "arch": str(trace.arch),
                "pair_id": str(pair_id),
                "side": str(side),
                "label_pair": str(label_pair),
                "prompt": str(prompt),
                "target_a": str(target_a),
                "target_b": str(target_b),
                "token_a_id": int(trace.token_a_id),
                "token_b_id": int(trace.token_b_id),
                "position": int(args.position),
                "lens": str(trace.lens),
                "apply_final_norm_intermediate": bool(trace.apply_final_norm_intermediate),
                "apply_final_norm_last": bool(trace.apply_final_norm_last),
                "final_logit_diff": float(trace.final_logit_diff),
                "state_index": int(p.state_index),
                "block_index": "" if p.block_index is None else int(p.block_index),
                "logit_a": float(p.logit_a),
                "logit_b": float(p.logit_b),
                "logit_diff": float(p.logit_diff),
                "delta_from_prev": float(p.delta_from_prev),
            }
        )

    if args.csv_path:
        _write_csv(rows, args.csv_path)
        print(f"Wrote CSV: {args.csv_path}", flush=True)

    if args.plot_path:
        import matplotlib.pyplot as plt

        xs = [int(p.state_index) for p in trace.points]
        ys = [float(p.logit_diff) for p in trace.points]

        pth = Path(args.plot_path)
        pth.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(xs, ys, marker="o", linewidth=2)
        plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Hidden-state index (0 = embeddings)")
        plt.ylabel(f"Logit diff ({target_a!r} - {target_b!r})")
        title = args.title.strip() if args.title else ""
        if not title:
            title = f"Logit lens trace: {args.model_name_or_path}"
            if pair_id:
                title += f" | {pair_id}:{side}"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(pth))
        print(f"Wrote plot: {args.plot_path}", flush=True)

    print(
        f"final_logit_diff={trace.final_logit_diff:.6f} "
        f"token_a_id={trace.token_a_id} token_b_id={trace.token_b_id} "
        f"lens={trace.lens} apply_final_norm_last={trace.apply_final_norm_last}",
        flush=True,
    )


if __name__ == "__main__":
    main()
