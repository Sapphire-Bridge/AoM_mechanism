from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from aom.utils import write_jsonl

_NAN_SENTINEL = object()


def _strip_nondeterministic_fields(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if str(k).endswith("_at_utc"):
            continue
        if str(k).endswith("wall_time_sec"):
            continue
        if str(k).startswith("eval_time_") and str(k).endswith("_sec"):
            continue
        out[k] = v
    return out


def _canonicalize_nans(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return _NAN_SENTINEL
    if isinstance(obj, dict):
        return {k: _canonicalize_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_canonicalize_nans(v) for v in obj)
    return obj


def _make_wordlevel_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    # Local import so the rest of the suite still runs if tokenizers changes.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


def test_run_once_deterministic(tmp_path: Path):
    # Minimal local model+tokenizer so AutoModel/AutoTokenizer can load offline.
    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        "I": 3,
        "sat": 4,
        "by": 5,
        "the": 6,
        "bank": 7,
        "and": 8,
        "watched": 9,
        "went": 10,
        "to": 11,
        "discuss": 12,
        "river": 13,
        "loan": 14,
        "Alice": 15,
        "did": 16,
        "not": 17,
        "go": 18,
        "store": 19,
        "Therefore": 20,
        "John": 21,
        "died": 22,
        "Later": 23,
        "was": 24,
        "remembered": 25,
        "walked": 26,
    }

    tokenizer = _make_wordlevel_tokenizer(vocab)
    tokenizer.save_pretrained(tmp_path)

    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=32,
        vocab_size=len(vocab),
        n_positions=64,
        pad_token_id=vocab["[PAD]"],
        eos_token_id=vocab["[EOS]"],
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.save_pretrained(tmp_path)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    cf_path = tmp_path / "counterfactual.jsonl"
    coh_path = tmp_path / "coherence.jsonl"

    write_jsonl(
        [
            {
                "pair_id": "bank-lex-0",
                "target": "bank",
                "target_occurrence": 0,
                "a": {"prompt": "I sat by the bank and watched the", "expected_label": "river"},
                "b": {"prompt": "I went to the bank to discuss the", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" loan"]},
                "metadata": {"type": "lexical"},
            }
        ],
        disamb_path,
    )
    write_jsonl(
        [
            {
                "item_id": "negation-0",
                "base": {"prompt": "Alice did go to the store Therefore Alice", "expected_label": "did"},
                "cf": {"prompt": "Alice did not go to the store Therefore Alice", "expected_label": "not"},
                "choices": {"did": [" did"], "not": [" not"]},
                "intervention_type": "negation",
            }
        ],
        cf_path,
    )
    write_jsonl(
        [
            {
                "item_id": "entity-0",
                "context": "John died Later John",
                "valid_continuations": [" was remembered"],
                "invalid_continuations": [" walked"],
                "constraint_type": "entity_state",
            }
        ],
        coh_path,
    )

    from aom_eval import run_once

    args = argparse.Namespace(
        model_name_or_path=str(tmp_path),
        models=None,
        torch_dtype=None,
        attn_implementation="eager",
        device_map=None,
        local_files_only=True,
        device="cpu",
        seed=0,
        sweep_seeds=None,
        disamb_path=str(disamb_path),
        cf_path=str(cf_path),
        coh_path=str(coh_path),
        run_patching=True,
        patch_layers="0",
        patch_allow_token_id_mismatch=False,
        no_length_norm=False,
        bootstrap_n=200,
        bootstrap_seed=123,
        ci=0.95,
        csv_path="",
    )

    r1 = run_once(args, seed=0)
    r2 = run_once(args, seed=0)
    assert _canonicalize_nans(_strip_nondeterministic_fields(r1)) == _canonicalize_nans(_strip_nondeterministic_fields(r2))

    for r in (r1, r2):
        started = datetime.fromisoformat(str(r["started_at_utc"]))
        ended = datetime.fromisoformat(str(r["ended_at_utc"]))
        wall = float(r["wall_time_sec"])
        eval_wall = float(r["eval_wall_time_sec"])
        assert ended >= started
        assert wall > 0.0
        assert eval_wall > 0.0
        assert wall >= eval_wall
