from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from aom.run_manifest import read_run_manifest, validate_run_manifest
from aom.utils import write_jsonl


def _make_wordlevel_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
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


def _write_local_tiny_gpt2(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

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
    tokenizer.save_pretrained(model_dir)

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
    model.save_pretrained(model_dir)


def _write_local_tiny_clt_bundle(clt_root: Path, *, layer: int = 0, width: str = "16k", run_name: str = "average_l0_71") -> None:
    run_dir = clt_root / f"layer_{int(layer)}" / f"width_{str(width)}" / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    d_in = 32
    d_latent = 32
    d_out = 32
    cfg = {
        "d_in": d_in,
        "d_latent": d_latent,
        "d_out": d_out,
        "encode_site": "resid_post",
        "decode_site": "resid_post",
        "writeback_site": "resid_post",
        "site_mode": "same_site_v1",
        "activation": "identity",
    }
    (run_dir / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    # Save in canonical orientation.
    W_enc = np.eye(d_in, d_latent, dtype=np.float32)
    W_dec = np.eye(d_latent, d_out, dtype=np.float32)
    b_enc = np.zeros((d_latent,), dtype=np.float32)
    b_dec = np.zeros((d_out,), dtype=np.float32)
    np.savez(run_dir / "params.npz", W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec)


def test_aom_eval_runs_with_clt_patching_and_writes_clt_fields(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    clt_root = tmp_path / "local_clt"
    _write_local_tiny_clt_bundle(clt_root, layer=0, width="16k", run_name="average_l0_71")

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

    results_dir = tmp_path / "results"
    run_name = "clt_integration"

    cmd = [
        sys.executable,
        str(root / "aom_eval.py"),
        "--models",
        str(model_dir),
        "--device",
        "cpu",
        "--local_files_only",
        "--results_dir",
        str(results_dir),
        "--run_name",
        run_name,
        "--seed",
        "0",
        "--bootstrap_n",
        "20",
        "--bootstrap_seed",
        "0",
        "--ci",
        "0.95",
        "--disamb_path",
        str(disamb_path),
        "--cf_path",
        str(cf_path),
        "--coh_path",
        str(coh_path),
        "--run_clt_patching",
        "--clt_repo",
        str(clt_root),
        "--clt_width",
        "16k",
        "--clt_run_name",
        "average_l0_71",
        "--clt_layers",
        "0",
        "--clt_scale",
        "1.0",
        "--clt_decode_strategy",
        "safe_2decode",
    ]
    subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)

    csv_path = results_dir / f"{run_name}.csv"
    manifest_path = results_dir / f"{run_name}.manifest.json"
    assert csv_path.exists()
    assert manifest_path.exists()

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert row["clt_repo"] == str(clt_root)
    assert row["clt_width"] == "16k"
    assert row["clt_layers"] == "0"
    assert row["clt_run_name"] == "average_l0_71"
    assert row["clt_site_mode"] == "same_site_v1"
    assert row["clt_encode_site"] == "resid_post"
    assert row["clt_decode_site"] == "resid_post"
    assert row["clt_writeback_site"] == "resid_post"
    assert "clt_cpt_mean_max_effect" in row
    assert "clt_cpt_mean_sham_max_effect" in row
    assert "clt_cpt_mean_identity_max_abs_effect" in row
    assert "clt_cpt_effect_layer_0" in row
    assert "clt_cpt_sham_effect_layer_0" in row
    assert "clt_cpt_identity_effect_layer_0" in row
    assert "clt_params_sha256_layer_0" in row
    assert len(str(row["clt_params_sha256_layer_0"])) == 64
    assert "clt_cfg_sha256_layer_0" in row
    assert len(str(row["clt_cfg_sha256_layer_0"])) == 64
    assert "clt_params_sha256_layers" in row and "0:" in str(row["clt_params_sha256_layers"])
    assert "clt_cfg_sha256_layers" in row and "0:" in str(row["clt_cfg_sha256_layers"])

    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") in {"PASS", "WARN"}
