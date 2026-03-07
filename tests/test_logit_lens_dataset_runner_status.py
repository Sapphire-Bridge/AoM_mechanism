from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from aom.io import write_jsonl
from aom.run_manifest import read_run_manifest, validate_run_manifest


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


def test_logit_lens_dataset_all_failures_sets_fail_status(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    # Force a non-"single token" failure: both labels map to the same token id.
    write_jsonl(
        [
            {
                "pair_id": "bank-lex-0",
                "target": "bank",
                "target_occurrence": 0,
                "a": {"prompt": "I sat by the bank and watched the", "expected_label": "river"},
                "b": {"prompt": "I went to the bank to discuss the", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" river"]},
                "metadata": {"type": "lexical"},
            }
        ],
        disamb_path,
    )

    manifest_path = tmp_path / "logit_lens.manifest.json"

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_logit_lens_dataset.py"),
        "--model_name_or_path",
        str(model_dir),
        "--device",
        "cpu",
        "--local_files_only",
        "--disamb_path",
        str(disamb_path),
        "--sides",
        "a",
        "--bootstrap_n",
        "10",
        "--bootstrap_seed",
        "0",
        "--ci",
        "0.95",
        "--png_path",
        "",
        "--table_csv_path",
        str(tmp_path / "table.csv"),
        "--manifest_path",
        str(manifest_path),
        "--error_policy",
        "warn_skip",
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert proc.returncode != 0

    assert manifest_path.exists()
    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "FAIL"
    assert manifest.get("device_backend") == "cpu"
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("determinism_requested") == "best_effort"
    assert repro.get("seed") == 0
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    dm = datasets.get("disamb")
    assert isinstance(dm, dict)
    assert dm.get("n_rows_total") == 1
    assert dm.get("n_rows_valid") == 1
    assert dm.get("n_rows_invalid") == 0
    assert manifest.get("run_summary", {}).get("succeeded") == 0
    assert manifest.get("run_summary", {}).get("failed") == 1
    assert any(t.get("type") == "ValueError" for t in manifest.get("run_summary", {}).get("top_failure_types", []))


def test_logit_lens_dataset_mixed_failures_sets_warn_status(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    write_jsonl(
        [
            {
                "pair_id": "bank-ok-0",
                "target": "bank",
                "target_occurrence": 0,
                "a": {"prompt": "I sat by the bank and watched the", "expected_label": "river"},
                "b": {"prompt": "I went to the bank to discuss the", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" loan"]},
                "metadata": {"type": "lexical"},
            },
            {
                "pair_id": "bank-fail-1",
                "target": "bank",
                "target_occurrence": 0,
                "a": {"prompt": "I sat by the bank and watched the", "expected_label": "river"},
                "b": {"prompt": "I went to the bank to discuss the", "expected_label": "loan"},
                "choices": {"river": [" river"], "loan": [" river"]},
                "metadata": {"type": "lexical"},
            },
        ],
        disamb_path,
    )

    manifest_path = tmp_path / "logit_lens.manifest.json"

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_logit_lens_dataset.py"),
        "--model_name_or_path",
        str(model_dir),
        "--device",
        "cpu",
        "--local_files_only",
        "--disamb_path",
        str(disamb_path),
        "--sides",
        "a",
        "--bootstrap_n",
        "10",
        "--bootstrap_seed",
        "0",
        "--ci",
        "0.95",
        "--png_path",
        "",
        "--table_csv_path",
        str(tmp_path / "table.csv"),
        "--manifest_path",
        str(manifest_path),
        "--error_policy",
        "warn_skip",
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)
    assert proc.returncode == 0

    assert manifest_path.exists()
    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "WARN"
    assert manifest.get("device_backend") == "cpu"
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("determinism_requested") == "best_effort"
    assert repro.get("seed") == 0
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    dm = datasets.get("disamb")
    assert isinstance(dm, dict)
    assert dm.get("n_rows_total") == 2
    assert dm.get("n_rows_valid") == 2
    assert dm.get("n_rows_invalid") == 0
    assert manifest.get("run_summary", {}).get("attempted") == 2
    assert manifest.get("run_summary", {}).get("succeeded") == 1
    assert manifest.get("run_summary", {}).get("failed") == 1
    assert any(t.get("type") == "ValueError" for t in manifest.get("run_summary", {}).get("top_failure_types", []))
