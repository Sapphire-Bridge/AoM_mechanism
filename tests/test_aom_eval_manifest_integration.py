from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from aom.run_manifest import read_run_manifest, validate_run_manifest
from aom.utils import write_jsonl


def _strip_nondeterministic_row_fields(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in row.items():
        if k.endswith("_at_utc"):
            continue
        if k.endswith("wall_time_sec"):
            continue
        if k.startswith("eval_time_") and k.endswith("_sec"):
            continue
        if k in {"argv_redacted_json", "argv_sha256"}:
            continue
        out[k] = v
    return out


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


def test_aom_eval_writes_csv_and_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

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
    run_name = "integration"

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
        "10",
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
    ]
    subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)

    csv_path = results_dir / f"{run_name}.csv"
    manifest_path = results_dir / f"{run_name}.manifest.json"
    assert csv_path.exists()
    assert manifest_path.exists()

    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("manifest_version") == "1.0"
    assert manifest.get("run_status") == "PASS"
    assert manifest.get("device_backend") == "cpu"
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("seed") == 0
    assert repro.get("seeds") == [0]
    assert repro.get("determinism_requested") == "best_effort"
    assert isinstance(repro.get("determinism_enforced"), bool)
    versions = manifest.get("versions")
    assert isinstance(versions, dict)
    assert isinstance(versions.get("python"), str) and versions.get("python")
    assert isinstance(versions.get("torch"), str)
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    assert set(datasets.keys()) >= {"disamb", "cf", "coh"}
    for role in ("disamb", "cf", "coh"):
        dm = datasets.get(role)
        assert isinstance(dm, dict)
        assert dm.get("role") == role
        assert dm.get("n_rows_total") == 1
        assert dm.get("n_rows_valid") == 1
        assert dm.get("n_rows_invalid") == 0
        assert isinstance(dm.get("sha256"), str) and len(dm.get("sha256")) == 64
    assert manifest.get("csv_path") == str(csv_path)
    assert isinstance(manifest.get("csv_sha256"), str) and len(manifest.get("csv_sha256")) == 64
    assert manifest.get("csv_n_rows") == 1


def test_aom_eval_empty_datasets_write_nan_and_warn(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    cf_path = tmp_path / "counterfactual.jsonl"
    coh_path = tmp_path / "coherence.jsonl"
    write_jsonl([], disamb_path)
    write_jsonl([], cf_path)
    write_jsonl([], coh_path)

    results_dir = tmp_path / "results"
    run_name = "empty"

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
        "10",
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
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)
    assert "[WARN] Composite invalid" in proc.stderr

    csv_path = results_dir / f"{run_name}.csv"
    manifest_path = results_dir / f"{run_name}.manifest.json"
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert row["disamb_accuracy"].lower() == "nan"
    assert row["cf_shift_direction_accuracy"].lower() == "nan"
    assert row["coh_constraint_accuracy"].lower() == "nan"
    assert row["aom_composite"].lower() == "nan"
    assert row["aom_composite_valid"].lower() == "false"

    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "WARN"
    assert "invalid_results > 0" in (manifest.get("run_status_reasons") or [])
    assert manifest.get("device_backend") == "cpu"
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("determinism_requested") == "best_effort"
    assert repro.get("seed") == 0
    assert repro.get("seeds") == [0]
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    assert set(datasets.keys()) >= {"disamb", "cf", "coh"}
    for role in ("disamb", "cf", "coh"):
        dm = datasets.get(role)
        assert isinstance(dm, dict)
        assert dm.get("role") == role
        assert dm.get("n_rows_total") == 0
        assert dm.get("n_rows_valid") == 0
        assert dm.get("n_rows_invalid") == 0
        assert isinstance(dm.get("sha256"), str) and len(dm.get("sha256")) == 64


def test_aom_eval_empty_datasets_fail_strict_metrics(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    cf_path = tmp_path / "counterfactual.jsonl"
    coh_path = tmp_path / "coherence.jsonl"
    write_jsonl([], disamb_path)
    write_jsonl([], cf_path)
    write_jsonl([], coh_path)

    results_dir = tmp_path / "results"
    run_name = "empty_strict"

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
        "10",
        "--bootstrap_seed",
        "0",
        "--ci",
        "0.95",
        "--strict_metrics",
        "--disamb_path",
        str(disamb_path),
        "--cf_path",
        str(cf_path),
        "--coh_path",
        str(coh_path),
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "missing/invalid primary metrics" in (proc.stderr + proc.stdout)


def test_aom_eval_all_runs_fail_sets_fail_status_and_writes_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    missing_model_dir = tmp_path / "missing_model"

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    cf_path = tmp_path / "counterfactual.jsonl"
    coh_path = tmp_path / "coherence.jsonl"
    write_jsonl([], disamb_path)
    write_jsonl([], cf_path)
    write_jsonl([], coh_path)

    results_dir = tmp_path / "results"
    run_name = "all_fail"
    manifest_path = results_dir / f"{run_name}.manifest.json"

    cmd = [
        sys.executable,
        str(root / "aom_eval.py"),
        "--models",
        str(missing_model_dir),
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
        "10",
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
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert proc.returncode != 0

    assert manifest_path.exists()
    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "FAIL"
    assert manifest.get("run_summary", {}).get("attempted") == 1
    assert manifest.get("run_summary", {}).get("succeeded") == 0
    assert manifest.get("run_summary", {}).get("failed") == 1
    assert manifest.get("device_backend") == "cpu"
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("determinism_requested") == "best_effort"
    assert repro.get("seed") == 0
    assert repro.get("seeds") == [0]
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    assert set(datasets.keys()) >= {"disamb", "cf", "coh"}


def test_aom_eval_invalid_dataset_row_warns_and_records_dataset_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

    disamb_path = tmp_path / "disamb_pairs.jsonl"
    cf_path = tmp_path / "counterfactual.jsonl"
    coh_path = tmp_path / "coherence.jsonl"

    # Write a valid row then append an invalid JSON line.
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
    disamb_path.write_text(disamb_path.read_text(encoding="utf-8") + "{\n", encoding="utf-8")

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
    run_name = "invalid_row_warn"

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
        "10",
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
    ]
    subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)

    manifest = read_run_manifest(results_dir / f"{run_name}.manifest.json")
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "WARN"
    assert "dataset disamb: invalid_rows 1 of 2" in (manifest.get("run_status_reasons") or [])
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    disamb = datasets.get("disamb")
    assert isinstance(disamb, dict)
    assert disamb.get("n_rows_total") == 2
    assert disamb.get("n_rows_valid") == 1
    assert disamb.get("n_rows_invalid") == 1


def test_aom_eval_invalid_dataset_row_fails_strict_data(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

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
    disamb_path.write_text(disamb_path.read_text(encoding="utf-8") + "{\n", encoding="utf-8")
    write_jsonl([], cf_path)
    write_jsonl([], coh_path)

    results_dir = tmp_path / "results"
    run_name = "invalid_row_strict"
    manifest_path = results_dir / f"{run_name}.manifest.json"

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
        "10",
        "--bootstrap_seed",
        "0",
        "--ci",
        "0.95",
        "--strict_data",
        "--disamb_path",
        str(disamb_path),
        "--cf_path",
        str(cf_path),
        "--coh_path",
        str(coh_path),
    ]
    proc = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert proc.returncode != 0

    assert manifest_path.exists()
    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    assert manifest.get("run_status") == "FAIL"
    datasets = manifest.get("datasets")
    assert isinstance(datasets, dict)
    disamb = datasets.get("disamb")
    assert isinstance(disamb, dict)
    assert disamb.get("n_rows_total") == 2
    assert disamb.get("n_rows_valid") == 1
    assert disamb.get("n_rows_invalid") == 1


def test_aom_eval_strict_determinism_repeats_identically_on_cpu(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "local_model"
    _write_local_tiny_gpt2(model_dir)

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

    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run_name = "det"

    base_cmd = [
        sys.executable,
        str(root / "aom_eval.py"),
        "--models",
        str(model_dir),
        "--device",
        "cpu",
        "--determinism",
        "strict",
        "--local_files_only",
        "--run_name",
        run_name,
        "--seed",
        "0",
        "--bootstrap_n",
        "10",
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
    ]

    subprocess.run([*base_cmd, "--results_dir", str(run1_dir)], cwd=str(root), check=True, capture_output=True, text=True)
    subprocess.run([*base_cmd, "--results_dir", str(run2_dir)], cwd=str(root), check=True, capture_output=True, text=True)

    csv1 = run1_dir / f"{run_name}.csv"
    csv2 = run2_dir / f"{run_name}.csv"
    rows1 = list(csv.DictReader(csv1.read_text(encoding="utf-8").splitlines()))
    rows2 = list(csv.DictReader(csv2.read_text(encoding="utf-8").splitlines()))
    assert [_strip_nondeterministic_row_fields(r) for r in rows1] == [_strip_nondeterministic_row_fields(r) for r in rows2]

    r = rows1[0]
    started = datetime.fromisoformat(str(r["started_at_utc"]))
    ended = datetime.fromisoformat(str(r["ended_at_utc"]))
    wall = float(r["wall_time_sec"])
    eval_wall = float(r["eval_wall_time_sec"])
    assert ended >= started
    assert wall > 0.0
    assert eval_wall > 0.0
    assert wall >= eval_wall
    assert abs((ended - started).total_seconds() - wall) <= 2.0

    manifest = read_run_manifest(run1_dir / f"{run_name}.manifest.json")
    validate_run_manifest(manifest)
    repro = manifest.get("repro")
    assert isinstance(repro, dict)
    assert repro.get("determinism_requested") == "strict"
    assert repro.get("determinism_enforced") is True
