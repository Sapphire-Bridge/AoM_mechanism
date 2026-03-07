from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerFast

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


def test_export_mom_flagship_manifest_end_to_end(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)

    vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
        "Harmful": 3,
        "Benign": 4,
        "request": 5,
        "Decide": 6,
        "R": 7,
        "G": 8,
    }
    tokenizer = _make_wordlevel_tokenizer(vocab)
    tokenizer.save_pretrained(tok_dir)

    disamb_path = tmp_path / "disamb.jsonl"
    write_jsonl(
        [
            {
                "pair_id": "safety-0",
                "target": "Decide",
                "target_occurrence": 0,
                "a": {"prompt": "Harmful request Decide ", "expected_label": "R"},
                "b": {"prompt": "Benign request Decide ", "expected_label": "G"},
                "choices": {"R": [" R"], "G": [" G"]},
                "metadata": {
                    "template_id": "tmpl-0",
                    "risk_domain": "cybersec",
                    "direction_ids": {"a": "harmful_to_safe", "b": "safe_to_benign"},
                },
            }
        ],
        disamb_path,
    )

    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        "\n".join(
            [
                "protocol:",
                "  name: mom_safety_flagship",
                "  version: v1",
                "  prereg_tag: protocol_v1",
                "  status: frozen",
                "task:",
                "  refusal_label: R",
                "  guidance_label: G",
                "",
            ]
        ),
        encoding="utf-8",
    )

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "default_selected_layers": [5],
                "default_selected_heads": {"5": [1, 3]},
                "by_pair_id": {
                    "safety-0": {
                        "by_side": {
                            "a": {"selected_layers": [7], "selected_heads": {"7": [2]}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "mom_flagship_manifest.json"
    cmd = [
        sys.executable,
        str(root / "scripts" / "export_mom_flagship_manifest.py"),
        "--model_name_or_path",
        str(tok_dir),
        "--tokenizer_name",
        str(tok_dir),
        "--local_files_only",
        "--disamb_path",
        str(disamb_path),
        "--protocol_path",
        str(protocol_path),
        "--selection_json",
        str(selection_path),
        "--output_path",
        str(output_path),
        "--boundary_check",
        "off",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)

    out = json.loads(output_path.read_text(encoding="utf-8"))
    assert out["manifest_version"] == "1.0"
    assert out["protocol_name"] == "mom_safety_flagship"
    assert out["n_entries"] == 2
    assert isinstance(out["protocol_sha256"], str) and len(out["protocol_sha256"]) == 64
    assert out["protocol_sha256_source"] == "computed_from_path"
    assert out["protocol_sha256_verified"] is True
    assert out["transformers_version"]

    items = out["items"]
    by_side = {str(it["side"]): it for it in items}
    assert set(by_side.keys()) == {"a", "b"}

    a_item = by_side["a"]
    b_item = by_side["b"]

    assert a_item["pair_id"] == "safety-0"
    assert a_item["refusal_label"] == "R"
    assert a_item["guidance_label"] == "G"
    assert isinstance(a_item["refusal_token_id"], int)
    assert isinstance(a_item["guidance_token_id"], int)
    assert int(a_item["target_pos"]) >= 0
    assert a_item["token_event_valid"] is True
    assert a_item["selected_layers"] == [7]
    assert a_item["selected_heads"] == {"7": [2]}

    assert b_item["token_event_valid"] is True
    assert b_item["selected_layers"] == [5]
    assert b_item["selected_heads"] == {"5": [1, 3]}
