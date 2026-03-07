from pathlib import Path

from aom.baselines import compute_disamb_keyword_baseline
from aom.data.loaders import load_disamb_pairs

import re


BASE = Path(__file__).resolve().parents[1]

def _normalize_continuation_token(text: str) -> str:
    tok = text.strip().lower()
    tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
    return tok


def _prompt_contains_token(prompt: str, token: str) -> bool:
    t = prompt.lower()
    if token.replace(" ", "").replace("-", "").isalpha():
        return re.search(rf"\b{re.escape(token)}\b", t) is not None
    return token in t


def test_disamb_suite_keyword_baseline_not_too_high():
    items = load_disamb_pairs(str(BASE / "data" / "disamb_pairs.jsonl"))
    res = compute_disamb_keyword_baseline(items, bootstrap_n=200, bootstrap_seed=0)
    assert 0.0 <= res["keyword_accuracy"] <= 0.65


def test_disamb_prompts_do_not_contain_any_continuation_verbatim():
    items = load_disamb_pairs(str(BASE / "data" / "disamb_pairs.jsonl"))
    for it in items:
        flat = []
        for conts in it.choices.values():
            flat.extend(list(conts))

        for side in (it.a, it.b):
            p = side.prompt.lower()
            hits = re.findall(rf"\b{re.escape(it.target.lower())}\b", p)
            assert len(hits) == 1
            for c in flat:
                tok = _normalize_continuation_token(c)
                if tok:
                    assert not _prompt_contains_token(p, tok)
