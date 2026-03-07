from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence, Tuple

from aom.data.schemas import DisambPair
from aom.stats import bootstrap_ci


def _contains_any(text: str, cues: Sequence[str]) -> int:
    t = text.lower()
    hits = 0
    for cue in cues:
        cue = cue.lower()
        # Use a word-boundary regex for simple alphabetic cues; fall back to substring otherwise.
        if cue.replace(" ", "").isalpha():
            if re.search(rf"\b{re.escape(cue)}\b", t):
                hits += 1
        else:
            if cue in t:
                hits += 1
    return hits


# Trivial keyword baseline keyed by the ambiguous target word.
# This is intentionally simple and aims to detect "easy cueing" in templated prompts.
_DISAMB_CUES: Dict[str, Dict[str, List[str]]] = {
    "bank": {
        "finance": ["loan", "money", "account", "manager", "approved", "discuss", "reviewed"],
        "river": ["river", "water", "stream", "muddy", "flood", "camped", "walked", "along"],
    },
    "bat": {
        "animal": ["cave", "dusk", "flew", "wings", "hung", "upside"],
        "sports": ["swung", "hit", "cracked", "struck", "wood", "pitcher", "game", "face"],
    },
    "spring": {
        "season": ["flowers", "bloom", "weather", "warm", "birds", "days"],
        "water": ["cold", "bubbled", "drank", "fed", "clear", "rocks", "stream"],
    },
    "match": {
        "fire": ["struck", "burned", "ignited", "candle", "stove", "light"],
        "game": ["score", "referee", "crowd", "postponed", "team", "final", "ended"],
    },
    "pitcher": {
        "container": ["poured", "water", "glass", "sink", "table", "full"],
        "baseball": ["threw", "fast", "mound", "batter", "struck out", "sweat"],
    },
    "mole": {
        "animal": ["dug", "tunnels", "ground", "soil", "lawn", "dirt"],
        "spy": ["leaked", "secrets", "suspected", "agency", "enemy", "information"],
    },
    "jam": {
        "music": ["band", "session", "guitarist", "improvise", "dance", "play"],
        "traffic": ["traffic", "cars", "road", "highway", "miles", "slowly", "cleared"],
    },
    "seal": {
        "animal": ["zoo", "swam", "flippers", "rocks", "ocean", "pool"],
        "stamp": ["pressed", "document", "wax", "letter", "envelope"],
    },
    "bark": {
        "tree": ["tree", "peeled", "trunk", "wood", "branch", "protected"],
        "dog": ["night", "heard", "loud", "dog", "puppy", "kennel", "startled"],
    },
    "crane": {
        "bird": ["wetland", "wings", "nested", "nest", "reeds", "silently"],
        "machine": ["construction", "site", "operator", "hook", "steel", "beam", "load"],
    },
    "club": {
        "group": ["joined", "meetings", "members", "discussed", "elected"],
        "weapon": ["carried", "heavy", "swung", "struck", "protect", "target", "ground"],
    },
    "watch": {
        "timepiece": ["checked", "time", "gold", "minutes", "stopped", "expensive"],
        "observe": ["watch the", "watched", "children", "game", "birds", "stands"],
    },
    "date": {
        "calendar": ["meeting", "schedule", "confirmed", "event"],
        "fruit": ["ate", "sweet", "dessert", "market", "chopped", "meal"],
    },
}


def predict_disamb_keyword(pair: DisambPair, prompt: str) -> str:
    word = str((pair.metadata or {}).get("word") or pair.target).lower()
    cues_for_word = _DISAMB_CUES.get(word, {})

    labels = sorted(pair.choices.keys())
    if not labels:
        raise ValueError(f"{pair.pair_id}: empty choices")

    scored: List[Tuple[int, str]] = []
    for lab in labels:
        scored.append((_contains_any(prompt, cues_for_word.get(lab, [])), lab))
    scored.sort(reverse=True)  # higher hits, then lexicographic label
    return scored[0][1]


def compute_disamb_keyword_baseline(
    items: Sequence[DisambPair],
    *,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, float]:
    """
    Keyword heuristic baseline for AoM-DISAMB.

    Intended as a "how easy is this suite?" sanity check, not as a competitive baseline.
    """
    pair_acc: List[float] = []
    for it in items:
        a_pred = predict_disamb_keyword(it, it.a.prompt)
        b_pred = predict_disamb_keyword(it, it.b.prompt)
        a_correct = float(a_pred == it.a.expected_label)
        b_correct = float(b_pred == it.b.expected_label)
        pair_acc.append(0.5 * (a_correct + b_correct))

    mu, lo, hi = bootstrap_ci(pair_acc, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    return {
        "keyword_accuracy": mu,
        "keyword_accuracy_ci_low": lo,
        "keyword_accuracy_ci_high": hi,
        "n_pairs_total": int(len(items)),
    }
