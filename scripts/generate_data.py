from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List


def write_jsonl(rows: List[Any], path: str | Path) -> None:
    """
    Minimal JSONL writer kept local to avoid importing heavyweight ML deps during data generation.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            if is_dataclass(row):
                row = asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data", help="Output directory for JSONL files.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_per_lex_item", type=int, default=4, help="Pairs per lexical ambiguity item.")
    p.add_argument(
        "--disamb_mode",
        type=str,
        default="hardened",
        choices=("easy", "hardened"),
        help="DISAMB suite variant; 'hardened' uses cue-balanced templates, paraphrases, and adversarial minimal pairs.",
    )
    p.add_argument("--n_coh", type=int, default=40, help="Number of coherence items.")
    p.add_argument("--coh_include_controls", action="store_true", help="Include coherence ablation controls (triplets).")
    p.add_argument("--cf_include_shams", action="store_true", help="Include meaning-preserving sham CF controls.")
    p.add_argument(
        "--cf_include_graded",
        action="store_true",
        help="Include graded CF items (meaning-relevant interventions expected to produce partial shifts).",
    )
    return p.parse_args()


def format_continuation(text: str) -> str:
    """
    Format continuation strings for tokenizer compatibility.

    GPT-2 tokenizers often expect a leading space for word boundaries (" river").
    For other tokenizers (e.g. Qwen), a leading space is typically still safe.
    """
    if not text.startswith(" "):
        return " " + text
    return text


def make_sham_prompt(prompt: str) -> str:
    """
    Create a minimal, meaning-preserving prompt edit for sham CF controls.

    The goal is to introduce a small surface change that should not change the
    diagnostic preference, allowing us to measure "no-effect" shift magnitudes.
    """
    if "Therefore " in prompt and "Therefore, " not in prompt:
        return prompt.replace("Therefore ", "Therefore, ", 1)
    if prompt.endswith(" was the"):
        return prompt[: -len(" was the")] + " was indeed the"
    return prompt


def join_sentences(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def pick_fillers_by_word_count(
    *,
    fillers: List[str],
    target_words: int,
    rng: random.Random,
    k_best: int = 6,
) -> List[str]:
    """
    Select filler sentences with word-count close to `target_words`.

    This is a lightweight proxy for token-length matching; the evaluator reports actual
    tokenizer token counts so length confounds can be checked per model.
    """
    pool = list(fillers)
    picks: List[str] = []
    for _ in range(2):
        pool.sort(key=lambda s: abs(word_count(s) - target_words))
        top = pool[: max(1, min(k_best, len(pool)))]
        chosen = rng.choice(top)
        picks.append(chosen)
        pool.remove(chosen)
    return picks

def _ensure_target_once(*, prompt: str, target: str) -> None:
    hits = re.findall(rf"\b{re.escape(target)}\b", prompt, flags=re.IGNORECASE)
    if len(hits) != 1:
        raise ValueError(f"DISAMB prompt must contain target exactly once: target={target!r} prompt={prompt!r}")

def _normalize_continuation_token(text: str) -> str:
    # Remove whitespace and surrounding punctuation so word-boundary checks behave as expected.
    tok = text.strip().lower()
    tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok)
    return tok


def _contains_any_continuation(*, prompt: str, continuations: List[str]) -> bool:
    t = prompt.lower()
    for c in continuations:
        tok = _normalize_continuation_token(c)
        if not tok:
            continue
        if tok.replace(" ", "").replace("-", "").isalpha():
            if re.search(rf"\b{re.escape(tok)}\b", t):
                return True
        else:
            if tok in t:
                return True
    return False


def generate_disamb_pairs_easy(seed: int, n_per_item: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    specs = [
        {
            "word": "bank",
            "labels": ("finance", "river"),
            "prompts": {
                "finance": [
                    "I went to the bank to discuss the",
                    "At the bank, I asked about the",
                    "The bank approved my",
                    "The bank manager reviewed the",
                ],
                "river": [
                    "I sat by the bank and watched the",
                    "We walked along the bank near the",
                    "The bank was muddy after the",
                    "They camped on the bank beside the",
                ],
            },
            "choices": {
                "finance": [" loan", " money", " account"],
                "river": [" river", " water", " stream"],
            },
        },
        {
            "word": "bat",
            "labels": ("animal", "sports"),
            "prompts": {
                "animal": [
                    "In the cave, the bat flew into the",
                    "At dusk, a bat emerged from the",
                    "The bat hung upside down in the",
                    "A bat had wings visible in the",
                ],
                "sports": [
                    "He swung the bat and hit the",
                    "The bat cracked when it struck the",
                    "She picked up the bat to face the",
                    "The bat was made of wood and hit the",
                ],
            },
            "choices": {
                "animal": [" cave", " night", " air"],
                "sports": [" ball", " pitcher", " game"],
            },
        },
        {
            "word": "spring",
            "labels": ("season", "water"),
            "prompts": {
                "season": [
                    "In spring, the flowers began to",
                    "Every spring, the weather starts to",
                    "In spring, many birds",
                    "During spring, the days become",
                ],
                "water": [
                    "A cold spring bubbled up from the",
                    "They drank from the spring near the",
                    "The spring fed a small",
                    "The spring was clear and full of",
                ],
            },
            "choices": {
                "season": [" bloom", " warm", " grow"],
                "water": [" water", " rocks", " stream"],
            },
        },
        {
            "word": "match",
            "labels": ("fire", "game"),
            "prompts": {
                "fire": [
                    "He struck a match to light the",
                    "A match burned brightly in the",
                    "She dropped the match into the",
                    "The match ignited the",
                ],
                "game": [
                    "The match ended with a final",
                    "They watched the match from the",
                    "The match was postponed due to",
                    "After the match, the team celebrated the",
                ],
            },
            "choices": {
                "fire": [" candle", " fire", " stove"],
                "game": [" score", " referee", " crowd"],
            },
        },
        {
            "word": "pitcher",
            "labels": ("container", "baseball"),
            "prompts": {
                "container": [
                    "She poured water from the pitcher into the",
                    "The pitcher was full of",
                    "He washed the pitcher in the",
                    "The pitcher sat on the table beside the",
                ],
                "baseball": [
                    "The pitcher threw a fast",
                    "The pitcher walked toward the",
                    "The pitcher struck out the",
                    "The pitcher wiped sweat from his",
                ],
            },
            "choices": {
                "container": [" glass", " water", " sink"],
                "baseball": [" ball", " mound", " batter"],
            },
        },
        {
            "word": "mole",
            "labels": ("animal", "spy"),
            "prompts": {
                "animal": [
                    "A mole dug tunnels under the",
                    "The mole popped out of the",
                    "A mole has poor",
                    "The mole pushed dirt from the",
                ],
                "spy": [
                    "The mole leaked secrets to the",
                    "They suspected a mole inside the",
                    "The mole worked for the",
                    "The mole passed information to the",
                ],
            },
            "choices": {
                "animal": [" ground", " soil", " lawn"],
                "spy": [" enemy", " agency", " press"],
            },
        },
        {
            "word": "jam",
            "labels": ("music", "traffic"),
            "prompts": {
                "music": [
                    "The band started a jam and everyone began to",
                    "They held a jam session in the",
                    "The guitarist loved to jam with the",
                    "A jam broke out during the",
                ],
                "traffic": [
                    "There was a traffic jam on the",
                    "The jam stretched for miles along the",
                    "Because of the jam, cars moved",
                    "The jam cleared after the",
                ],
            },
            "choices": {
                "music": [" play", " improvise", " dance"],
                "traffic": [" road", " highway", " slowly"],
            },
        },
        {
            "word": "seal",
            "labels": ("animal", "stamp"),
            "prompts": {
                "animal": [
                    "At the zoo, the seal balanced the",
                    "A seal swam quickly through the",
                    "The seal clapped its flippers in the",
                    "A seal rested on the rocks near the",
                ],
                "stamp": [
                    "She pressed the seal onto the",
                    "The document needed a seal for the",
                    "The wax seal was placed on the",
                    "He checked the seal on the",
                ],
            },
            "choices": {
                "animal": [" water", " pool", " ocean"],
                "stamp": [" letter", " document", " envelope"],
            },
        },
        {
            "word": "bark",
            "labels": ("tree", "dog"),
            "prompts": {
                "tree": [
                    "The bark of the tree was rough to the",
                    "She peeled the bark from the",
                    "The bark protected the tree from the",
                    "The bark was thick on the",
                ],
                "dog": [
                    "At night, I heard the bark of the",
                    "The bark startled the",
                    "A loud bark came from the",
                    "The bark grew louder as the",
                ],
            },
            "choices": {
                "tree": [" trunk", " wood", " branch"],
                "dog": [" dog", " puppy", " kennel"],
            },
        },
        {
            "word": "crane",
            "labels": ("bird", "machine"),
            "prompts": {
                "bird": [
                    "In the wetland, the crane lifted its",
                    "A crane stood silently near the",
                    "The crane spread its wings above the",
                    "A crane nested close to the",
                ],
                "machine": [
                    "At the construction site, the crane lifted the",
                    "The crane swung the",
                    "A crane operator moved the",
                    "The crane's hook grabbed the",
                ],
            },
            "choices": {
                "bird": [" water", " reeds", " nest"],
                "machine": [" steel", " beam", " load"],
            },
        },
        {
            "word": "club",
            "labels": ("group", "weapon"),
            "prompts": {
                "group": [
                    "She joined the club to meet new",
                    "The club held meetings every",
                    "At the club, members discussed the",
                    "The club elected a new",
                ],
                "weapon": [
                    "He carried a club to protect",
                    "The club was heavy and made of",
                    "She swung the club at the",
                    "The club struck the",
                ],
            },
            "choices": {
                "group": [" people", " week", " topic"],
                "weapon": [" wood", " target", " ground"],
            },
        },
        {
            "word": "watch",
            "labels": ("timepiece", "observe"),
            "prompts": {
                "timepiece": [
                    "He checked his watch to see the",
                    "The watch showed the correct",
                    "Her watch was expensive and made of",
                    "The watch stopped working after the",
                ],
                "observe": [
                    "I will watch the children while you",
                    "They watch the game from the",
                    "We watched the birds as they",
                    "She watched him as he",
                ],
            },
            "choices": {
                "timepiece": [" time", " gold", " minutes"],
                "observe": [" play", " stands", " flew"],
            },
        },
        {
            "word": "date",
            "labels": ("calendar", "fruit"),
            "prompts": {
                "calendar": [
                    "They set a date for the",
                    "The date of the meeting was",
                    "She forgot the date of the",
                    "He confirmed the date with the",
                ],
                "fruit": [
                    "He ate a date after the",
                    "A date is sweet and often served with",
                    "She bought a date at the",
                    "They chopped a date into the",
                ],
            },
            "choices": {
                "calendar": [" meeting", " event", " schedule"],
                "fruit": [" dessert", " market", " meal"],
            },
        },
    ]

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        word = spec["word"]
        l1, l2 = spec["labels"]
        p1 = list(spec["prompts"][l1])
        p2 = list(spec["prompts"][l2])
        rng.shuffle(p1)
        rng.shuffle(p2)
        choices = {k: [format_continuation(x) for x in v] for k, v in spec["choices"].items()}
        n = min(n_per_item, len(p1), len(p2))
        for i in range(n):
            rows.append(
                {
                    "pair_id": f"{word}-lex-{i}",
                    "target": word,
                    "target_occurrence": 0,
                    "a": {"prompt": p1[i], "expected_label": l1},
                    "b": {"prompt": p2[i], "expected_label": l2},
                    "choices": choices,
                    "metadata": {"type": "lexical", "word": word, "labels": [l1, l2], "source": "template", "variant": "easy"},
                }
            )
    return rows


def generate_disamb_pairs_hardened(seed: int, n_per_item: int) -> List[Dict[str, Any]]:
    """
    Hardened AoM-DISAMB suite:
    - includes distractor-balanced variants (opposite-sense cues in an irrelevant clause)
    - includes paraphrase variants (surface variation without changing the disambiguating structure)
    - adversarial minimal pairs: include cases where naive keyword cues are present on both sides
    """
    rng = random.Random(seed)

    names = ["Alex", "Sam", "Taylor", "Jordan", "Casey", "Riley"]
    times = ["Earlier,", "Before that,"]

    # NOTE: Prompts are written to avoid the disambiguating continuation strings appearing verbatim in the prompt.
    # This makes "copy-the-answer" heuristics harder and reduces overlap with the keyword baseline cues.
    hardened_specs: List[Dict[str, Any]] = [
        {
            "word": "bank",
            "labels": ("finance", "river"),
            "choices": {"finance": [" loan", " money", " account"], "river": [" river", " water", " stream"]},
            "pairs": [
                # Clean: disambiguate via institution vs shoreline, using non-baseline cues.
                (
                    "{time} {name} visited the {target} to handle paperwork for the",
                    "{time} {name} sat beside the {target} on the shoreline and watched the",
                    "clean",
                ),
                # Adversarial: add distractors that belong to the opposite sense.
                (
                    "Because of the overflowed creek, {name} visited the {target} to handle paperwork for the",
                    "After an ATM withdrawal, {name} sat beside the {target} on the shoreline and watched the",
                    "distractor",
                ),
                # Paraphrase variants
                (
                    "{time} {name} headed to the {target} to file forms related to the",
                    "{time} {name} rested near the {target} at the embankment and watched the",
                    "paraphrase_clean",
                ),
                (
                    "Due to the overflowing creek, {name} headed to the {target} to file forms related to the",
                    "After an ATM withdrawal, {name} rested near the {target} at the embankment and watched the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "bat",
            "labels": ("animal", "sports"),
            "choices": {"animal": [" cave", " night", " air"], "sports": [" ball", " pitcher", " game"]},
            "pairs": [
                (
                    "{time} {name} saw a {target} fly through the",
                    "{time} {name} carried the {target} to the stadium and watched the",
                    "clean",
                ),
                (
                    "After the stadium announcer spoke, {name} saw a {target} fly through the",
                    "After hearing ultrasonic squeaks, {name} carried the {target} to the stadium and watched the",
                    "distractor",
                ),
                (
                    "{time} {name} noticed a {target} gliding through the",
                    "{time} {name} brought the {target} to the field and watched the",
                    "paraphrase_clean",
                ),
                (
                    "After the stadium announcer spoke, {name} noticed a {target} gliding through the",
                    "After hearing ultrasonic squeaks, {name} brought the {target} to the field and watched the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "spring",
            "labels": ("season", "water"),
            "choices": {"season": [" bloom", " warm", " grow"], "water": [" water", " rocks", " stream"]},
            "pairs": [
                (
                    "{time} {name} looks forward to {target} because gardens begin to",
                    "{time} {name} followed the {target} from a hillside source toward the",
                    "clean",
                ),
                (
                    "After a hike past a hillside source, {name} looks forward to {target} because gardens begin to",
                    "After planning a garden, {name} followed the {target} from a hillside source toward the",
                    "distractor",
                ),
                (
                    "{time} {name} enjoys {target} since the air starts to",
                    "{time} {name} traced the {target} from an underground source toward the",
                    "paraphrase_clean",
                ),
                (
                    "After a hike past an underground source, {name} enjoys {target} since the air starts to",
                    "After planning a garden, {name} traced the {target} from an underground source toward the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "match",
            "labels": ("fire", "game"),
            "choices": {"fire": [" candle", " fire", " stove"], "game": [" score", " referee", " crowd"]},
            "pairs": [
                (
                    "{time} {name} used a {target} to light the",
                    "{time} {name} attended the {target} at the arena to see the",
                    "clean",
                ),
                (
                    "After an arena announcement, {name} used a {target} to light the",
                    "After lighting a small flame, {name} attended the {target} at the arena to see the",
                    "distractor",
                ),
                (
                    "{time} {name} grabbed a {target} to light the",
                    "{time} {name} went to the {target} at the arena and looked at the",
                    "paraphrase_clean",
                ),
                (
                    "After an arena announcement, {name} grabbed a {target} to light the",
                    "After lighting a small flame, {name} went to the {target} at the arena and looked at the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "pitcher",
            "labels": ("container", "baseball"),
            "choices": {"container": [" glass", " water", " sink"], "baseball": [" ball", " mound", " batter"]},
            "pairs": [
                (
                    "{time} {name} filled a {target} with lemonade and reached for the",
                    "{time} {name} praised the {target} after the inning and waited for the",
                    "clean",
                ),
                (
                    "After the inning ended, {name} filled a {target} with lemonade and reached for the",
                    "After pouring lemonade, {name} praised the {target} after the inning and waited for the",
                    "distractor",
                ),
                (
                    "{time} {name} rinsed a {target} in the kitchen and reached for the",
                    "{time} {name} cheered the {target} in the dugout and waited for the",
                    "paraphrase_clean",
                ),
                (
                    "After the inning ended, {name} rinsed a {target} in the kitchen and reached for the",
                    "After pouring lemonade, {name} cheered the {target} in the dugout and waited for the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "mole",
            "labels": ("animal", "spy"),
            "choices": {"animal": [" ground", " soil", " lawn"], "spy": [" enemy", " agency", " press"]},
            "pairs": [
                (
                    "{time} {name} spotted a {target} making a burrow under the",
                    "{time} {name} uncovered a {target} acting as a double agent for the",
                    "clean",
                ),
                (
                    "After reading about a double agent, {name} spotted a {target} making a burrow under the",
                    "After inspecting a garden bed, {name} uncovered a {target} acting as a double agent for the",
                    "distractor",
                ),
                (
                    "{time} {name} noticed a {target} tunneling below the",
                    "{time} {name} confronted a {target} passing classified notes to the",
                    "paraphrase_clean",
                ),
                (
                    "After reading about classified notes, {name} noticed a {target} tunneling below the",
                    "After inspecting a garden bed, {name} confronted a {target} passing classified notes to the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "jam",
            "labels": ("music", "traffic"),
            "choices": {"music": [" play", " improvise", " dance"], "traffic": [" road", " highway", " street"]},
            "pairs": [
                (
                    "{time} {name} joined a {target} with musicians and began to",
                    "{time} {name} got stuck in a {target} on the",
                    "clean",
                ),
                (
                    "After sitting at an intersection, {name} joined a {target} with musicians and began to",
                    "After a long rehearsal, {name} got stuck in a {target} on the",
                    "distractor",
                ),
                (
                    "{time} {name} started a {target} with friends and began to",
                    "{time} {name} faced a {target} on the",
                    "paraphrase_clean",
                ),
                (
                    "After sitting at an intersection, {name} started a {target} with friends and began to",
                    "After a long rehearsal, {name} faced a {target} on the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "seal",
            "labels": ("animal", "stamp"),
            "choices": {"animal": [" water", " pool", " ocean"], "stamp": [" letter", " document", " envelope"]},
            "pairs": [
                (
                    "{time} {name} saw a {target} at the aquarium and looked into the",
                    "{time} {name} needed a {target} to certify the form and checked the",
                    "clean",
                ),
                (
                    "After filling out a form, {name} saw a {target} at the aquarium and looked into the",
                    "After visiting an aquarium, {name} needed a {target} to certify the form and checked the",
                    "distractor",
                ),
                (
                    "{time} {name} watched a {target} near the coast and looked into the",
                    "{time} {name} asked for a {target} to authenticate the form and checked the",
                    "paraphrase_clean",
                ),
                (
                    "After filling out a form, {name} watched a {target} near the coast and looked into the",
                    "After visiting the coast, {name} asked for a {target} to authenticate the form and checked the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "bark",
            "labels": ("tree", "dog"),
            "choices": {"tree": [" trunk", " wood", " branch"], "dog": [" dog", " puppy", " kennel"]},
            "pairs": [
                (
                    "{time} {name} touched the {target} on an old oak and pointed to the",
                    "{time} {name} heard a {target} from a nearby pet and looked for the",
                    "clean",
                ),
                (
                    "After checking on a nearby pet, {name} touched the {target} on an old oak and pointed to the",
                    "After walking in a forest, {name} heard a {target} from a nearby pet and looked for the",
                    "distractor",
                ),
                (
                    "{time} {name} examined the {target} of an oak and pointed to the",
                    "{time} {name} noticed a {target} outside the house and looked for the",
                    "paraphrase_clean",
                ),
                (
                    "After checking outside the house, {name} examined the {target} of an oak and pointed to the",
                    "After walking in a forest, {name} noticed a {target} outside the house and looked for the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "crane",
            "labels": ("bird", "machine"),
            "choices": {"bird": [" water", " reeds", " nest"], "machine": [" steel", " beam", " load"]},
            "pairs": [
                (
                    "{time} {name} photographed a {target} in the marsh and stood near the",
                    "{time} {name} operated a {target} at a building project and raised the",
                    "clean",
                ),
                (
                    "After visiting a building project, {name} photographed a {target} in the marsh and stood near the",
                    "After walking through a marsh, {name} operated a {target} at a building project and raised the",
                    "distractor",
                ),
                (
                    "{time} {name} spotted a {target} in the marsh and stood near the",
                    "{time} {name} guided a {target} at a building project and raised the",
                    "paraphrase_clean",
                ),
                (
                    "After visiting a building project, {name} spotted a {target} in the marsh and stood near the",
                    "After walking through a marsh, {name} guided a {target} at a building project and raised the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "club",
            "labels": ("group", "weapon"),
            "choices": {"group": [" agenda", " bylaws", " topic"], "weapon": [" intruder", " target", " ground"]},
            "pairs": [
                (
                    "{time} {name} joined the {target} meeting and discussed the",
                    "{time} {name} grabbed a {target} and struck the",
                    "clean",
                ),
                (
                    "After training with a blunt tool, {name} joined the {target} meeting and discussed the",
                    "After attending a committee meeting, {name} grabbed a {target} and struck the",
                    "distractor",
                ),
                (
                    "{time} {name} helped run the {target} and voted on the",
                    "{time} {name} swung the {target} and hit the",
                    "paraphrase_clean",
                ),
                (
                    "After training with a heavy tool, {name} helped run the {target} and voted on the",
                    "After attending a committee meeting, {name} swung the {target} and hit the",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "watch",
            "labels": ("timepiece", "observe"),
            "choices": {"timepiece": [" time", " minutes", " seconds"], "observe": [" play", " run", " leave"]},
            "pairs": [
                (
                    "{time} {name}'s {target} had a cracked face and showed the",
                    "{time} {name} will {target} the children as they",
                    "clean",
                ),
                (
                    "After supervising the kids, {name}'s {target} had a cracked face and showed the",
                    "After replacing the strap, {name} will {target} the children as they",
                    "distractor",
                ),
                (
                    "{time} {name}'s {target} needed a new band and showed the",
                    "{time} {name} will {target} the kids as they",
                    "paraphrase_clean",
                ),
                (
                    "After supervising the kids, {name}'s {target} needed a new band and showed the",
                    "After replacing the strap, {name} will {target} the kids as they",
                    "paraphrase_distractor",
                ),
            ],
        },
        {
            "word": "date",
            "labels": ("calendar", "fruit"),
            "choices": {"calendar": [" meeting", " event", " schedule"], "fruit": [" dessert", " market", " meal"]},
            "pairs": [
                (
                    "{time} {name} picked a {target} for the appointment and wrote the",
                    "{time} {name} packed a {target} for a snack and finished the",
                    "clean",
                ),
                (
                    "After packing a snack, {name} picked a {target} for the appointment and wrote the",
                    "After writing an appointment note, {name} packed a {target} for a snack and finished the",
                    "distractor",
                ),
                (
                    "{time} {name} chose a {target} for the invite and wrote the",
                    "{time} {name} carried a {target} in a lunch bag and finished the",
                    "paraphrase_clean",
                ),
                (
                    "After carrying a lunch bag, {name} chose a {target} for the invite and wrote the",
                    "After writing an invite, {name} carried a {target} in a lunch bag and finished the",
                    "paraphrase_distractor",
                ),
            ],
        },
    ]

    rows: List[Dict[str, Any]] = []
    for spec in hardened_specs:
        word = spec["word"]
        l1, l2 = spec["labels"]
        choices = {k: [format_continuation(x) for x in v] for k, v in spec["choices"].items()}

        candidates: List[Dict[str, Any]] = []
        for j, (a_t, b_t, variant) in enumerate(spec["pairs"]):
            name = rng.choice(names)
            time = rng.choice(times)
            a_prompt = a_t.format(target=word, name=name, time=time)
            b_prompt = b_t.format(target=word, name=name, time=time)
            _ensure_target_once(prompt=a_prompt, target=word)
            _ensure_target_once(prompt=b_prompt, target=word)

            # Avoid "copy-the-answer" overlap: do not include any continuation substring in prompts.
            flat_conts: List[str] = []
            for conts in choices.values():
                flat_conts.extend(list(conts))
            if _contains_any_continuation(prompt=a_prompt, continuations=flat_conts):
                continue
            if _contains_any_continuation(prompt=b_prompt, continuations=flat_conts):
                continue

            candidates.append(
                {
                    "pair_id": f"{word}-lex-{j}",
                    "target": word,
                    "target_occurrence": 0,
                    "a": {"prompt": a_prompt, "expected_label": l1},
                    "b": {"prompt": b_prompt, "expected_label": l2},
                    "choices": choices,
                    "metadata": {
                        "type": "lexical",
                        "word": word,
                        "labels": [l1, l2],
                        "source": "template",
                        "variant": "hardened",
                        "pair_variant": variant,
                    },
                }
            )

        if not candidates:
            raise ValueError(f"No DISAMB candidates for {word}")

        rng.shuffle(candidates)
        chosen = candidates[: min(n_per_item, len(candidates))]
        # Re-stabilize ids after selection.
        for i, row in enumerate(chosen):
            row["pair_id"] = f"{word}-lex-{i}"
            rows.append(row)
    return rows


def generate_disamb_pairs(seed: int, n_per_item: int, *, mode: str = "hardened") -> List[Dict[str, Any]]:
    if mode == "easy":
        return generate_disamb_pairs_easy(seed, n_per_item)
    if mode == "hardened":
        return generate_disamb_pairs_hardened(seed, n_per_item)
    raise ValueError(f"Unknown disamb mode: {mode!r}")


def generate_counterfactual(seed: int, *, include_shams: bool = False, include_graded: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    shams: List[Dict[str, Any]] = []

    # Negation flips
    neg_specs = [
        ("Alice", "go to the store"),
        ("Bob", "attend the meeting"),
        ("Maria", "finish the report"),
        ("John", "call the client"),
        ("Lena", "submit the form"),
        ("Omar", "open the door"),
        ("Nina", "pay the bill"),
        ("Paul", "read the email"),
        ("Eva", "visit the doctor"),
        ("Kai", "join the class"),
        ("Iris", "sign the contract"),
        ("Tom", "wash the dishes"),
        ("Sofia", "take the train"),
        ("Alex", "answer the question"),
        ("Mia", "write the code"),
        ("Noah", "buy the ticket"),
        ("Zoe", "send the message"),
        ("Hugo", "close the window"),
        ("Sara", "cook the dinner"),
        ("Liam", "start the engine"),
    ]
    for i, (name, action) in enumerate(neg_specs):
        base_prompt = f"{name} did {action}. Therefore {name}"
        cf_prompt = f"{name} didn't {action}. Therefore {name}"
        L0, L1 = "did", "did_not"
        rows.append(
            {
                "item_id": f"negation-{i}",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": cf_prompt, "expected_label": L1},
                "choices": {L0: [format_continuation("did")], L1: [format_continuation("did not")]},
                "intervention_type": "negation",
                "contrast_labels": [L0, L1],
                "expected_effect": "shift",
                "metadata": {"source": "template"},
            }
        )
        shams.append(
            {
                "item_id": f"negation-{i}__sham",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": make_sham_prompt(base_prompt), "expected_label": L0},
                "choices": {L0: [format_continuation("did")], L1: [format_continuation("did not")]},
                "intervention_type": "sham_punctuation",
                "contrast_labels": [L0, L1],
                "expected_effect": "invariant",
                "metadata": {"source": "template", "control": "sham"},
            }
        )

    # Active/passive / role swap
    role_specs = [
        ("journalist", "politician"),
        ("doctor", "patient"),
        ("teacher", "student"),
        ("chef", "critic"),
        ("lawyer", "client"),
        ("pilot", "passenger"),
        ("manager", "employee"),
        ("actor", "director"),
        ("captain", "sailor"),
        ("guard", "visitor"),
        ("detective", "suspect"),
        ("driver", "pedestrian"),
        ("seller", "buyer"),
        ("coach", "player"),
        ("nurse", "surgeon"),
        ("author", "editor"),
        ("artist", "curator"),
        ("builder", "inspector"),
        ("scientist", "journalist"),
        ("programmer", "tester"),
    ]
    for i, (a, b) in enumerate(role_specs):
        base_prompt = f"The {a} attacked the {b}. The attacker was the"
        cf_prompt = f"The {b} attacked the {a}. The attacker was the"
        L0, L1 = a, b
        rows.append(
            {
                "item_id": f"role-swap-{i}",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": cf_prompt, "expected_label": L1},
                "choices": {L0: [format_continuation(L0)], L1: [format_continuation(L1)]},
                "intervention_type": "role_swap",
                "contrast_labels": [L0, L1],
                "expected_effect": "shift",
                "metadata": {"source": "template"},
            }
        )
        shams.append(
            {
                "item_id": f"role-swap-{i}__sham",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": make_sham_prompt(base_prompt), "expected_label": L0},
                "choices": {L0: [format_continuation(L0)], L1: [format_continuation(L1)]},
                "intervention_type": "sham_emphasis",
                "contrast_labels": [L0, L1],
                "expected_effect": "invariant",
                "metadata": {"source": "template", "control": "sham"},
            }
        )

    # Quantifier swap
    quant_specs = [
        ("lights", "on", "off"),
        ("computers", "working", "broken"),
        ("tickets", "valid", "invalid"),
        ("doors", "open", "closed"),
        ("files", "complete", "missing"),
        ("engines", "running", "stalled"),
        ("orders", "approved", "rejected"),
        ("keys", "available", "lost"),
        ("phones", "charged", "dead"),
        ("servers", "online", "offline"),
        ("accounts", "active", "disabled"),
        ("packages", "delivered", "lost"),
        ("answers", "correct", "wrong"),
        ("tables", "clean", "dirty"),
        ("rooms", "ready", "occupied"),
        ("batteries", "full", "empty"),
        ("routes", "open", "blocked"),
        ("seats", "free", "taken"),
        ("reports", "finished", "unfinished"),
        ("documents", "signed", "unsigned"),
    ]
    for i, (plural, pos, neg) in enumerate(quant_specs):
        base_prompt = f"All the {plural} are {pos}. Therefore at least one {plural[:-1]} is"
        cf_prompt = f"None of the {plural} are {pos}. Therefore at least one {plural[:-1]} is"
        L0, L1 = pos, neg
        rows.append(
            {
                "item_id": f"quantifier-{i}",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": cf_prompt, "expected_label": L1},
                "choices": {L0: [format_continuation(L0)], L1: [format_continuation(L1)]},
                "intervention_type": "quantifier",
                "contrast_labels": [L0, L1],
                "expected_effect": "shift",
                "metadata": {"source": "template"},
            }
        )
        shams.append(
            {
                "item_id": f"quantifier-{i}__sham",
                "base": {"prompt": base_prompt, "expected_label": L0},
                "cf": {"prompt": make_sham_prompt(base_prompt), "expected_label": L0},
                "choices": {L0: [format_continuation(L0)], L1: [format_continuation(L1)]},
                "intervention_type": "sham_punctuation",
                "contrast_labels": [L0, L1],
                "expected_effect": "invariant",
                "metadata": {"source": "template", "control": "sham"},
            }
        )

    if include_shams:
        rows.extend(shams)
    if include_graded:
        graded_specs = [
            ("movie", "excellent", "decent", "good", "bad", "graded_modifier"),
            ("meal", "delicious", "acceptable", "good", "bad", "graded_modifier"),
            ("essay", "brilliant", "reasonable", "good", "bad", "graded_modifier"),
            ("performance", "outstanding", "solid", "good", "bad", "graded_modifier"),
            ("design", "excellent", "adequate", "good", "bad", "graded_modifier"),
            ("room", "spotless", "pretty tidy", "clean", "dirty", "graded_modifier"),
            ("kitchen", "immaculate", "fairly tidy", "clean", "dirty", "graded_modifier"),
            ("system", "secure", "mostly secure", "safe", "dangerous", "graded_modifier"),
            ("trip", "risk-free", "mostly secure", "safe", "dangerous", "graded_modifier"),
            ("route", "clear", "mostly clear", "open", "blocked", "graded_modifier"),
            ("device", "dependable", "fairly dependable", "reliable", "unreliable", "graded_modifier"),
            ("answer", "accurate", "probably accurate", "correct", "wrong", "graded_hedge"),
            ("solution", "obviously right", "probably right", "correct", "wrong", "graded_hedge"),
            ("box", "enormous", "quite large", "big", "small", "graded_modifier"),
            ("task", "trivial", "manageable", "easy", "hard", "graded_modifier"),
            ("car", "rapid", "fairly quick", "fast", "slow", "graded_modifier"),
            ("bread", "crisp", "a bit musty", "fresh", "stale", "graded_modifier"),
            ("signal", "clear", "somewhat noisy", "strong", "weak", "graded_modifier"),
            ("argument", "compelling", "somewhat plausible", "strong", "weak", "graded_modifier"),
            ("explanation", "convincing", "fairly plausible", "strong", "weak", "graded_modifier"),
        ]
        for i, (thing, strong, weak, L0, L1, itype) in enumerate(graded_specs):
            base_prompt = f"The {thing} was {strong}. Therefore the {thing} was"
            cf_prompt = f"The {thing} was {weak}. Therefore the {thing} was"
            rows.append(
                {
                    "item_id": f"graded-{i}",
                    "base": {"prompt": base_prompt, "expected_label": L0},
                    "cf": {"prompt": cf_prompt, "expected_label": L0},
                    "choices": {L0: [format_continuation(L0)], L1: [format_continuation(L1)]},
                    "intervention_type": str(itype),
                    "contrast_labels": [L0, L1],
                    "expected_effect": "graded",
                    "metadata": {"source": "template", "control": "graded"},
                }
            )
    rng.shuffle(rows)
    return rows


def generate_coherence(seed: int, n_items: int, *, include_controls: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    bases: List[Dict[str, Any]] = []

    filler = [
        "The sun set quickly.",
        "It got cold outside.",
        "The room was quiet.",
        "A clock ticked softly.",
        "The lights flickered once.",
        "The weather outside was calm and quiet.",
        "A nearby clock ticked softly on the wall.",
        "In the distance, birds sang above the trees.",
        "The room was lit by a small lamp in the corner.",
        "A gentle breeze moved through the open window.",
        "Someone down the hall hummed a familiar tune.",
        "The streetlights flickered as evening approached.",
        "A cup of tea cooled slowly on the table.",
        "The old chair creaked when someone shifted their weight.",
        "A dog barked once and then fell silent again.",
        "A distant car passed by slowly.",
        "Someone closed a door quietly.",
        "A short message appeared on screen.",
        "The hallway smelled faintly of soap.",
        "A page turned with a soft sound.",
    ]

    # Entity-state constraints (alive/dead)
    names = [
        "John",
        "Maria",
        "Alex",
        "Sofia",
        "Lena",
        "Omar",
        "Nina",
        "Paul",
        "Eva",
        "Kai",
        "Iris",
        "Tom",
        "Mia",
        "Noah",
        "Zoe",
        "Hugo",
        "Sara",
        "Liam",
        "Ava",
        "Ethan",
    ]
    for i, name in enumerate(names):
        bases.append(
            {
                "item_id": f"entity-state-{i}",
                "pre": f"{name} was injured in the accident.",
                "relevant": f"The next morning, {name} died.",
                "tail": f"Later that day, {name}",
                "valid_continuations": [format_continuation("was remembered by friends."), format_continuation("was mourned by family.")],
                "invalid_continuations": [
                    format_continuation("walked into the room smiling."),
                    format_continuation("said hello to everyone."),
                ],
                "constraint_type": "entity_state",
                "metadata": {"source": "template", "n_constraints": 1, "constraint_types": ["entity_state"]},
            }
        )

    # Location constraints
    origins = ["office", "home", "library", "school", "hotel", "clinic", "factory", "museum", "kitchen", "studio"]
    dests = ["airport", "station", "cafe", "gym", "park", "theater", "market", "garage", "garden", "harbor"]
    for i, name in enumerate(names):
        a = origins[i % len(origins)]
        b = dests[i % len(dests)]
        bases.append(
            {
                "item_id": f"location-{i}",
                "pre": "",
                "relevant": f"{name} left the {a} and drove to the {b}.",
                "tail": f"After an hour, {name} arrived at the",
                "valid_continuations": [format_continuation(f"{b}."), format_continuation(f"{b} quietly.")],
                "invalid_continuations": [format_continuation(f"{a}."), format_continuation(f"{a} again.")],
                "constraint_type": "location",
                "metadata": {"source": "template", "n_constraints": 1, "constraint_types": ["location"]},
            }
        )

    # Possession constraints
    items = [
        "keys",
        "wallet",
        "phone",
        "passport",
        "ticket",
        "ring",
        "letter",
        "badge",
        "camera",
        "notebook",
    ]
    for i, name in enumerate(names):
        item = items[i % len(items)]
        bases.append(
            {
                "item_id": f"possession-{i}",
                "pre": "",
                "relevant": f"{name} put the {item} into a drawer and locked it.",
                "tail": f"Minutes later, {name} looked for the {item} and found it in the",
                "valid_continuations": [format_continuation("drawer."), format_continuation("locked drawer.")],
                "invalid_continuations": [format_continuation("trash."), format_continuation("street.")],
                "constraint_type": "possession",
                "metadata": {"source": "template", "n_constraints": 1, "constraint_types": ["possession"]},
            }
        )

    # Multi-constraint items (location + possession).
    for i, name in enumerate(names):
        a = origins[(i + 3) % len(origins)]
        b = dests[(i + 5) % len(dests)]
        item = items[(i + 2) % len(items)]
        bases.append(
            {
                "item_id": f"multi-constraint-{i}",
                "pre": "",
                "relevant": join_sentences(
                    f"{name} left the {a} and drove to the {b}.",
                    f"{name} put the {item} into a drawer and locked it.",
                ),
                "tail": f"After an hour, {name}",
                "valid_continuations": [
                    format_continuation(f"arrived at the {b} and found the {item} in the drawer."),
                    format_continuation(f"arrived at the {b} and retrieved the {item} from the drawer."),
                ],
                "invalid_continuations": [
                    format_continuation(f"arrived at the {a} and found the {item} in the drawer."),
                    format_continuation(f"arrived at the {b} and found the {item} on the street."),
                ],
                "constraint_type": "multi_constraint",
                "metadata": {
                    "source": "template",
                    "n_constraints": 2,
                    "constraint_types": ["location", "possession"],
                },
            }
        )

    rng.shuffle(bases)
    bases = bases[: max(0, int(n_items))]

    rows: List[Dict[str, Any]] = []
    for b in bases:
        pre = str(b.get("pre", ""))
        rel = str(b.get("relevant", ""))
        tail = str(b.get("tail", ""))
        rel_wc = word_count(rel)
        f1, f2 = pick_fillers_by_word_count(fillers=filler, target_words=rel_wc, rng=rng)
        f1_wc = word_count(f1)
        f2_wc = word_count(f2)
        meta = dict(b.get("metadata", {}))
        meta.update(
            {
                "relevant_word_count": rel_wc,
                "filler1_word_count": f1_wc,
                "filler2_word_count": f2_wc,
                "filler_word_count_abs_diff": abs(f2_wc - rel_wc),
            }
        )
        base_common = {
            "valid_continuations": b["valid_continuations"],
            "invalid_continuations": b["invalid_continuations"],
            "constraint_type": b["constraint_type"],
            "metadata": meta,
        }

        rows.append(
            {
                "item_id": f"{b['item_id']}__main",
                "context": join_sentences(pre, rel, f1, tail),
                "group": "main",
                **base_common,
            }
        )
        if include_controls:
            rows.append(
                {
                    "item_id": f"{b['item_id']}__ablate_relevant",
                    "context": join_sentences(pre, f2, f1, tail),
                    "group": "ablate_relevant",
                    **base_common,
                }
            )
            rows.append(
                {
                    "item_id": f"{b['item_id']}__ablate_irrelevant",
                    "context": join_sentences(pre, rel, f2, tail),
                    "group": "ablate_irrelevant",
                    **base_common,
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disamb = generate_disamb_pairs(args.seed, n_per_item=int(args.n_per_lex_item), mode=str(args.disamb_mode))
    cf = generate_counterfactual(
        args.seed,
        include_shams=bool(args.cf_include_shams),
        include_graded=bool(getattr(args, "cf_include_graded", False)),
    )
    coh = generate_coherence(args.seed, n_items=int(args.n_coh), include_controls=bool(args.coh_include_controls))

    write_jsonl(disamb, out_dir / "disamb_pairs.jsonl")
    write_jsonl(cf, out_dir / "counterfactual.jsonl")
    write_jsonl(coh, out_dir / "coherence.jsonl")

    print(f"Wrote {len(disamb)} disamb pairs to {out_dir / 'disamb_pairs.jsonl'}")
    print(f"Wrote {len(cf)} counterfactual pairs to {out_dir / 'counterfactual.jsonl'}")
    print(f"Wrote {len(coh)} coherence items to {out_dir / 'coherence.jsonl'}")


if __name__ == "__main__":
    main()
