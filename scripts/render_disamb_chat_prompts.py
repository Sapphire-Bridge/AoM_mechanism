#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from transformers import AutoTokenizer

# Ensure repo root importability when invoked as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.prompting import normalize_prompt_boundary


IH_INLINE_RE = re.compile(
    r"^\s*System:\s*(?P<system>.*?)\s*User:\s*(?P<user>.*?)\s*(?:The assistant should)?\s*$",
    flags=re.DOTALL,
)


def _strip_assistant_should(s: str) -> Tuple[str, bool]:
    s0 = str(s)
    needle = "The assistant should"
    if s0.rstrip().endswith(needle):
        out = s0.rstrip()
        out = out[: -len(needle)].rstrip()
        return out, True
    return s0, False


def _parse_ih_inline(prompt: str) -> Tuple[str | None, str, bool]:
    """
    Parse inline IH format into (system, user, parsed_ok).

    Expected pattern:
      "System: ... User: ... (The assistant should)"
    """
    m = IH_INLINE_RE.match(str(prompt).strip())
    if m is None:
        return None, str(prompt), False
    system = str(m.group("system")).strip()
    user = str(m.group("user")).strip()
    user, _stripped = _strip_assistant_should(user)
    return system, user, True


def _render_chat(
    *,
    tokenizer,
    system_text: str | None,
    user_text: str,
    add_generation_prompt: bool,
) -> str:
    messages = []
    if system_text is not None and str(system_text).strip():
        messages.append({"role": "system", "content": str(system_text)})
    messages.append({"role": "user", "content": str(user_text)})
    rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )
    return str(rendered)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Transform DisambPair prompts into chat-format inputs. "
            "Modes: user_only (strip cloze suffix) or chat_prerendered (apply_chat_template)."
        )
    )
    p.add_argument("--in_jsonl", type=str, required=True, help="Input DisambPair JSONL.")
    p.add_argument("--out_jsonl", type=str, required=True, help="Output DisambPair JSONL.")
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["user_only", "chat_prerendered"],
        help="Prompt transformation mode.",
    )
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="",
        help="Tokenizer source (required for mode=chat_prerendered).",
    )
    p.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tokenizer loading policy (default: --local_files_only).",
    )
    p.add_argument(
        "--add_generation_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to apply_chat_template (default: --add_generation_prompt).",
    )
    p.add_argument(
        "--parse_ih_inline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse inline 'System: ... User: ...' prompts into roles when present.",
    )
    p.add_argument(
        "--ih_user_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When an inline IH prompt is detected, drop the System: text and keep only the User: content. "
            "Useful as a positive control (system removed)."
        ),
    )
    p.add_argument(
        "--strip_assistant_should",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip trailing 'The assistant should' from cloze prompts when present.",
    )
    p.add_argument(
        "--normalize_boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a single space if prompt does not end with whitespace (default: on).",
    )
    p.add_argument(
        "--require_target_present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if target substring is not present in the transformed prompt (default: on).",
    )
    p.add_argument(
        "--chat_system_role",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When possible, pass IH system text as a system role message (chat_prerendered mode).",
    )
    p.add_argument(
        "--system_in_user_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If system role is unsupported by chat_template, embed it into the user message (default: on).",
    )
    args = p.parse_args()

    if str(args.mode) == "chat_prerendered" and not str(args.tokenizer_name_or_path).strip():
        raise ValueError("--tokenizer_name_or_path is required for --mode chat_prerendered")

    tokenizer = None
    if str(args.mode) == "chat_prerendered":
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_name_or_path),
            local_files_only=bool(args.local_files_only),
        )
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template.")

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_parsed_ih_a = 0
    n_parsed_ih_b = 0
    n_stripped_a = 0
    n_stripped_b = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{in_path}:{line_no}: invalid JSON ({e})") from e
            if not isinstance(row, dict):
                raise ValueError(f"{in_path}:{line_no}: row must be JSON object")
            if not isinstance(row.get("a"), dict) or not isinstance(row.get("b"), dict):
                raise ValueError(f"{in_path}:{line_no}: missing side objects a/b")

            target = str(row.get("target", "") or "")

            def _transform_side(side_key: str) -> Tuple[str, bool, bool]:
                side = row[side_key]
                prompt = str(side.get("prompt", "") or "")
                system_text = None
                prompt_text = prompt
                parsed_ih = False

                stripped = False
                if bool(args.strip_assistant_should):
                    prompt_text, stripped = _strip_assistant_should(prompt_text)

                user_text = prompt_text
                if bool(args.parse_ih_inline):
                    system_text, user_text, parsed_ih = _parse_ih_inline(prompt_text)

                user_text = str(user_text).strip()

                user_message = user_text
                if parsed_ih and not bool(args.ih_user_only):
                    # Preserve the system text as part of the user message for comparability across
                    # render modes, and as a fallback for templates that do not support system role.
                    user_message = f"System: {system_text} User: {user_text}".strip()

                if str(args.mode) == "user_only":
                    out_prompt = user_message
                else:
                    assert tokenizer is not None
                    # Some tokenizers (e.g. Gemma) do not support system role in chat_template.
                    # Prefer system role when requested, but fall back to embedding system text
                    # in the user message if needed.
                    use_system_role = bool(args.chat_system_role) and (not bool(args.ih_user_only)) and parsed_ih
                    if use_system_role:
                        try:
                            out_prompt = _render_chat(
                                tokenizer=tokenizer,
                                system_text=system_text,
                                user_text=user_text,
                                add_generation_prompt=bool(args.add_generation_prompt),
                            )
                        except Exception:
                            if not bool(args.system_in_user_fallback):
                                raise
                            out_prompt = _render_chat(
                                tokenizer=tokenizer,
                                system_text=None,
                                user_text=user_message,
                                add_generation_prompt=bool(args.add_generation_prompt),
                            )
                    else:
                        out_prompt = _render_chat(
                            tokenizer=tokenizer,
                            system_text=None,
                            user_text=user_message,
                            add_generation_prompt=bool(args.add_generation_prompt),
                        )

                if bool(args.normalize_boundaries):
                    out_prompt = normalize_prompt_boundary(out_prompt)

                if bool(args.require_target_present) and target and target not in out_prompt:
                    raise ValueError(
                        f"{in_path}:{line_no}: target substring not found after transform "
                        f"(pair_id={row.get('pair_id')!r} side={side_key!r} target={target!r})"
                    )

                side["prompt"] = out_prompt
                return out_prompt, parsed_ih, stripped

            _a_prompt, a_parsed, a_stripped = _transform_side("a")
            _b_prompt, b_parsed, b_stripped = _transform_side("b")

            n_parsed_ih_a += int(a_parsed)
            n_parsed_ih_b += int(b_parsed)
            n_stripped_a += int(a_stripped)
            n_stripped_b += int(b_stripped)

            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(
                {
                    "prompt_transform_mode": str(args.mode),
                    "prompt_transform_parse_ih_inline": bool(args.parse_ih_inline),
                    "prompt_transform_ih_user_only": bool(args.ih_user_only),
                    "prompt_transform_strip_assistant_should": bool(args.strip_assistant_should),
                    "prompt_transform_normalize_boundaries": bool(args.normalize_boundaries),
                    "prompt_transform_require_target_present": bool(args.require_target_present),
                }
            )
            if str(args.mode) == "chat_prerendered":
                metadata.update(
                    {
                        "prompt_render_add_generation_prompt": bool(args.add_generation_prompt),
                        "prompt_render_tokenizer_source": str(args.tokenizer_name_or_path),
                        "prompt_render_chat_system_role": bool(args.chat_system_role),
                        "prompt_render_system_in_user_fallback": bool(args.system_in_user_fallback),
                    }
                )
            row["metadata"] = metadata

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

    # Validate output with canonical parser/validator.
    validated = load_disamb_pairs(str(out_path), validate=True)
    print(f"Wrote {len(validated)} pairs -> {out_path}")
    print(f"IH parsed sides: a={n_parsed_ih_a} b={n_parsed_ih_b} (rows={n_rows})")
    print(f"Stripped 'The assistant should': a={n_stripped_a} b={n_stripped_b} (rows={n_rows})")
    print("Validation: OK (load_disamb_pairs(validate=True))")


if __name__ == "__main__":
    main()
