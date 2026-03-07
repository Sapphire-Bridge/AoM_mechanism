from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any, Literal


def validate_prompt_boundary(prompt: str, *, mode: str = "raw") -> None:
    """
    Warn if `prompt` does not end in whitespace/newline.

    When scoring prompt and continuation separately, missing trailing whitespace can change
    tokenization around the boundary (sometimes called "token healing").
    """
    if not prompt:
        return
    if prompt[-1].isspace():
        return
    warnings.warn(
        f"Prompt boundary may be unstable (mode={mode}): prompt does not end with whitespace.",
        UserWarning,
        stacklevel=2,
    )


def normalize_prompt_boundary(prompt: str) -> str:
    """
    If `prompt` ends with neither whitespace nor newline, append a single space.
    """
    if not prompt:
        return prompt
    if prompt[-1].isspace():
        return prompt
    return prompt + " "


def sha256_text(s: str | None) -> str | None:
    if s is None:
        return None
    h = hashlib.sha256()
    h.update(str(s).encode("utf-8"))
    return h.hexdigest()


def sha256_chat_template(tokenizer: Any) -> str | None:
    """
    Hash `tokenizer.chat_template` (which may be str | dict | None).

    This is best-effort and must not crash if tokenizers omit this attribute.
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl is None:
        return None
    if isinstance(tmpl, str):
        return sha256_text(tmpl)
    if isinstance(tmpl, dict):
        try:
            s = json.dumps(tmpl, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            warnings.warn(
                "chat_template contains non-JSON-serializable values; using repr() fallback (may be less stable).",
                UserWarning,
                stacklevel=2,
            )
            s = repr(sorted(tmpl.items(), key=lambda kv: str(kv[0])))
        return sha256_text(s)
    return sha256_text(str(tmpl))


@dataclass(frozen=True)
class PromptRenderConfig:
    prompt_mode: Literal["raw", "chat_template"] = "raw"
    system_prompt: str | None = None
    add_generation_prompt: bool = True


def render_prompt(tokenizer: Any, user_text: str, cfg: PromptRenderConfig) -> tuple[str, dict[str, Any]]:
    """
    Render a prompt from user text under a single explicit pathway.

    Security posture: the returned meta contains *hashes only* (no raw prompt/system text).
    """
    if cfg.prompt_mode == "raw":
        rendered = str(user_text)
    elif cfg.prompt_mode == "chat_template":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template (required for prompt_mode=chat_template).")
        messages = []
        if cfg.system_prompt is not None:
            messages.append({"role": "system", "content": str(cfg.system_prompt)})
        messages.append({"role": "user", "content": str(user_text)})
        rendered = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=False,
            add_generation_prompt=bool(cfg.add_generation_prompt),
        )
    else:
        raise ValueError(f"Unknown prompt_mode: {cfg.prompt_mode!r}")

    meta = {
        "system_prompt_sha256": sha256_text(cfg.system_prompt),
        "chat_template_sha256": sha256_chat_template(tokenizer),
        "rendered_prompt_sha256": sha256_text(rendered),
    }
    return str(rendered), meta
