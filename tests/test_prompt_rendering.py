from __future__ import annotations

from argparse import Namespace

import pytest

from aom.prompting import PromptRenderConfig, render_prompt, sha256_chat_template, sha256_text


def test_sha256_chat_template_handles_str_dict_none():
    class T:
        chat_template = None

    assert sha256_chat_template(T()) is None

    class T2:
        chat_template = "{% for m in messages %}{{ m }}{% endfor %}"

    assert sha256_chat_template(T2()) == sha256_text(T2.chat_template)

    class T3:
        chat_template = {"b": 2, "a": 1}

    class T4:
        chat_template = {"a": 1, "b": 2}

    assert sha256_chat_template(T3()) == sha256_chat_template(T4())


def test_render_prompt_raw_is_identity_and_hashes():
    class Tok:
        chat_template = None

    rendered, meta = render_prompt(Tok(), "Hello", PromptRenderConfig(prompt_mode="raw", system_prompt=None))
    assert rendered == "Hello"
    assert meta["rendered_prompt_sha256"] == sha256_text("Hello")
    assert meta["system_prompt_sha256"] is None


def test_render_prompt_chat_template_calls_apply_chat_template():
    class Tok:
        chat_template = {"template": "dummy"}

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):  # noqa: ANN001
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                }
            )
            sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            user_msg = messages[-1]["content"] if messages else ""
            suffix = "<GEN>" if add_generation_prompt else ""
            return f"S:{sys_msg} U:{user_msg}{suffix}"

    tok = Tok()
    cfg = PromptRenderConfig(prompt_mode="chat_template", system_prompt="SYS", add_generation_prompt=True)
    rendered, meta = render_prompt(tok, "USER", cfg)
    assert rendered == "S:SYS U:USER<GEN>"
    assert tok.calls and tok.calls[0]["tokenize"] is False
    assert tok.calls[0]["add_generation_prompt"] is True
    assert tok.calls[0]["messages"][0]["role"] == "system"
    assert tok.calls[0]["messages"][-1]["role"] == "user"
    assert meta["system_prompt_sha256"] == sha256_text("SYS")
    assert meta["chat_template_sha256"] == sha256_chat_template(tok)


def test_double_render_guard_raises():
    from aom_eval import _validate_prompt_rendering_args

    args = Namespace(prompt_input="full_prompt", prompt_mode="chat_template")
    with pytest.raises(ValueError, match="chat_template mode expects user_message inputs"):
        _validate_prompt_rendering_args(args)

