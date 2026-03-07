from __future__ import annotations

import pytest

from aom.prompting import normalize_prompt_boundary, validate_prompt_boundary


def test_prompt_boundary_validation_warns():
    with pytest.warns(UserWarning, match="Prompt boundary may be unstable"):
        validate_prompt_boundary("Hello", mode="raw")


def test_prompt_boundary_validation_no_warn_for_space_or_newline():
    validate_prompt_boundary("Hello ", mode="raw")
    validate_prompt_boundary("Hello\n", mode="raw")


def test_normalize_prompt_boundary_appends_one_space_only():
    assert normalize_prompt_boundary("Hello") == "Hello "
    assert normalize_prompt_boundary("Hello ") == "Hello "
    assert normalize_prompt_boundary("Hello\n") == "Hello\n"

