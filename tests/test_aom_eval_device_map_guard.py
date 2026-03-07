from __future__ import annotations

import argparse

import pytest

from aom_eval import run_once


def test_run_once_rejects_device_map_with_patching() -> None:
    args = argparse.Namespace(
        device_map="auto",
        run_patching=True,
        run_sae_patching=False,
        run_clt_patching=False,
        run_patching_specificity=False,
    )
    with pytest.raises(ValueError, match="device_map"):
        _ = run_once(args, seed=0)
