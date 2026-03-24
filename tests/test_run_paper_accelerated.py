from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import run_paper_accelerated


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_mode_and_device_prefers_cuda(monkeypatch) -> None:
    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _MPS()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    assert run_paper_accelerated._resolve_mode_and_device() == ("a100", "cuda")


def test_resolve_mode_and_device_uses_mps_when_cuda_missing(monkeypatch) -> None:
    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return True

    class _Backends:
        mps = _MPS()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    assert run_paper_accelerated._resolve_mode_and_device() == ("m1max_safe", "mps")


def test_main_errors_when_no_accelerator(monkeypatch, capsys) -> None:
    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Backends:
        mps = _MPS()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

    monkeypatch.setitem(sys.modules, "torch", _Torch())
    code = run_paper_accelerated.main([])

    captured = capsys.readouterr()
    assert code == 2
    assert "No accelerator available" in captured.err


def test_main_forwards_args_to_run_paper(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_paper_accelerated, "_resolve_mode_and_device", lambda: ("m1max_safe", "mps"))

    recorded: dict[str, object] = {}

    def fake_run(cmd, cwd, check):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        recorded["check"] = check

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    code = run_paper_accelerated.main(["--dry_run", "--results_dir", "/tmp/out"])

    captured = capsys.readouterr()
    assert code == 0
    assert "[info] accelerator_device=mps paper_mode=m1max_safe" in captured.out
    assert recorded["cmd"] == [
        sys.executable,
        str(ROOT / "scripts" / "run_paper.py"),
        "m1max_safe",
        "--dry_run",
        "--results_dir",
        "/tmp/out",
    ]
