#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve_mode_and_device() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "cuda_validated", "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "m1max_safe", "mps"
    raise RuntimeError("No accelerator available. Expected CUDA or MPS.")


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    try:
        mode, device = _resolve_mode_and_device()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(ROOT / "scripts" / "run_paper.py"), mode, *forwarded]
    print(f"[info] accelerator_device={device} paper_mode={mode}")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
