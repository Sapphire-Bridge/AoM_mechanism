from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_mom_paper_dry_run_emits_core_and_sae_support_commands(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "mom-paper"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_mom_paper.py"),
        "--dry_run",
        "--run_root",
        str(run_root),
        "--local_files_only",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    out = proc.stdout
    assert "scripts/clt_raw_comparability.py" in out
    assert "--device cpu" in out
    assert "--torch_dtype float32" in out
    assert "--run_clt_patching" in out
    assert "--run_sae_patching" in out
    assert "--run_patching_specificity" in out
    assert "--patch_layers 4,8,12,16,20,24" in out
    assert "gemma2b_sae.csv" in out
