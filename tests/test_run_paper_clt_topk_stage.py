from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_paper_smoke_dry_run_emits_clt_topk_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "paper_smoke_clt_topk_stage"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_paper.py"),
        "smoke",
        "--dry_run",
        "--results_dir",
        str(out_dir),
        "--clt_topk_recovery",
        "--clt_repo",
        "/tmp/clt_bundle",
        "--clt_topk_layers",
        "0",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    stdout = proc.stdout

    assert "aom_clt_topk_recovery.py" in stdout
    assert "clt_topk_recovery.csv" in stdout
    assert "--clt_repo /tmp/clt_bundle" in stdout
