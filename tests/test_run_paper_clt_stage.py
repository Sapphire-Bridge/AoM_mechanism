from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_paper_smoke_dry_run_emits_dedicated_clt_stage(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "paper_smoke_clt_stage"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_paper.py"),
        "smoke",
        "--dry_run",
        "--results_dir",
        str(out_dir),
        "--run_clt_stage",
        "--clt_repo",
        "/tmp/clt_bundle",
        "--clt_layers",
        "0",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), check=True, capture_output=True, text=True)
    stdout = proc.stdout

    # Primary AoM command plus a dedicated CLT-stage AoM command.
    assert stdout.count("aom_eval.py") == 2
    assert "clt_cpt_disamb_only.csv" in stdout
    assert "--run_clt_patching" in stdout
    assert "--clt_repo /tmp/clt_bundle" in stdout
