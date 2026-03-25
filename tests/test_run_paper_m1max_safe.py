from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from scripts import run_paper


ROOT = Path(__file__).resolve().parents[1]


def test_m1max_safe_dry_run_emits_per_model_behavioral_commands(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_paper.py"),
            "m1max_safe",
            "--dry_run",
            "--skip_dataset",
            "--data_dir",
            str(data_dir),
            "--results_dir",
            str(results_dir),
        ],
        check=True,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    stdout = proc.stdout
    assert "aom_eval.py --models gpt2" in stdout
    assert "aom_eval.py --models Qwen/Qwen2.5-0.5B" in stdout
    assert "--attn_implementation eager" in stdout
    assert "--torch_dtype float32" in stdout
    assert "behavioral/gpt2.csv" in stdout
    assert "behavioral/Qwen_Qwen2.5-0.5B.csv" in stdout


def test_run_with_timeout_kills_process_group() -> None:
    marker = f"timeout_marker_{int(time.time() * 1000)}"
    cmd = [
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        marker,
    ]

    outcome = run_paper._run_with_timeout(cmd, cwd=ROOT, dry_run=False, timeout_seconds=1)

    assert outcome.timed_out is True
    assert outcome.returncode == 124

    time.sleep(0.5)
    ps = subprocess.run(
        [
            "bash",
            "-lc",
            f"ps -axo command | grep {marker!r} | grep -v grep",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert ps.stdout.strip() == ""
