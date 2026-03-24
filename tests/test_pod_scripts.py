from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POD_SCRIPTS = (
    ROOT / "scripts" / "pod_env.sh",
    ROOT / "scripts" / "pod_after_success.sh",
    ROOT / "scripts" / "pod_run_one_result_gpu.sh",
    ROOT / "scripts" / "pod_run_all_results_gpu.sh",
    ROOT / "scripts" / "pod_run_paper_cpu.sh",
)


def _make_fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / ".venv" / "bin" / "activate").write_text("export VIRTUAL_ENV=\"$PWD/.venv\"\n", encoding="utf-8")
    fake_python = repo / ".venv" / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "outdir=\"\"\n"
        "prev=\"\"\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$prev\" == \"--run_root\" || \"$prev\" == \"--results_dir\" ]]; then\n"
        "    outdir=\"$arg\"\n"
        "  fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "if [[ -n \"$outdir\" ]]; then\n"
        "  mkdir -p \"$outdir\"\n"
        "  printf 'ok\\n' > \"$outdir/done.txt\"\n"
        "fi\n"
        "printf '%s\n' \"$@\" > \"$REPO_ROOT/fake_args.txt\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    fake_runpodctl = repo / ".venv" / "bin" / "runpodctl"
    fake_runpodctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\n' \"$@\" > \"$REPO_ROOT/fake_args.txt\"\n",
        encoding="utf-8",
    )
    fake_runpodctl.chmod(fake_runpodctl.stat().st_mode | stat.S_IXUSR)
    for name in ("run_one_result_check.py", "run_paper.py", "run_mom_paper.py"):
        (repo / "scripts" / name).write_text("# placeholder\n", encoding="utf-8")
    for name in ("pod_env.sh", "pod_after_success.sh"):
        target = repo / "scripts" / name
        target.write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return repo


def _wait_for(path: Path, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        # The fake background shims only write tiny files; existence is enough here.
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


def _wait_for_text(path: Path, needle: str, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and needle in path.read_text(encoding="utf-8"):
            return
        time.sleep(0.05)
    contents = path.read_text(encoding="utf-8") if path.exists() else "<missing>"
    raise AssertionError(f"Timed out waiting for {path} to contain {needle!r}; last contents: {contents!r}")


def test_pod_scripts_have_valid_shell_syntax() -> None:
    for script in POD_SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True, cwd=str(ROOT))


def test_pod_env_derives_repo_root_from_script_location(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    script_path = repo / "scripts" / "pod_env.sh"
    script_path.write_text((ROOT / "scripts" / "pod_env.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        ["bash", "-lc", f'source "{script_path}"; printf "%s" "$REPO_ROOT"'],
        check=True,
        cwd=str(repo),
        capture_output=True,
        text=True,
    )

    assert proc.stdout == str(repo)


def test_pod_run_one_result_gpu_strict_local_files_only_boolean(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    run_root = repo / "results" / "one_result_gpu"
    status_path = Path(f"{run_root}.status")
    archive_path = Path(f"{run_root}.tar.gz")
    args_path = repo / "fake_args.txt"

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo)
    env["LOCAL_FILES_ONLY"] = "0"
    env["RUNPOD_POST_SUCCESS_ACTION"] = "none"

    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "pod_run_one_result_gpu.sh"), str(run_root)],
        check=True,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    _wait_for(args_path)
    _wait_for(status_path)
    _wait_for(archive_path)
    _wait_for_text(status_path, "SUCCESS")
    args = args_path.read_text(encoding="utf-8").splitlines()

    assert "--local_files_only" not in args
    assert "SUCCESS" in status_path.read_text(encoding="utf-8")
    assert f"log: {run_root}.log" in proc.stdout
    assert f"pid_file: {run_root}.pid" in proc.stdout
    assert f"archive: {run_root}.tar.gz" in proc.stdout


def test_pod_run_all_results_gpu_appends_local_files_only_only_when_enabled(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    results_dir = repo / "results" / "paper_a100"
    status_path = Path(f"{results_dir}.status")
    archive_path = Path(f"{results_dir}.tar.gz")
    args_path = repo / "fake_args.txt"

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo)
    env["LOCAL_FILES_ONLY"] = "1"
    env["RUNPOD_POST_SUCCESS_ACTION"] = "none"

    subprocess.run(
        ["bash", str(ROOT / "scripts" / "pod_run_all_results_gpu.sh"), str(results_dir)],
        check=True,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    _wait_for(args_path)
    _wait_for(status_path)
    _wait_for(archive_path)
    _wait_for_text(status_path, "SUCCESS")
    args = args_path.read_text(encoding="utf-8").splitlines()

    assert args[:2] == ["scripts/run_paper_accelerated.py", "--results_dir"]
    assert "--local_files_only" in args
    assert "SUCCESS" in status_path.read_text(encoding="utf-8")


def test_pod_run_paper_cpu_rejects_nonempty_run_root(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    run_root = repo / "results" / "mom_paper_cpu"
    run_root.mkdir(parents=True)
    (run_root / "stale.txt").write_text("stale\n", encoding="utf-8")

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo)

    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "pod_run_paper_cpu.sh"), str(run_root)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "Directory must start empty" in proc.stderr


def test_pod_env_fails_clearly_when_venv_is_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("run_one_result_check.py", "run_paper.py", "run_mom_paper.py"):
        (repo / "scripts" / name).write_text("# placeholder\n", encoding="utf-8")
    script_path = repo / "scripts" / "pod_env.sh"
    script_path.write_text((ROOT / "scripts" / "pod_env.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    proc = subprocess.run(
        ["bash", "-lc", f'source "{script_path}"'],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "Missing virtualenv activate script" in proc.stderr


def test_pod_after_success_writes_failed_status_and_skips_archive(tmp_path: Path) -> None:
    repo = _make_fake_repo(tmp_path)
    result_path = repo / "results" / "failed_run"
    archive_path = Path(f"{result_path}.tar.gz")
    status_path = Path(f"{result_path}.status")
    pid_path = Path(f"{result_path}.pid")
    fail_script = repo / "scripts" / "fail.sh"
    fail_script.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    fail_script.chmod(fail_script.stat().st_mode | stat.S_IXUSR)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("12345\n", encoding="utf-8")

    env = os.environ.copy()
    env["REPO_ROOT"] = str(repo)
    env["RUNPOD_POST_SUCCESS_ACTION"] = "none"

    proc = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "pod_after_success.sh"),
            "--result-path",
            str(result_path),
            "--archive-path",
            str(archive_path),
            "--status-path",
            str(status_path),
            "--pid-path",
            str(pid_path),
            "--",
            str(fail_script),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 7
    assert "FAILED: command exit 7" in status_path.read_text(encoding="utf-8")
    assert not archive_path.exists()
    assert not pid_path.exists()
