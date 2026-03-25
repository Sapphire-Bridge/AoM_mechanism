from __future__ import annotations

import subprocess

from scripts import paper_requirements, reviewer_assets, reviewer_quickcheck, run_mom_paper


def test_quickcheck_and_runner_use_shared_paper_requirements() -> None:
    assert reviewer_quickcheck.MODEL_REVISION == paper_requirements.PAPER_MODEL_REVISION
    assert reviewer_quickcheck.README_CORE_BUNDLE_REVISION == paper_requirements.PAPER_SCOPE_REVISION
    assert reviewer_quickcheck.README_CORE_BUNDLE_PATH == paper_requirements.PAPER_CLT_BUNDLE_PATH
    assert reviewer_assets.MODEL_REVISION == paper_requirements.PAPER_MODEL_REVISION
    assert reviewer_assets.README_CORE_BUNDLE_PATH == paper_requirements.PAPER_CLT_BUNDLE_PATH
    assert run_mom_paper.MODEL_REVISION == paper_requirements.PAPER_MODEL_REVISION
    assert run_mom_paper.README_CORE_BUNDLE_PATH == paper_requirements.PAPER_CLT_BUNDLE_PATH


def test_run_step_timeout_returns_distinct_status(tmp_path, monkeypatch) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["python", "foo.py"], timeout=7, output="partial stdout", stderr="partial stderr")

    monkeypatch.setattr(reviewer_quickcheck.subprocess, "run", _raise_timeout)

    result = reviewer_quickcheck._run_step(
        name="timeout_case",
        argv=("python", "foo.py"),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        timeout_seconds=7,
    )

    assert result.status == reviewer_quickcheck.TIMEOUT
    assert "timed out after 7s" in result.detail
    assert (tmp_path / "logs" / "timeout_case.stdout.log").exists()
    assert (tmp_path / "logs" / "timeout_case.stderr.log").exists()


def test_timeout_default_can_come_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MOM_REVIEWER_SMOKE_TIMEOUT_SECONDS", "321")
    assert reviewer_quickcheck._timeout_default("MOM_REVIEWER_SMOKE_TIMEOUT_SECONDS", 180) == 321


def test_next_commands_snippet_uses_public_make_targets(monkeypatch) -> None:
    monkeypatch.setattr(reviewer_quickcheck, "_best_accelerator", lambda: "cuda")
    snippet = reviewer_quickcheck._next_commands_snippet()
    assert "make one-result-check" in snippet
    assert "make one-result-check-gpu" in snippet
    assert 'make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"' in snippet


def test_next_commands_snippet_prompts_asset_bootstrap_when_assets_missing() -> None:
    snippet = reviewer_quickcheck._next_commands_snippet(
        [
            reviewer_quickcheck.StepResult(
                name="local_model_cache",
                status=reviewer_quickcheck.FAILURE,
                detail="missing model cache",
            )
        ]
    )
    assert snippet == "make reviewer-assets\nmake reviewer-check"


def test_smoke_command_skips_repo_dataset_generation(tmp_path) -> None:
    cmd = reviewer_quickcheck._smoke_command(tmp_path)
    assert "--skip_dataset" in cmd
