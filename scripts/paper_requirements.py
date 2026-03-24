#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PAPER_MODEL_REPO_ID = "google/gemma-2-2b"
PAPER_MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
PAPER_SCOPE_REPO_ID = "google/gemma-scope-2b-pt-res"
PAPER_SCOPE_REVISION = "fd571b47c1c64851e9b1989792367b9babb4af63"
PAPER_CLT_BUNDLE_PATH = Path("clt_bundles/gemma-scope-2b-pt-res_sweep_smoke")
PAPER_CLT_REQUIRED_RUNS = {
    4: "average_l0_60",
    8: "average_l0_71",
    12: "average_l0_176",
    16: "average_l0_78",
    20: "average_l0_71",
    24: "average_l0_73",
}
PAPER_SIX_LAYER_PROFILE_LAYERS = tuple(sorted(PAPER_CLT_REQUIRED_RUNS))
PAPER_FIXED_LAYER_SAE_LAYER = 24
PAPER_FIXED_LAYER_SAE_RUN_NAME = "average_l0_457"


def resolve_cached_snapshot(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import scan_cache_dir

    target_revision = str(revision or "").strip()
    for repo in scan_cache_dir().repos:
        if repo.repo_id != repo_id:
            continue
        if target_revision:
            for cached_revision in repo.revisions:
                if cached_revision.commit_hash == target_revision:
                    return Path(str(cached_revision.snapshot_path))
            raise FileNotFoundError(f"Cached revision {target_revision!r} not found for {repo_id}")
        if repo.revisions:
            return Path(str(sorted(repo.revisions, key=lambda item: item.commit_hash)[-1].snapshot_path))
    raise FileNotFoundError(f"No cached snapshot found for {repo_id}")


def required_scope_params_paths() -> tuple[Path, ...]:
    return tuple(
        Path(f"layer_{int(layer)}") / "width_16k" / str(run_name) / "params.npz"
        for layer, run_name in sorted(PAPER_CLT_REQUIRED_RUNS.items())
    )


def representative_scope_params_path() -> Path:
    return required_scope_params_paths()[0]


def fixed_layer_sae_params_path() -> Path:
    return (
        Path(f"layer_{int(PAPER_FIXED_LAYER_SAE_LAYER)}")
        / "width_16k"
        / str(PAPER_FIXED_LAYER_SAE_RUN_NAME)
        / "params.npz"
    )
