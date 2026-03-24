#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_requirements import (
    PAPER_CLT_BUNDLE_PATH,
    PAPER_FIXED_LAYER_SAE_RUN_NAME,
    PAPER_FIXED_LAYER_SAE_LAYER,
    PAPER_MODEL_REPO_ID,
    PAPER_MODEL_REVISION,
    PAPER_SCOPE_REPO_ID,
    PAPER_SCOPE_REVISION,
    fixed_layer_sae_params_path,
    required_scope_params_paths,
    resolve_cached_snapshot,
)


MODEL_REPO_ID = PAPER_MODEL_REPO_ID
MODEL_REVISION = PAPER_MODEL_REVISION
SCOPE_REPO_ID = PAPER_SCOPE_REPO_ID
SCOPE_REVISION = PAPER_SCOPE_REVISION
README_CORE_BUNDLE_PATH = PAPER_CLT_BUNDLE_PATH


@dataclass(frozen=True)
class AssetCheckResult:
    name: str
    ok: bool
    detail: str


def _snapshot_has_any(snapshot: Path, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if next(snapshot.glob(pattern), None) is not None:
            return True
    return False


def _read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _read_bytes(path: Path, *, n_bytes: int = 64) -> None:
    with path.open("rb") as handle:
        sample = handle.read(n_bytes)
    if not sample:
        raise ValueError(f"{path} is empty")


def probe_tokenizer(snapshot: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    encoded = tokenizer.encode("AoM reviewer quickcheck")
    if not encoded:
        raise ValueError(f"Tokenizer at {snapshot} produced an empty encoding")


def probe_weight_file(snapshot: Path) -> Path:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} missing a non-empty weight_map")
        weight_names = sorted({str(value) for value in weight_map.values()})
        missing = [name for name in weight_names if not (snapshot / name).exists()]
        if missing:
            sample = ", ".join(missing[:3])
            if len(missing) > 3:
                sample += ", ..."
            raise FileNotFoundError(f"{index_path} referenced missing weight shards: {sample}")
        probe_path = snapshot / weight_names[0]
        _read_bytes(probe_path)
        return probe_path

    for pattern in ("model.safetensors", "model-*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin"):
        candidate = next(snapshot.glob(pattern), None)
        if candidate is not None:
            _read_bytes(candidate)
            return candidate
    raise FileNotFoundError(f"No readable model weight file found under {snapshot}")


def probe_npz(path: Path) -> None:
    import numpy as np

    with np.load(str(path), allow_pickle=False) as payload:
        if not payload.files:
            raise ValueError(f"{path} contained no arrays")


def required_params_tree_ready(base: Path, *, label: str) -> tuple[bool, str]:
    required = required_scope_params_paths()
    missing = [rel for rel in required if not (base / rel).exists()]
    if missing:
        sample = ", ".join(str(rel) for rel in missing[:3])
        if len(missing) > 3:
            sample += ", ..."
        return False, f"{label} missing required CLT params.npz files: {sample}"
    for rel in required:
        probe_npz(base / rel)
    return True, f"{label} ready; verified readable params for {len(required)} required CLT files"


def model_snapshot_ready(snapshot: Path) -> tuple[bool, str]:
    try:
        config_path = snapshot / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        _read_json(config_path)
        tokenizer_cfg = snapshot / "tokenizer_config.json"
        if tokenizer_cfg.exists():
            _read_json(tokenizer_cfg)
        if not _snapshot_has_any(snapshot, ("tokenizer.model", "tokenizer.json", "tokenizer_config.json")):
            raise FileNotFoundError(f"No tokenizer files found under {snapshot}")
        probe_tokenizer(snapshot)
        weight_path = probe_weight_file(snapshot)
    except Exception as exc:
        return False, f"cached model snapshot incomplete at {snapshot}; {exc}"
    return True, f"cached model snapshot ready at {snapshot}; verified config/tokenizer/weights via {weight_path.name}"


def scope_snapshot_bundle_source_ready(snapshot: Path) -> tuple[bool, str]:
    try:
        ok, detail = required_params_tree_ready(snapshot, label="cached Gemma Scope snapshot")
    except Exception as exc:
        return False, f"cached Gemma Scope snapshot unreadable for CLT bundle materialization at {snapshot}; {exc}"
    return ok, detail


def scope_snapshot_fixed_layer_sae_ready(snapshot: Path) -> tuple[bool, str]:
    sae_rel = fixed_layer_sae_params_path()
    sae_path = snapshot / sae_rel
    if not sae_path.exists():
        return False, f"cached Gemma Scope snapshot missing fixed-layer SAE params.npz file: {sae_rel}"
    try:
        probe_npz(sae_path)
    except Exception as exc:
        return False, f"cached Gemma Scope snapshot has unreadable fixed-layer SAE params.npz file at {sae_rel}; {exc}"
    return True, f"cached Gemma Scope snapshot contains readable fixed-layer SAE support at {sae_path}"


def bundle_or_cache_ready(bundle_path: Path, snapshot: Path) -> tuple[bool, str]:
    if bundle_path.exists():
        try:
            ok, detail = required_params_tree_ready(bundle_path, label="CLT bundle")
        except Exception as exc:
            return False, f"CLT bundle present but unreadable at {bundle_path}; {exc}"
        return ok, detail
    ok, detail = scope_snapshot_bundle_source_ready(snapshot)
    if ok:
        return True, f"CLT bundle absent, but cached scope snapshot is ready for auto-materialization from {snapshot}"
    return False, f"CLT bundle absent and source cache is incomplete; {detail}"


def local_asset_results() -> list[AssetCheckResult]:
    results: list[AssetCheckResult] = []

    try:
        model_snapshot = resolve_cached_snapshot(MODEL_REPO_ID, MODEL_REVISION)
        ok, detail = model_snapshot_ready(model_snapshot)
    except Exception as exc:
        ok = False
        detail = f"cached model snapshot missing for {MODEL_REPO_ID}@{MODEL_REVISION}: {exc}"
    results.append(AssetCheckResult(name="local_model_cache", ok=ok, detail=detail))

    try:
        scope_snapshot = resolve_cached_snapshot(SCOPE_REPO_ID, SCOPE_REVISION)
    except Exception as exc:
        scope_snapshot = None
        results.append(
            AssetCheckResult(
                name="local_scope_cache",
                ok=False,
                detail=f"cached Gemma Scope snapshot missing for {SCOPE_REPO_ID}@{SCOPE_REVISION}: {exc}",
            )
        )
    else:
        bundle_ok, bundle_detail = bundle_or_cache_ready(ROOT / README_CORE_BUNDLE_PATH, scope_snapshot)
        results.append(AssetCheckResult(name="local_clt_bundle_or_cache", ok=bundle_ok, detail=bundle_detail))
        sae_ok, sae_detail = scope_snapshot_fixed_layer_sae_ready(scope_snapshot)
        results.append(AssetCheckResult(name="local_sae_cache", ok=sae_ok, detail=sae_detail))

    return results


def _friendly_hf_error(*, repo_id: str, revision: str | None, exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if any(token in lowered for token in ("401", "403", "forbidden", "unauthorized", "gated", "access denied")):
        hf_home = os.environ.get("HF_HOME")
        cli_path = Path(sys.executable).with_name("huggingface-cli")
        login_cmd = str(cli_path) if cli_path.exists() else "huggingface-cli"
        guidance_parts = []
        if hf_home:
            guidance_parts.append(f"HF_HOME={hf_home}")
            guidance_parts.append(f"If this is a fresh isolated cache, run: {login_cmd} login")
        else:
            guidance_parts.append(f"Run: {login_cmd} login")
        guidance_parts.append("Then rerun: make reviewer-assets")
        return (
            f"could not access {repo_id}@{revision}; check Hugging Face login/token and confirm "
            f"the required model license is accepted; {'; '.join(guidance_parts)}; raw_error={text}"
        )
    return f"could not prepare cache for {repo_id}@{revision}; raw_error={text}"


def prepare_model_snapshot() -> Path:
    try:
        snapshot_path = Path(str(snapshot_download(repo_id=MODEL_REPO_ID, revision=MODEL_REVISION)))
    except Exception as exc:
        raise RuntimeError(_friendly_hf_error(repo_id=MODEL_REPO_ID, revision=MODEL_REVISION, exc=exc)) from exc
    resolved = resolve_cached_snapshot(MODEL_REPO_ID, MODEL_REVISION)
    return resolved if resolved.exists() else snapshot_path


def prepare_fixed_layer_sae_cache() -> Path:
    rel_dir = Path(f"layer_{int(PAPER_FIXED_LAYER_SAE_LAYER)}") / "width_16k" / str(PAPER_FIXED_LAYER_SAE_RUN_NAME)
    params_rel = rel_dir / "params.npz"
    cfg_rel = rel_dir / "cfg.json"
    try:
        params_path = Path(
            str(
                hf_hub_download(
                    repo_id=SCOPE_REPO_ID,
                    filename=str(params_rel),
                    revision=SCOPE_REVISION,
                )
            )
        )
    except Exception as exc:
        raise RuntimeError(_friendly_hf_error(repo_id=SCOPE_REPO_ID, revision=SCOPE_REVISION, exc=exc)) from exc

    try:
        hf_hub_download(
            repo_id=SCOPE_REPO_ID,
            filename=str(cfg_rel),
            revision=SCOPE_REVISION,
        )
    except EntryNotFoundError:
        pass

    return params_path


def build_clt_bundle_materialization_command(*, python_executable: str | None = None) -> tuple[str, ...]:
    return (
        python_executable or sys.executable,
        str(ROOT / "scripts" / "gemma_scope_to_clt.py"),
        "--preset",
        "readme_core_bundle",
        "--width",
        "16k",
        "--revision",
        SCOPE_REVISION,
        "--out_dir",
        str(ROOT / README_CORE_BUNDLE_PATH),
    )


def materialize_reviewer_clt_bundle(*, python_executable: str | None = None) -> Path:
    cmd = build_clt_bundle_materialization_command(python_executable=python_executable)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    if int(proc.returncode) != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        snippet = " | ".join(line.strip() for line in tail if line.strip()) or "no stderr/stdout captured"
        raise RuntimeError(
            f"failed to materialize reviewer CLT bundle via gemma_scope_to_clt.py; exit_code={int(proc.returncode)}; "
            f"tail={snippet}"
        )
    return ROOT / README_CORE_BUNDLE_PATH
