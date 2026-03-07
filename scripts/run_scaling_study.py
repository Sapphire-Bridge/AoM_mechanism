from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.config import load_config
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.scaling.aggregate import aggregate_scaling_outputs
from aom.scaling.layer_map import LayerMapResult, map_layers
from scripts.run_paper import (
    _aom_eval_cmd,
    _cf_patching_cmd,
    _coh_patching_cmd,
    _completeness_cmd,
    _disamb_path_decomp_cmd,
    _feature_families_cmd,
    _why_fetch_cmd,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _shlex_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in argv)


def _run(argv: List[str], *, cwd: Path, dry_run: bool) -> None:
    print(_shlex_join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, cwd=str(cwd), check=True)


def _try_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_label(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _resolve_rel_path(raw: str, *, base_dir: Path) -> str:
    p = Path(str(raw))
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return bool(default)


def _read_model_n_layers(
    *,
    model_name_or_path: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Optional[int]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            str(model_name_or_path),
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )
    except Exception:
        return None

    for key in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_name_or_path: str
    backend: str
    device: str
    torch_dtype: Optional[str]
    attn_implementation: str
    local_files_only: bool
    trust_remote_code: bool
    n_layers: Optional[int]


@dataclass(frozen=True)
class StudyConfig:
    study_name: str
    tasks: tuple[str, ...]
    bootstrap_n: int
    bootstrap_seed: int
    ci: float
    models: tuple[ModelSpec, ...]
    data_paths: Dict[str, str]
    battery: Dict[str, bool]
    layer_map_strategy: str
    layer_map_positions: tuple[float, ...]
    why_fetch_n_examples: int
    why_fetch_heads_topk: int
    disamb_max_pairs: int
    disamb_mode: str
    disamb_position: int
    completeness_explanation_path: str
    sae_repo: str
    sae_layer: int
    sae_width: str
    sae_run_name: Optional[str]
    sae_l0_target: Optional[int]
    sae_scale: float
    sae_n_features: int
    sae_max_pairs: int
    sae_similarity_threshold: float
    sae_null_n: int


@dataclass(frozen=True)
class ModelRunResult:
    model_label: str
    model_name_or_path: str
    run_dir: str
    run_manifest_path: str
    layer_map_layers: tuple[int, ...]


def _parse_config(path: Path) -> StudyConfig:
    cfg = load_config(path)
    root = cfg.get("scaling", cfg)
    if not isinstance(root, Mapping):
        raise ValueError("Scaling config must be a mapping/object")

    study_name = str(root.get("study_name", "aom_scaling_v1")).strip() or "aom_scaling_v1"
    tasks = tuple(str(t).strip() for t in list(root.get("tasks", ["disamb", "cf", "coh"])) if str(t).strip())
    if not tasks:
        raise ValueError("scaling.tasks must be non-empty")

    bootstrap_n = int(root.get("bootstrap_n", 500))
    bootstrap_seed = int(root.get("bootstrap_seed", 42))
    ci = float(root.get("ci", 0.95))

    data = root.get("data", {})
    if not isinstance(data, Mapping):
        raise ValueError("scaling.data must be a mapping")
    base = path.resolve().parent
    data_paths = {
        "disamb": _resolve_rel_path(str(data.get("disamb_path", "data/disamb_pairs.jsonl")), base_dir=base),
        "cf": _resolve_rel_path(str(data.get("cf_path", "data/counterfactual.jsonl")), base_dir=base),
        "coh": _resolve_rel_path(str(data.get("coh_path", "data/coherence.jsonl")), base_dir=base),
    }

    battery_raw = root.get("battery", {})
    if not isinstance(battery_raw, Mapping):
        raise ValueError("scaling.battery must be a mapping")
    battery = {
        "eval": _as_bool(battery_raw.get("eval", True), True),
        "patching": _as_bool(battery_raw.get("patching", True), True),
        "sae_sterility": _as_bool(battery_raw.get("sae_sterility", False), False),
        "feature_families": _as_bool(battery_raw.get("feature_families", False), False),
        "disamb_path_decomp": _as_bool(battery_raw.get("disamb_path_decomp", False), False),
        "why_fetch": _as_bool(battery_raw.get("why_fetch", False), False),
        "completeness": _as_bool(battery_raw.get("completeness", False), False),
    }

    lm = root.get("layer_map", {})
    if not isinstance(lm, Mapping):
        raise ValueError("scaling.layer_map must be a mapping")
    layer_map_strategy = str(lm.get("strategy", "relative_depth"))
    layer_map_positions = tuple(float(x) for x in list(lm.get("positions", [0.25, 0.5, 0.75])))

    why_fetch_cfg = root.get("why_fetch", {})
    if not isinstance(why_fetch_cfg, Mapping):
        why_fetch_cfg = {}

    decomp_cfg = root.get("disamb_path_decomp", {})
    if not isinstance(decomp_cfg, Mapping):
        decomp_cfg = {}

    completeness_cfg = root.get("completeness", {})
    if not isinstance(completeness_cfg, Mapping):
        completeness_cfg = {}

    sae_cfg = root.get("sae", {})
    if not isinstance(sae_cfg, Mapping):
        sae_cfg = {}

    default_device = str(root.get("device", "auto"))
    default_dtype = root.get("torch_dtype", None)
    default_attn = str(root.get("attn_implementation", "eager"))
    default_local = _as_bool(root.get("local_files_only", False), False)
    default_trust = _as_bool(root.get("trust_remote_code", False), False)

    models_raw = list(root.get("models", []))
    if not models_raw:
        raise ValueError("scaling.models must be non-empty")

    models: List[ModelSpec] = []
    for raw in models_raw:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid model entry: {raw!r}")
        model_name = str(raw.get("model_name_or_path", raw.get("hf_id", ""))).strip()
        if not model_name:
            raise ValueError(f"Model entry missing model_name_or_path/hf_id: {raw!r}")
        label = str(raw.get("label", _safe_label(model_name))).strip()
        backend = str(raw.get("backend", "transformer_lens"))
        device = str(raw.get("device", default_device))
        dtype = raw.get("torch_dtype", default_dtype)
        torch_dtype = None if dtype is None or str(dtype).strip() in {"", "none", "null"} else str(dtype)
        attn_impl = str(raw.get("attn_implementation", default_attn))
        local_files_only = _as_bool(raw.get("local_files_only", default_local), default_local)
        trust_remote_code = _as_bool(raw.get("trust_remote_code", default_trust), default_trust)

        n_layers_val = raw.get("n_layers", None)
        n_layers = int(n_layers_val) if isinstance(n_layers_val, int) and int(n_layers_val) > 0 else None

        models.append(
            ModelSpec(
                label=str(label),
                model_name_or_path=str(model_name),
                backend=str(backend),
                device=str(device),
                torch_dtype=torch_dtype,
                attn_implementation=str(attn_impl),
                local_files_only=bool(local_files_only),
                trust_remote_code=bool(trust_remote_code),
                n_layers=n_layers,
            )
        )

    return StudyConfig(
        study_name=study_name,
        tasks=tasks,
        bootstrap_n=int(bootstrap_n),
        bootstrap_seed=int(bootstrap_seed),
        ci=float(ci),
        models=tuple(models),
        data_paths=data_paths,
        battery=battery,
        layer_map_strategy=str(layer_map_strategy),
        layer_map_positions=tuple(float(x) for x in layer_map_positions),
        why_fetch_n_examples=int(why_fetch_cfg.get("n_examples", 200)),
        why_fetch_heads_topk=int(why_fetch_cfg.get("heads_topk", 4)),
        disamb_max_pairs=int(decomp_cfg.get("max_pairs", 200)),
        disamb_mode=str(decomp_cfg.get("mode", "layer")),
        disamb_position=int(decomp_cfg.get("position", -1)),
        completeness_explanation_path=_resolve_rel_path(
            str(completeness_cfg.get("explanation_path", "")),
            base_dir=base,
        )
        if str(completeness_cfg.get("explanation_path", "")).strip()
        else "",
        sae_repo=str(sae_cfg.get("repo", "")),
        sae_layer=int(sae_cfg.get("layer", 15)),
        sae_width=str(sae_cfg.get("width", "16k")),
        sae_run_name=(None if sae_cfg.get("run_name", None) in {None, ""} else str(sae_cfg.get("run_name"))),
        sae_l0_target=(int(sae_cfg.get("l0_target")) if isinstance(sae_cfg.get("l0_target"), int) else None),
        sae_scale=float(sae_cfg.get("scale", 1.0)),
        sae_n_features=int(sae_cfg.get("n_features", 64)),
        sae_max_pairs=int(sae_cfg.get("max_pairs", 32)),
        sae_similarity_threshold=float(sae_cfg.get("similarity_threshold", 0.8)),
        sae_null_n=int(sae_cfg.get("null_n", 500)),
    )


def _sae_check_cmd(
    *,
    model_name_or_path: str,
    sae_repo: str,
    sae_layer: int,
    sae_width: str,
    sae_run_name: Optional[str],
    sae_l0_target: Optional[int],
    sae_scale: float,
    disamb_path: str,
    device: str,
    torch_dtype: Optional[str],
    attn_implementation: str,
    local_files_only: bool,
    trust_remote_code: bool,
    seed: int,
    out_json: str,
) -> List[str]:
    argv = [sys.executable, str(ROOT / "aom_sae_check.py")]
    argv += ["--model_name_or_path", str(model_name_or_path)]
    argv += ["--sae_repo", str(sae_repo)]
    argv += ["--layer", str(int(sae_layer))]
    argv += ["--width", str(sae_width)]
    if sae_run_name is not None and str(sae_run_name).strip():
        argv += ["--run_name", str(sae_run_name)]
    if sae_l0_target is not None:
        argv += ["--l0_target", str(int(sae_l0_target))]
    argv += ["--scale", str(float(sae_scale))]
    argv += ["--disamb_path", str(disamb_path)]
    argv += ["--device", str(device)]
    if torch_dtype is not None:
        argv += ["--torch_dtype", str(torch_dtype)]
    argv += ["--attn_implementation", str(attn_implementation)]
    if local_files_only:
        argv.append("--local_files_only")
    if trust_remote_code:
        argv.append("--trust_remote_code")
    argv += ["--seed", str(int(seed))]
    argv += ["--out_json", str(out_json)]
    return argv


def _prepare_model_layer_map(
    *,
    model: ModelSpec,
    strategy: str,
    positions: Sequence[float],
) -> Optional[LayerMapResult]:
    n_layers = model.n_layers
    if n_layers is None:
        n_layers = _read_model_n_layers(
            model_name_or_path=str(model.model_name_or_path),
            local_files_only=bool(model.local_files_only),
            trust_remote_code=bool(model.trust_remote_code),
        )
    if n_layers is None:
        return None
    return map_layers(n_layers=int(n_layers), strategy=str(strategy), positions=list(positions))


def _write_model_manifest(
    *,
    manifest_path: Path,
    study_name: str,
    model: ModelSpec,
    run_dir: Path,
    commands: Sequence[Sequence[str]],
    attempted: int,
    succeeded: int,
    failed: int,
    reasons: Sequence[str],
    layer_map: Optional[LayerMapResult],
) -> None:
    primary_csv = run_dir / "aom_eval.csv"
    csv_path = str(primary_csv) if primary_csv.exists() else ""
    csv_sha = _sha256_file(primary_csv) if primary_csv.exists() else ""

    result_row: Dict[str, Any] = {
        "study_name": str(study_name),
        "model_label": str(model.label),
        "model_name_or_path": str(model.model_name_or_path),
        "backend": str(model.backend),
        "run_dir": str(run_dir),
        "n_commands": int(attempted),
        "n_commands_succeeded": int(succeeded),
        "n_commands_failed": int(failed),
    }
    if layer_map is not None:
        result_row.update(layer_map.to_row())

    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path="",
        csv_path=csv_path,
        csv_sha256=csv_sha,
        csv_n_rows=None,
    )
    status = "PASS" if int(failed) == 0 else "FAIL"
    manifest["run_status"] = status
    manifest["run_status_reasons"] = [str(r) for r in reasons]
    manifest["run_summary"] = {
        "attempted": int(attempted),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "skipped": 0,
        "invalid": 0,
        "fail_rate": float(failed / max(1, attempted)),
        "skip_rate": 0.0,
        "invalid_rate": 0.0,
        "top_failure_types":
        [{"type": "command_failed", "count": int(failed), "example": str(reasons[0])}] if failed else [],
        "top_skip_types": [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    manifest["study_name"] = str(study_name)
    manifest["model"] = {
        "label": str(model.label),
        "model_name_or_path": str(model.model_name_or_path),
        "backend": str(model.backend),
        "device": str(model.device),
        "torch_dtype": "" if model.torch_dtype is None else str(model.torch_dtype),
        "attn_implementation": str(model.attn_implementation),
        "local_files_only": bool(model.local_files_only),
        "trust_remote_code": bool(model.trust_remote_code),
    }
    manifest["commands"] = [list(c) for c in commands]
    if layer_map is not None:
        manifest["layer_map"] = layer_map.to_row()
    manifest["git_commit"] = _try_git_commit()
    manifest["generated_at_utc"] = _utc_now_iso()
    write_run_manifest(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run canonical AoM scaling study across model scales.")
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "scaling_study.yaml"))
    p.add_argument("--results_root", type=str, default=str(ROOT / "runs"))
    p.add_argument("--study_name", type=str, default="", help="Optional override for scaling.study_name")
    p.add_argument("--smoke", action="store_true", help="Use small/fast settings for mechanistic stages.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    p.add_argument("--continue_on_error", action="store_true", help="Continue to next model when one command fails.")
    p.add_argument("--skip_aggregate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(str(args.config)).resolve()
    study_cfg = _parse_config(cfg_path)
    study_name = str(args.study_name).strip() or str(study_cfg.study_name)
    stamp = _timestamp_slug()

    results_root = Path(str(args.results_root)).resolve()
    study_root = results_root / str(study_name)
    study_run_dir = study_root / str(stamp)
    study_run_dir.mkdir(parents=True, exist_ok=True)

    model_runs: List[ModelRunResult] = []
    model_errors: List[str] = []

    for model in study_cfg.models:
        run_dir = study_root / str(model.label) / str(stamp)
        run_dir.mkdir(parents=True, exist_ok=True)

        layer_map = _prepare_model_layer_map(
            model=model,
            strategy=str(study_cfg.layer_map_strategy),
            positions=list(study_cfg.layer_map_positions),
        )
        patch_layers = ""
        if layer_map is not None:
            patch_layers = ",".join(str(int(x)) for x in layer_map.layers)

        commands: List[List[str]] = []
        reasons: List[str] = []
        attempted = 0
        succeeded = 0
        failed = 0

        disamb_path = str(study_cfg.data_paths["disamb"])
        cf_path = str(study_cfg.data_paths["cf"])
        coh_path = str(study_cfg.data_paths["coh"])
        tasks = set(study_cfg.tasks)

        try:
            if bool(study_cfg.battery.get("eval", False)):
                eval_cmd = _aom_eval_cmd(
                    models=[str(model.model_name_or_path)],
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    attn_implementation=str(model.attn_implementation),
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    disamb_path=(disamb_path if "disamb" in tasks else ""),
                    cf_path=(cf_path if "cf" in tasks else ""),
                    coh_path=(coh_path if "coh" in tasks else ""),
                    bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    csv_path=str(run_dir / "aom_eval.csv"),
                    run_patching=False,
                )
                commands.append(eval_cmd)

            if bool(study_cfg.battery.get("patching", False)) and "disamb" in tasks:
                cpt_cmd = _aom_eval_cmd(
                    models=[str(model.model_name_or_path)],
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    attn_implementation="eager",
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    disamb_path=str(disamb_path),
                    cf_path="",
                    coh_path="",
                    bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    csv_path=str(run_dir / "cpt_layer_sweep_disamb_only.csv"),
                    run_patching=True,
                    patch_layers=str(patch_layers),
                )
                commands.append(cpt_cmd)

            if bool(study_cfg.battery.get("patching", False)) and "cf" in tasks:
                cf_cmd = _cf_patching_cmd(
                    models=[str(model.model_name_or_path)],
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    cf_path=str(cf_path),
                    bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    csv_path=str(run_dir / "cf_patching.csv"),
                    patch_layers=str(patch_layers),
                )
                commands.append(cf_cmd)

            if bool(study_cfg.battery.get("patching", False)) and "coh" in tasks:
                coh_cmd = _coh_patching_cmd(
                    models=[str(model.model_name_or_path)],
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    coh_path=str(coh_path),
                    bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    csv_path=str(run_dir / "coh_patching.csv"),
                    patch_layers=str(patch_layers),
                )
                commands.append(coh_cmd)

            if bool(study_cfg.battery.get("sae_sterility", False)):
                if not str(study_cfg.sae_repo).strip():
                    raise ValueError("battery.sae_sterility=true requires scaling.sae.repo")
                sae_cmd = _sae_check_cmd(
                    model_name_or_path=str(model.model_name_or_path),
                    sae_repo=str(study_cfg.sae_repo),
                    sae_layer=int(study_cfg.sae_layer),
                    sae_width=str(study_cfg.sae_width),
                    sae_run_name=study_cfg.sae_run_name,
                    sae_l0_target=study_cfg.sae_l0_target,
                    sae_scale=float(study_cfg.sae_scale),
                    disamb_path=str(disamb_path),
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    attn_implementation="eager",
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    seed=int(study_cfg.bootstrap_seed),
                    out_json=str(run_dir / "sae_sterility.json"),
                )
                commands.append(sae_cmd)

            if bool(study_cfg.battery.get("feature_families", False)):
                if not str(study_cfg.sae_repo).strip():
                    raise ValueError("battery.feature_families=true requires scaling.sae.repo")
                ff_cmd = _feature_families_cmd(
                    model_name_or_path=str(model.model_name_or_path),
                    sae_repo=str(study_cfg.sae_repo),
                    sae_layer=int(study_cfg.sae_layer),
                    sae_width=str(study_cfg.sae_width),
                    sae_scale=float(study_cfg.sae_scale),
                    n_features=int(study_cfg.sae_n_features),
                    max_pairs=(min(int(study_cfg.sae_max_pairs), 8) if bool(args.smoke) else int(study_cfg.sae_max_pairs)),
                    similarity_threshold=float(study_cfg.sae_similarity_threshold),
                    null_n=(min(int(study_cfg.sae_null_n), 100) if bool(args.smoke) else int(study_cfg.sae_null_n)),
                    disamb_path=str(disamb_path),
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    out_prefix=str(run_dir / f"feature_families_{_safe_label(model.label)}"),
                    smoke=bool(args.smoke),
                )
                commands.append(ff_cmd)

            if bool(study_cfg.battery.get("disamb_path_decomp", False)) and "disamb" in tasks:
                if str(model.backend).lower() != "transformer_lens":
                    reasons.append("disamb_path_decomp skipped: backend!=transformer_lens")
                else:
                    decomp_cmd = _disamb_path_decomp_cmd(
                        model_name_or_path=str(model.model_name_or_path),
                        device=str(model.device),
                        torch_dtype=model.torch_dtype,
                        local_files_only=bool(model.local_files_only),
                        trust_remote_code=bool(model.trust_remote_code),
                        disamb_path=str(disamb_path),
                        bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                        bootstrap_seed=int(study_cfg.bootstrap_seed),
                        ci=float(study_cfg.ci),
                        max_pairs=(min(int(study_cfg.disamb_max_pairs), 8) if bool(args.smoke) else int(study_cfg.disamb_max_pairs)),
                        mode=str(study_cfg.disamb_mode),
                        position=int(study_cfg.disamb_position),
                        rows_csv_path=str(run_dir / f"disamb_path_decomp_rows_{_safe_label(model.label)}.csv"),
                        summary_csv_path=str(run_dir / f"disamb_path_decomp_summary_{_safe_label(model.label)}.csv"),
                    )
                    if bool(args.smoke):
                        decomp_cmd.append("--smoke")
                    commands.append(decomp_cmd)

            if bool(study_cfg.battery.get("why_fetch", False)):
                if str(model.backend).lower() != "transformer_lens":
                    reasons.append("why_fetch skipped: backend!=transformer_lens")
                else:
                    for task in study_cfg.tasks:
                        wf_cmd = _why_fetch_cmd(
                            task=str(task),
                            model_name_or_path=str(model.model_name_or_path),
                            device=str(model.device),
                            torch_dtype=model.torch_dtype,
                            local_files_only=bool(model.local_files_only),
                            trust_remote_code=bool(model.trust_remote_code),
                            disamb_path=str(disamb_path),
                            cf_path=str(cf_path),
                            coh_path=str(coh_path),
                            n_examples=(
                                min(int(study_cfg.why_fetch_n_examples), 10) if bool(args.smoke) else int(study_cfg.why_fetch_n_examples)
                            ),
                            heads_topk=(
                                min(int(study_cfg.why_fetch_heads_topk), 2) if bool(args.smoke) else int(study_cfg.why_fetch_heads_topk)
                            ),
                            bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                            bootstrap_seed=int(study_cfg.bootstrap_seed),
                            ci=float(study_cfg.ci),
                            rows_csv_path=str(run_dir / f"why_fetch_{task}_rows_{_safe_label(model.label)}.csv"),
                            summary_csv_path=str(run_dir / f"why_fetch_{task}_summary_{_safe_label(model.label)}.csv"),
                            smoke=bool(args.smoke),
                        )
                        commands.append(wf_cmd)

            if bool(study_cfg.battery.get("completeness", False)):
                comp_cmd = _completeness_cmd(
                    model_name_or_path=str(model.model_name_or_path),
                    tasks=",".join(study_cfg.tasks),
                    disamb_path=str(disamb_path),
                    cf_path=str(cf_path),
                    coh_path=str(coh_path),
                    layers=str(patch_layers),
                    explanation_path=str(study_cfg.completeness_explanation_path),
                    device=str(model.device),
                    torch_dtype=model.torch_dtype,
                    attn_implementation="eager",
                    local_files_only=bool(model.local_files_only),
                    trust_remote_code=bool(model.trust_remote_code),
                    bootstrap_n=(min(int(study_cfg.bootstrap_n), 100) if bool(args.smoke) else int(study_cfg.bootstrap_n)),
                    bootstrap_seed=int(study_cfg.bootstrap_seed),
                    ci=float(study_cfg.ci),
                    csv_path=str(run_dir / f"completeness_{_safe_label(model.label)}.csv"),
                    smoke=bool(args.smoke),
                )
                commands.append(comp_cmd)

            for cmd in commands:
                attempted += 1
                try:
                    _run(cmd, cwd=ROOT, dry_run=bool(args.dry_run))
                    succeeded += 1
                except Exception as e:
                    failed += 1
                    reasons.append(f"command_failed: {e}")
                    if not bool(args.continue_on_error):
                        break

            model_manifest_path = run_dir / "RUN_MANIFEST.json"
            _write_model_manifest(
                manifest_path=model_manifest_path,
                study_name=str(study_name),
                model=model,
                run_dir=run_dir,
                commands=commands,
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
                reasons=reasons,
                layer_map=layer_map,
            )

            model_runs.append(
                ModelRunResult(
                    model_label=str(model.label),
                    model_name_or_path=str(model.model_name_or_path),
                    run_dir=str(run_dir),
                    run_manifest_path=str(model_manifest_path),
                    layer_map_layers=tuple(layer_map.layers if layer_map is not None else ()),
                )
            )

            if failed and not bool(args.continue_on_error):
                raise RuntimeError(f"Model run failed for {model.label}: {reasons[-1]}")

        except Exception as e:
            model_errors.append(f"{model.label}: {e}")
            if not bool(args.continue_on_error):
                raise

    outputs: Dict[str, str] = {}
    if not bool(args.skip_aggregate):
        agg = aggregate_scaling_outputs(
            study_name=str(study_name),
            model_runs=[
                {
                    "model_label": mr.model_label,
                    "model_name_or_path": mr.model_name_or_path,
                    "run_dir": mr.run_dir,
                }
                for mr in model_runs
            ],
            out_dir=study_run_dir,
        )
        outputs = {k: str(v) for k, v in agg.items()}

    study_result_row: Dict[str, Any] = {
        "study_name": str(study_name),
        "timestamp": str(stamp),
        "n_models": int(len(study_cfg.models)),
        "n_models_completed": int(len(model_runs)),
        "n_model_errors": int(len(model_errors)),
        "study_run_dir": str(study_run_dir),
        "metrics_summary_csv": str(outputs.get("metrics", "")),
        "mechanistic_summary_csv": str(outputs.get("mechanistic", "")),
        "completeness_summary_csv": str(outputs.get("completeness", "")),
    }

    primary = Path(outputs.get("metrics", "")) if outputs.get("metrics", "") else None
    study_manifest = build_run_manifest(
        argv=sys.argv,
        results_row=study_result_row,
        dataset_manifest_path="",
        csv_path=(str(primary) if primary is not None else ""),
        csv_sha256=(_sha256_file(primary) if primary is not None and primary.exists() else ""),
        csv_n_rows=None,
    )
    study_manifest["run_status"] = "PASS" if not model_errors else "FAIL"
    study_manifest["run_status_reasons"] = [str(x) for x in model_errors]
    study_manifest["run_summary"] = {
        "attempted": int(len(study_cfg.models)),
        "succeeded": int(len(model_runs)),
        "failed": int(len(model_errors)),
        "skipped": 0,
        "invalid": 0,
        "fail_rate": float(len(model_errors) / max(1, len(study_cfg.models))),
        "skip_rate": 0.0,
        "invalid_rate": 0.0,
        "top_failure_types":
        [{"type": "model_failed", "count": int(len(model_errors)), "example": str(model_errors[0])}] if model_errors else [],
        "top_skip_types": [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    study_manifest["study"] = {
        "name": str(study_name),
        "timestamp": str(stamp),
        "config_path": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "generated_at_utc": _utc_now_iso(),
        "git_commit": _try_git_commit(),
    }
    study_manifest["model_runs"] = [
        {
            "model_label": str(mr.model_label),
            "model_name_or_path": str(mr.model_name_or_path),
            "run_dir": str(mr.run_dir),
            "run_manifest_path": str(mr.run_manifest_path),
            "layer_map_layers": ",".join(str(int(x)) for x in mr.layer_map_layers),
        }
        for mr in model_runs
    ]
    study_manifest["outputs"] = outputs

    study_manifest_path = study_run_dir / "RUN_MANIFEST.json"
    write_run_manifest(study_manifest_path, study_manifest)

    print(f"Wrote study manifest: {study_manifest_path}", flush=True)
    if outputs:
        for name, path in sorted(outputs.items()):
            print(f"Wrote {name} summary: {path}", flush=True)


if __name__ == "__main__":
    main()
