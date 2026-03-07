from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers

from aom.mechanistic.backends.hf_eager import HFEagerBackend
from aom.mechanistic.backends.transformer_lens import TransformerLensBackend
from aom.mechanistic.induction import bootstrap_ci_matrix, expected_uniform
from aom.utils import get_best_device, set_seed


def _parse_layers(arg: str, n_layers: int) -> List[int]:
    raw = str(arg or "").strip().lower()
    if raw == "all":
        return list(range(int(n_layers)))
    if not raw:
        raise ValueError("layers must be 'all' or a comma-separated list")
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _git_commit() -> str:
    try:
        import subprocess

        root = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _concat_samples(samples: List[torch.Tensor]) -> torch.Tensor:
    if not samples:
        return torch.empty((0, 0), dtype=torch.float32)
    normed = [s.unsqueeze(0) if s.ndim == 1 else s for s in samples]
    return torch.cat(normed, dim=0)


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--backend", type=str, default="hf", choices=["hf", "transformer_lens"])
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--base_len", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", type=str, default="shuffle", choices=["shuffle", "offset0"])
    p.add_argument("--debug_store_attn", action="store_true")
    p.add_argument("--validate_attn", type=str, default="none", choices=["none", "first", "always"])
    p.add_argument("--tl_crosscheck", action="store_true", help="(TransformerLens backend) Add TL similarity columns.")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--bootstrap_n", type=int, default=200)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--ci", type=float, default=0.95)

    p.add_argument("--csv_path", type=str, default=str(Path(__file__).resolve().parent / "results" / "induction_heads.csv"))
    return p.parse_args()


def _make_backend(name: str):
    if name == "hf":
        return HFEagerBackend()
    if name == "transformer_lens":
        return TransformerLensBackend()
    raise ValueError(f"unknown backend: {name!r}")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[args.device])

    backend = _make_backend(str(args.backend))
    loaded = backend.load(
        model_name_or_path=str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )
    architecture = str(loaded.architecture)

    if int(args.repeats) != 2:
        raise ValueError("repeats must be 2 for V1 induction metric")
    if int(args.base_len) < 2:
        raise ValueError("base_len must be >= 2")

    max_pos = loaded.max_seq_len
    seq_len = int(args.base_len) * int(args.repeats)
    if max_pos is not None and seq_len > max_pos:
        raise ValueError(f"sequence length {seq_len} exceeds model max_position_embeddings={max_pos}")

    layers = _parse_layers(args.layers, int(loaded.n_layers))
    if not layers:
        raise ValueError("no layers specified")
    for l in layers:
        if l < 0 or l >= int(loaded.n_layers):
            raise ValueError(f"layer {l} out of range [0, {int(loaded.n_layers)})")

    commit = _git_commit()
    pyver = str(sys.version.split()[0])
    torchver = str(torch.__version__)
    txver = str(transformers.__version__)
    exp_uniform = float(expected_uniform(int(args.base_len), int(args.repeats)))
    model_param_dtype = str(loaded.model_param_dtype)
    model_param_device = str(loaded.model_param_device)
    rows: List[Dict[str, Any]] = []

    run_result = backend.run_batches(
        loaded=loaded,
        layers=layers,
        base_len=int(args.base_len),
        repeats=int(args.repeats),
        batch_size=int(args.batch_size),
        n_batches=int(args.n_batches),
        seed=int(args.seed),
        baseline=str(args.baseline),
        debug_store_attn=bool(args.debug_store_attn),
        validate_attn=str(args.validate_attn),
        tl_crosscheck=bool(args.tl_crosscheck),
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    collector = run_result.collector

    # Aggregate per-layer, per-head stats.
    for layer in layers:
        repeat_samples = _concat_samples(collector.get_samples(layer, "repeat"))
        control_samples = _concat_samples(collector.get_samples(layer, "control"))
        mass_repeat_samples = _concat_samples(collector.get_mass_samples(layer, "repeat"))
        mass_control_samples = _concat_samples(collector.get_mass_samples(layer, "control"))
        frac_repeat_samples = _concat_samples(collector.get_fraction_samples(layer, "repeat"))
        frac_control_samples = _concat_samples(collector.get_fraction_samples(layer, "control"))
        if repeat_samples.numel() == 0 or control_samples.numel() == 0:
            raise ValueError(f"missing samples for layer={layer} (baseline={args.baseline})")
        if repeat_samples.shape != control_samples.shape:
            raise ValueError(f"repeat/control sample shape mismatch at layer={layer}")
        if mass_repeat_samples.numel() == 0 or mass_control_samples.numel() == 0:
            raise ValueError(f"missing first-half mass samples for layer={layer}")
        if frac_repeat_samples.numel() == 0 or frac_control_samples.numel() == 0:
            raise ValueError(f"missing diag fraction samples for layer={layer}")
        if mass_repeat_samples.shape != mass_control_samples.shape:
            raise ValueError(f"first-half mass sample shape mismatch at layer={layer}")
        if frac_repeat_samples.shape != frac_control_samples.shape:
            raise ValueError(f"diag fraction sample shape mismatch at layer={layer}")

        advantage_samples = repeat_samples - control_samples
        mass_adv_samples = mass_repeat_samples - mass_control_samples
        frac_adv_samples = frac_repeat_samples - frac_control_samples
        rep_mean, rep_lo, rep_hi = bootstrap_ci_matrix(
            repeat_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        ctl_mean, ctl_lo, ctl_hi = bootstrap_ci_matrix(
            control_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        adv_mean, adv_lo, adv_hi = bootstrap_ci_matrix(
            advantage_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        mass_rep_mean, mass_rep_lo, mass_rep_hi = bootstrap_ci_matrix(
            mass_repeat_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        mass_ctl_mean, mass_ctl_lo, mass_ctl_hi = bootstrap_ci_matrix(
            mass_control_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        mass_adv_mean, mass_adv_lo, mass_adv_hi = bootstrap_ci_matrix(
            mass_adv_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        frac_rep_mean, frac_rep_lo, frac_rep_hi = bootstrap_ci_matrix(
            frac_repeat_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        frac_ctl_mean, frac_ctl_lo, frac_ctl_hi = bootstrap_ci_matrix(
            frac_control_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )
        frac_adv_mean, frac_adv_lo, frac_adv_hi = bootstrap_ci_matrix(
            frac_adv_samples.numpy(), n_bootstrap=int(args.bootstrap_n), ci=float(args.ci), seed=int(args.bootstrap_seed)
        )

        n_heads_observed = int(repeat_samples.shape[1])
        num_heads_cfg = loaded.num_attention_heads
        num_kv_heads_cfg = loaded.num_key_value_heads

        for h in range(n_heads_observed):
            row: Dict[str, Any] = {
                "model": args.model_name_or_path,
                "arch": architecture,
                "backend": str(args.backend),
                "backend_version": str(loaded.backend_version),
                "layer": int(layer),
                "head": int(h),
                "induction_score_repeat": float(rep_mean[h]),
                "induction_score_repeat_ci_low": float(rep_lo[h]),
                "induction_score_repeat_ci_high": float(rep_hi[h]),
                "induction_score_control": float(ctl_mean[h]),
                "induction_score_control_ci_low": float(ctl_lo[h]),
                "induction_score_control_ci_high": float(ctl_hi[h]),
                "induction_advantage": float(adv_mean[h]),
                "induction_advantage_ci_low": float(adv_lo[h]),
                "induction_advantage_ci_high": float(adv_hi[h]),
                "first_half_mass_repeat": float(mass_rep_mean[h]),
                "first_half_mass_repeat_ci_low": float(mass_rep_lo[h]),
                "first_half_mass_repeat_ci_high": float(mass_rep_hi[h]),
                "first_half_mass_control": float(mass_ctl_mean[h]),
                "first_half_mass_control_ci_low": float(mass_ctl_lo[h]),
                "first_half_mass_control_ci_high": float(mass_ctl_hi[h]),
                "first_half_mass_advantage": float(mass_adv_mean[h]),
                "first_half_mass_advantage_ci_low": float(mass_adv_lo[h]),
                "first_half_mass_advantage_ci_high": float(mass_adv_hi[h]),
                "diag_fraction_repeat": float(frac_rep_mean[h]),
                "diag_fraction_repeat_ci_low": float(frac_rep_lo[h]),
                "diag_fraction_repeat_ci_high": float(frac_rep_hi[h]),
                "diag_fraction_control": float(frac_ctl_mean[h]),
                "diag_fraction_control_ci_low": float(frac_ctl_lo[h]),
                "diag_fraction_control_ci_high": float(frac_ctl_hi[h]),
                "diag_fraction_advantage": float(frac_adv_mean[h]),
                "diag_fraction_advantage_ci_low": float(frac_adv_lo[h]),
                "diag_fraction_advantage_ci_high": float(frac_adv_hi[h]),
                "expected_uniform": exp_uniform,
                "base_len": int(args.base_len),
                "repeats": int(args.repeats),
                "batch_size": int(args.batch_size),
                "n_batches": int(args.n_batches),
                "seed": int(args.seed),
                "baseline_mode": str(args.baseline),
                "device": str(device),
                "torch_dtype": str(args.torch_dtype),
                "model_param_dtype": model_param_dtype,
                "model_param_device": model_param_device,
                "attn_implementation": str(args.attn_implementation),
                "git_commit": commit,
                "python_version": pyver,
                "torch_version": torchver,
                "transformers_version": txver,
                "n_heads_observed": n_heads_observed,
                "num_attention_heads": int(num_heads_cfg) if isinstance(num_heads_cfg, int) else "",
                "num_key_value_heads": int(num_kv_heads_cfg) if isinstance(num_kv_heads_cfg, int) else "",
            }
            extra = run_result.extras.get(int(layer), {}).get(int(h))
            if extra:
                row.update(extra)
            rows.append(row)

    _write_csv(rows, args.csv_path)


if __name__ == "__main__":
    main()
