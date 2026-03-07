from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from aom.interventions.sae_adapter import SAEInputTransform, SAEProtocol


@dataclass(frozen=True)
class GemmaScopeSAEMetadata:
    repo_id_or_path: str
    layer: int
    width: str
    run_name: str
    d_in: int
    d_sae: int
    params_path: str
    cfg_path: Optional[str]
    inferred_from_weights: bool = False


class GemmaScopeSAE(nn.Module):
    """
    Minimal SAE implementation compatible with `SAEProtocol`.

    Gemma Scope residual-stream SAEs are exported as a standard 2-layer AE:
      f = relu(x @ W_enc + b_enc)
      x_hat = f @ W_dec + b_dec
    """

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        d_in = int(cfg["d_in"])
        d_sae = int(cfg["d_sae"])

        if W_enc.shape != (d_in, d_sae):
            raise ValueError(f"W_enc must have shape (d_in, d_sae)={(d_in, d_sae)}; got {tuple(W_enc.shape)}")
        if W_dec.shape != (d_sae, d_in):
            raise ValueError(f"W_dec must have shape (d_sae, d_in)={(d_sae, d_in)}; got {tuple(W_dec.shape)}")
        if b_enc.shape != (d_sae,):
            raise ValueError(f"b_enc must have shape (d_sae,)={(d_sae,)}; got {tuple(b_enc.shape)}")
        if b_dec.shape != (d_in,):
            raise ValueError(f"b_dec must have shape (d_in,)={(d_in,)}; got {tuple(b_dec.shape)}")

        # Store as parameters to keep device/dtype attached and easy to move.
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.cfg = dict(cfg)

        self._d_in = d_in
        self._d_sae = d_sae

    @property
    def d_in(self) -> int:  # noqa: D401
        """Model hidden size."""
        return self._d_in

    @property
    def d_sae(self) -> int:  # noqa: D401
        """SAE dictionary width."""
        return self._d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"encode expects (B,S,H); got {tuple(x.shape)}")
        if x.size(-1) != self._d_in:
            raise ValueError(f"encode expected hidden dim {self._d_in}; got {int(x.size(-1))}")
        x2 = x.reshape(-1, self._d_in)
        f2 = torch.relu(x2 @ self.W_enc + self.b_enc)
        return f2.reshape(x.shape[0], x.shape[1], self._d_sae)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"decode expects (B,S,D); got {tuple(features.shape)}")
        if features.size(-1) != self._d_sae:
            raise ValueError(f"decode expected d_sae {self._d_sae}; got {int(features.size(-1))}")
        f2 = features.reshape(-1, self._d_sae)
        x2 = f2 @ self.W_dec + self.b_dec
        return x2.reshape(features.shape[0], features.shape[1], self._d_in)


def _as_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype))


def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "d_in" not in cfg or "d_sae" not in cfg:
        raise ValueError("cfg.json must contain keys 'd_in' and 'd_sae'")
    return cfg


def _load_params_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _infer_cfg_from_weights(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if "W_enc" not in arrays or "b_enc" not in arrays:
        raise ValueError(
            "Cannot infer cfg without at least 'W_enc' and 'b_enc' in params.npz "
            f"(found keys={sorted(arrays.keys())})"
        )

    W_enc_np = arrays["W_enc"]
    b_enc_np = arrays["b_enc"]

    if W_enc_np.ndim != 2:
        raise ValueError(f"Expected W_enc to be 2D; got shape {tuple(W_enc_np.shape)}")

    b_enc_flat = np.asarray(b_enc_np).reshape(-1)
    if b_enc_flat.ndim != 1 or b_enc_flat.size < 1:
        raise ValueError(f"Expected b_enc to be 1D with len>0; got shape {tuple(b_enc_np.shape)}")

    d_sae = int(b_enc_flat.shape[0])
    W_enc_shape = W_enc_np.shape  # (d_in, d_sae) or (d_sae, d_in)

    if W_enc_shape[0] == d_sae and W_enc_shape[1] != d_sae:
        d_in = int(W_enc_shape[1])
    elif W_enc_shape[1] == d_sae and W_enc_shape[0] != d_sae:
        d_in = int(W_enc_shape[0])
    elif W_enc_shape[0] == d_sae and W_enc_shape[1] == d_sae:
        d_in = int(d_sae)
    else:
        raise ValueError(
            "Cannot infer d_in from params.npz: expected W_enc to have one dimension equal to "
            f"d_sae={d_sae} (from b_enc), but got W_enc shape {tuple(W_enc_shape)}"
        )

    if "b_dec" in arrays:
        b_dec_flat = np.asarray(arrays["b_dec"]).reshape(-1)
        if int(b_dec_flat.shape[0]) != d_in:
            raise ValueError(
                f"Inferred d_in={d_in} from W_enc/b_enc, but b_dec has length {int(b_dec_flat.shape[0])}"
            )

    if "W_dec" in arrays:
        W_dec_np = arrays["W_dec"]
        if W_dec_np.ndim != 2:
            raise ValueError(f"Expected W_dec to be 2D; got shape {tuple(W_dec_np.shape)}")
        if W_dec_np.shape not in ((d_sae, d_in), (d_in, d_sae)):
            raise ValueError(
                f"Inferred (d_sae, d_in)=({d_sae}, {d_in}), but W_dec has shape {tuple(W_dec_np.shape)}"
            )

    return {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": str(W_enc_np.dtype),
        "inferred_from_weights": True,
    }


def _canonicalize_weights(
    *,
    cfg: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for k in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if k not in arrays:
            raise ValueError(f"params.npz missing key {k!r}; found keys={sorted(arrays.keys())}")

    d_in = int(cfg["d_in"])
    d_sae = int(cfg["d_sae"])
    W_enc_np = arrays["W_enc"]
    W_dec_np = arrays["W_dec"]
    b_enc_np = arrays["b_enc"]
    b_dec_np = arrays["b_dec"]

    W_enc = torch.from_numpy(W_enc_np).to(device=device, dtype=dtype)
    W_dec = torch.from_numpy(W_dec_np).to(device=device, dtype=dtype)
    b_enc = torch.from_numpy(b_enc_np).to(device=device, dtype=dtype).reshape(-1)
    b_dec = torch.from_numpy(b_dec_np).to(device=device, dtype=dtype).reshape(-1)

    # Orientation ambiguity: accept either orientation and canonicalize to (d_in, d_sae) / (d_sae, d_in).
    if W_enc.shape == (d_sae, d_in):
        W_enc = W_enc.t()
    if W_dec.shape == (d_in, d_sae):
        W_dec = W_dec.t()

    if W_enc.shape != (d_in, d_sae):
        raise ValueError(
            f"W_enc shape mismatch: got {tuple(W_enc.shape)}, expected {(d_in, d_sae)} (or transpose)"
        )
    if W_dec.shape != (d_sae, d_in):
        raise ValueError(
            f"W_dec shape mismatch: got {tuple(W_dec.shape)}, expected {(d_sae, d_in)} (or transpose)"
        )
    if b_enc.shape != (d_sae,):
        raise ValueError(f"b_enc shape mismatch: got {tuple(b_enc.shape)}, expected {(d_sae,)}")
    if b_dec.shape != (d_in,):
        raise ValueError(f"b_dec shape mismatch: got {tuple(b_dec.shape)}, expected {(d_in,)}")

    return W_enc, W_dec, b_enc, b_dec


def _resolve_local_run_dir(
    root: Path,
    *,
    layer: int,
    width: str,
    run_name: Optional[str],
    l0_target: Optional[int],
) -> Path:
    base = root / f"layer_{int(layer)}" / f"width_{str(width)}"
    if not base.exists():
        raise FileNotFoundError(f"Missing Gemma Scope directory: {str(base)}")
    if run_name is not None:
        run_dir = base / str(run_name)
        if not run_dir.exists():
            available = sorted([p.name for p in base.iterdir() if p.is_dir()])
            raise FileNotFoundError(f"Missing run directory: {str(run_dir)} (available runs: {available})")
        return run_dir

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if l0_target is not None:
        tag = f"average_l0_{int(l0_target)}"
        candidates = [p for p in candidates if tag in p.name]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) < 1:
        raise FileNotFoundError(f"No run directories found under {str(base)}")
    raise ValueError(
        f"Multiple Gemma Scope runs under {str(base)}; specify run_name (one of {[p.name for p in candidates]})"
    )


def list_gemma_scope_runs(
    repo_id_or_path: str,
    *,
    layer: int,
    width: str = "16k",
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> list[str]:
    """
    List available Gemma Scope run directory names for a given layer/width.

    Supports:
    - local directory path
    - HF repo id (requires network unless already cached and the hub client supports offline listing)
    """
    root = Path(repo_id_or_path)
    if root.exists():
        base = root / f"layer_{int(layer)}" / f"width_{str(width)}"
        if not base.exists():
            raise FileNotFoundError(f"Missing Gemma Scope directory: {str(base)}")
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    if bool(local_files_only):
        raise ValueError("Cannot list runs for HF repo with local_files_only=True; provide a local path or allow network.")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise ImportError("huggingface_hub is required to list Gemma Scope SAE runs") from e

    api = HfApi()
    prefix = f"layer_{int(layer)}/width_{str(width)}/"
    files = api.list_repo_files(repo_id_or_path, revision=revision)
    return sorted({Path(f).parent.name for f in files if f.startswith(prefix) and f.endswith("params.npz")})


def load_gemma_scope_sae(
    repo_id_or_path: str,
    *,
    layer: int,
    width: str = "16k",
    run_name: Optional[str] = None,
    l0_target: Optional[int] = None,
    device: str = "mps",
    dtype: str | torch.dtype = "auto",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple[GemmaScopeSAE, GemmaScopeSAEMetadata]:
    """
    Load a Gemma Scope residual-stream SAE as an `SAEProtocol`.

    Supports:
    - local directory path (offline)
    - HF repo id (downloads via huggingface_hub)
    """
    device_t = torch.device(str(device))
    if isinstance(dtype, str) and str(dtype).strip().lower() in {"", "none", "null", "auto"}:
        dtype_t = torch.float16 if str(device_t.type).lower() == "mps" else torch.float32
    else:
        dtype_t = _as_torch_dtype(dtype)

    root = Path(repo_id_or_path)
    if root.exists():
        run_dir = _resolve_local_run_dir(root, layer=layer, width=width, run_name=run_name, l0_target=l0_target)
        params_path = run_dir / "params.npz"
        cfg_path: Optional[Path] = run_dir / "cfg.json"
    else:
        # Remote HF repo. We require selection to be unambiguous.
        if run_name is None and l0_target is None:
            raise ValueError("For HF repo ids, specify run_name or l0_target to disambiguate the run directory.")
        try:
            from huggingface_hub import HfApi, hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
        except ImportError as e:  # pragma: no cover
            raise ImportError("huggingface_hub is required to download Gemma Scope SAEs") from e

        prefix = f"layer_{int(layer)}/width_{str(width)}/"
        if run_name is not None:
            run_dir_rel = str(Path(prefix) / str(run_name))
        else:
            run_names = list_gemma_scope_runs(
                repo_id_or_path,
                layer=int(layer),
                width=str(width),
                revision=revision,
                local_files_only=bool(local_files_only),
            )
            if l0_target is not None:
                tag = f"average_l0_{int(l0_target)}"
                run_names = [n for n in run_names if tag in n]
            if len(run_names) != 1:
                run_dirs = [str(Path(prefix) / n) for n in run_names]
                raise ValueError(f"Expected exactly 1 run dir under {prefix}; found {run_dirs}")
            run_dir_rel = str(Path(prefix) / run_names[0])

        try:
            params_path = Path(
                hf_hub_download(
                    repo_id_or_path,
                    filename=str(Path(run_dir_rel) / "params.npz"),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=bool(local_files_only),
                )
            )
        except EntryNotFoundError as e:
            # Give a more actionable error including available runs at this layer/width.
            try:
                runs = list_gemma_scope_runs(repo_id_or_path, layer=int(layer), width=str(width), revision=revision)
            except Exception:
                runs = []
            raise FileNotFoundError(
                f"Missing params.npz at {run_dir_rel!r} in repo {repo_id_or_path!r} (available runs: {runs})"
            ) from e
        run_dir = Path(run_dir_rel)
        cfg_path = None

    arrays = _load_params_npz(params_path)

    if root.exists():
        if cfg_path is not None and cfg_path.exists():
            cfg = _load_cfg(cfg_path)
        else:
            cfg = _infer_cfg_from_weights(arrays)
            cfg_path = None
    else:
        try:
            cfg_path = Path(
                hf_hub_download(
                    repo_id_or_path,
                    filename=str(Path(run_dir) / "cfg.json"),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=bool(local_files_only),
                )
            )
            cfg = _load_cfg(cfg_path)
        except EntryNotFoundError:
            cfg = _infer_cfg_from_weights(arrays)
            cfg_path = None

    W_enc, W_dec, b_enc, b_dec = _canonicalize_weights(cfg=cfg, arrays=arrays, dtype=dtype_t, device=device_t)
    sae = GemmaScopeSAE(W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, cfg=cfg)

    meta = GemmaScopeSAEMetadata(
        repo_id_or_path=str(repo_id_or_path),
        layer=int(layer),
        width=str(width),
        run_name=str(run_dir.name),
        d_in=int(cfg["d_in"]),
        d_sae=int(cfg["d_sae"]),
        params_path=str(params_path),
        cfg_path=str(cfg_path) if cfg_path is not None else None,
        inferred_from_weights=bool(cfg.get("inferred_from_weights", False)),
    )
    return sae, meta


@torch.no_grad()
def calibrate_sae_scale(
    *,
    model: nn.Module,
    sae: SAEProtocol,
    layer: int,
    scales: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    calibration_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[float, Dict[float, float]]:
    """
    Grid search over input scales to minimize reconstruction MSE at the exact hooked tensor.

    This deliberately captures the same tensor you patch (block forward output),
    rather than relying on `output_hidden_states` semantics.
    """
    from aom.interventions.activation_patching import get_decoder_blocks

    if calibration_input_ids is None:
        raise ValueError("calibration_input_ids is required (tokenization should happen outside this helper)")

    blocks = get_decoder_blocks(model)  # may raise if architecture unsupported
    if int(layer) < 0 or int(layer) >= len(blocks):
        raise ValueError(f"layer {int(layer)} out of range [0, {len(blocks)})")

    captured: list[torch.Tensor] = []

    def _capture(_module: nn.Module, _inputs: Tuple, output: Tuple | torch.Tensor):
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            raise ValueError("expected hidden tensor (B,S,H) from block output")
        captured.append(h.detach())

    handle = blocks[int(layer)].register_forward_hook(_capture)
    try:
        model(
            input_ids=calibration_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one capture; got {len(captured)}")
    hidden = captured[0]
    if device is not None:
        hidden = hidden.to(device)

    best_scale = float(scales[0])
    best_mse = float("inf")
    mse_by_scale: Dict[float, float] = {}

    for s in scales:
        tr = SAEInputTransform(scale=float(s))
        x_in = tr.forward(hidden)
        f = sae.encode(x_in)
        recon = sae.decode(f)
        recon_model = tr.inverse(recon)
        mse = float(torch.mean((recon_model - hidden) ** 2).item())
        mse_by_scale[float(s)] = mse
        if mse < best_mse - 1e-12:
            best_mse = mse
            best_scale = float(s)
    return best_scale, mse_by_scale
