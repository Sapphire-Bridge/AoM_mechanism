from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol

import torch


class InductionCollector(Protocol):
    def get_samples(self, layer: int, mode: str) -> List[torch.Tensor]: ...

    def get_mass_samples(self, layer: int, mode: str) -> List[torch.Tensor]: ...

    def get_fraction_samples(self, layer: int, mode: str) -> List[torch.Tensor]: ...


@dataclass(frozen=True)
class BackendLoadResult:
    model: Any
    tokenizer: Any
    architecture: str
    device: torch.device
    n_layers: int
    vocab_size: int
    exclude_token_ids: frozenset[int]
    max_seq_len: Optional[int]
    model_param_dtype: str
    model_param_device: str
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    backend_version: str


@dataclass(frozen=True)
class BackendRunResult:
    collector: InductionCollector
    extras: dict[int, dict[int, dict[str, float]]]


class InductionBackend(Protocol):
    name: str

    def load(
        self,
        *,
        model_name_or_path: str,
        device: torch.device,
        torch_dtype: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: str,
    ) -> BackendLoadResult: ...

    def run_batches(
        self,
        *,
        loaded: BackendLoadResult,
        layers: List[int],
        base_len: int,
        repeats: int,
        batch_size: int,
        n_batches: int,
        seed: int,
        baseline: str,
        debug_store_attn: bool,
        validate_attn: str,
        tl_crosscheck: bool,
        bootstrap_n: int,
        bootstrap_seed: int,
        ci: float,
    ) -> BackendRunResult: ...
