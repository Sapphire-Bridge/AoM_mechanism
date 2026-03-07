from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol


@dataclass(frozen=True)
class HookSpec:
    """
    Backend-agnostic hook descriptor.

    `name` should use the normalized key naming convention (for example:
    `pattern.0`, `value.12`, `mlp_out.5`, `resid_post.7`).
    """

    name: str
    fn: Callable[..., Any]


@dataclass(frozen=True)
class CachedRun:
    logits: Any
    cache: Dict[str, Any]
    meta: Dict[str, Any]


class HookableBackend(Protocol):
    def run_with_cache(self, prompt: str, *, capture: List[str], **kwargs: Any) -> CachedRun: ...

    def run_with_hooks(self, prompt: str, *, hooks: List[HookSpec], **kwargs: Any) -> Any: ...
