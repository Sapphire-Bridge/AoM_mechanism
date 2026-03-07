from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass(frozen=True)
class LayerMapResult:
    strategy: str
    n_layers: int
    positions: tuple[float, ...]
    layers: tuple[int, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "layer_map_strategy": str(self.strategy),
            "n_layers": int(self.n_layers),
            "layer_map_positions": ",".join(f"{float(x):.6g}" for x in self.positions),
            "layer_map_layers": ",".join(str(int(x)) for x in self.layers),
        }


def _coerce_positions(positions: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for p in positions:
        try:
            out.append(float(p))
        except Exception as e:  # pragma: no cover - defensive branch
            raise ValueError(f"invalid relative layer position: {p!r}") from e
    if not out:
        raise ValueError("layer map positions must be non-empty")
    return out


def map_layers_relative_depth(*, n_layers: int, positions: Sequence[float]) -> List[int]:
    if int(n_layers) <= 0:
        raise ValueError(f"n_layers must be positive, got {int(n_layers)}")
    if int(n_layers) == 1:
        return [0]

    out: List[int] = []
    for raw in positions:
        p = min(1.0, max(0.0, float(raw)))
        idx = int(round((int(n_layers) - 1) * p))
        out.append(int(idx))
    uniq = sorted(set(int(x) for x in out if 0 <= int(x) < int(n_layers)))
    if not uniq:
        raise ValueError("layer mapping resolved to empty layer set")
    return uniq


def map_layers(
    *,
    n_layers: int,
    strategy: str,
    positions: Sequence[float],
) -> LayerMapResult:
    strat = str(strategy).strip().lower()
    pos = _coerce_positions(positions)
    if strat != "relative_depth":
        raise ValueError(f"Unsupported layer map strategy: {strategy!r}")
    mapped = map_layers_relative_depth(n_layers=int(n_layers), positions=pos)
    return LayerMapResult(
        strategy=str(strat),
        n_layers=int(n_layers),
        positions=tuple(float(x) for x in pos),
        layers=tuple(int(x) for x in mapped),
    )
