#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _stable_u64(seed: int, s: str) -> int:
    h = hashlib.sha256()
    h.update(str(int(seed)).encode("utf-8"))
    h.update(b"|")
    h.update(str(s).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], byteorder="little", signed=False)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_id(row: Dict[str, Any], *, id_key: str) -> str:
    v = row.get(str(id_key), None)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Missing non-empty id field {id_key!r} in row: {row!r}")
    return str(v)


def _group_key(pair_id: str, *, mode: str) -> str:
    if mode == "prefix":
        return str(pair_id).split("_", 1)[0] if "_" in str(pair_id) else ""
    raise ValueError(f"Unknown group_mode: {mode!r}")


def _select_dev_ids(pair_ids: List[str], *, seed: int, dev_n: int) -> List[str]:
    if dev_n < 0:
        raise ValueError("dev_n must be >= 0")
    scored: List[Tuple[int, str]] = [(_stable_u64(seed, pid), pid) for pid in pair_ids]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [pid for _score, pid in scored[:dev_n]]


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministically split DisambPair JSONL into dev/test by pair_id.")
    p.add_argument("--in_jsonl", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0, help="Split seed (default: 0).")
    p.add_argument("--dev_n", type=int, default=10, help="Dev pairs per group (default: 10).")
    p.add_argument(
        "--id_key",
        type=str,
        default="pair_id",
        help="Row id key to split on (default: pair_id).",
    )
    p.add_argument(
        "--group_mode",
        type=str,
        default="prefix",
        choices=["prefix"],
        help="Grouping mode for stratified splits (default: prefix before underscore).",
    )
    p.add_argument(
        "--groups",
        type=str,
        default="csd,ih",
        help="Comma-separated group keys to include (default: csd,ih).",
    )
    p.add_argument(
        "--write_combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write combined dev/test across groups (default: on).",
    )
    args = p.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    rows = _read_jsonl(in_path)

    by_group: Dict[str, List[Dict[str, Any]]] = {g: [] for g in groups}
    seen_ids: set[str] = set()
    for r in rows:
        pid = _get_id(r, id_key=str(args.id_key))
        if pid in seen_ids:
            raise ValueError(f"Duplicate id {pid!r} in {in_path}")
        seen_ids.add(pid)
        g = _group_key(pid, mode=str(args.group_mode))
        if g in by_group:
            by_group[g].append(r)

    manifest: Dict[str, Any] = {
        "in_jsonl": str(in_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "dev_n_per_group": int(args.dev_n),
        "id_key": str(args.id_key),
        "group_mode": str(args.group_mode),
        "groups": list(groups),
        "counts": {},
        "dev_ids": {},
    }

    combined_dev: List[Dict[str, Any]] = []
    combined_test: List[Dict[str, Any]] = []

    stem = in_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]

    for g in groups:
        rr = by_group.get(g, [])
        manifest["counts"][g] = int(len(rr))
        if not rr:
            continue
        if int(args.dev_n) >= len(rr):
            raise ValueError(f"dev_n ({int(args.dev_n)}) must be < group size ({len(rr)}) for group {g!r}")

        pair_ids = [_get_id(x, id_key=str(args.id_key)) for x in rr]
        dev_ids = set(_select_dev_ids(pair_ids, seed=int(args.seed), dev_n=int(args.dev_n)))
        manifest["dev_ids"][g] = sorted(dev_ids)

        dev_rows = [x for x in rr if _get_id(x, id_key=str(args.id_key)) in dev_ids]
        test_rows = [x for x in rr if _get_id(x, id_key=str(args.id_key)) not in dev_ids]

        out_dev = out_dir / f"{stem}_{g}_dev.jsonl"
        out_test = out_dir / f"{stem}_{g}_test.jsonl"
        _write_jsonl(out_dev, dev_rows)
        _write_jsonl(out_test, test_rows)

        if bool(args.write_combined):
            combined_dev.extend(dev_rows)
            combined_test.extend(test_rows)

    if bool(args.write_combined):
        _write_jsonl(out_dir / f"{stem}_dev.jsonl", combined_dev)
        _write_jsonl(out_dir / f"{stem}_test.jsonl", combined_test)

    out_manifest = out_dir / f"{stem}_split_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote splits under {out_dir}")
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()

