from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract a DISAMB adversarial (cue-swapped) subset from a DISAMB JSONL.")
    p.add_argument("--in_path", type=str, default=str(ROOT / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--out_path", type=str, default=str(ROOT / "data" / "disamb_pairs_adversarial.jsonl"))
    p.add_argument(
        "--variant_pattern",
        type=str,
        default="distractor",
        help="Regex matched against metadata.pair_variant (default: 'distractor').",
    )
    return p.parse_args()


def _get_pair_variant(row: Dict[str, Any]) -> str:
    meta = row.get("metadata", None)
    if not isinstance(meta, dict):
        return ""
    v = meta.get("pair_variant", "")
    return str(v) if isinstance(v, str) else ""


def main() -> None:
    args = parse_args()
    pat = re.compile(str(args.variant_pattern))
    rows: List[Dict[str, Any]] = read_jsonl(str(args.in_path))
    out = [r for r in rows if pat.search(_get_pair_variant(r))]
    write_jsonl(out, str(args.out_path))
    print(f"Wrote {len(out)}/{len(rows)} rows to {args.out_path}", flush=True)


if __name__ == "__main__":
    main()

