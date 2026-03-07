from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_CONTRACT_ID_CELL_RE = re.compile(r"^`([ERC]\d+[a-z]?)`$")
_EVIDENCE_ID_RE = re.compile(r"^[ERC]\d+[a-z]?$")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_BRACE_RE = re.compile(r"\{([^{}]+)\}")
_TABLE_SEPARATOR_RE = re.compile(r"^\|\s*-{3,}\s*(\|\s*-{3,}\s*)+\|?$")


@dataclass(frozen=True)
class ContractRow:
    evidence_id: str
    metric_cell: str
    artifacts_cell: str


def _iter_contract_rows(contract_text: str) -> list[ContractRow]:
    rows: list[ContractRow] = []
    for line in contract_text.splitlines():
        if not line.startswith("|"):
            continue
        if _TABLE_SEPARATOR_RE.fullmatch(line.strip()) is not None:
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells or cells[0] == "Evidence ID":
            continue

        m = _CONTRACT_ID_CELL_RE.fullmatch(cells[0])
        if m is None:
            continue

        if len(cells) < 4:
            continue

        rows.append(
            ContractRow(
                evidence_id=m.group(1),
                metric_cell=cells[2],
                artifacts_cell=cells[3],
            )
        )
    return rows


def _extract_backticked(text: str) -> list[str]:
    return _BACKTICK_RE.findall(text)


def _split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in str(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_metric_token(token: str) -> list[str]:
    """
    Normalize metric-contract shorthand into one or more field/value patterns.

    Supported shorthands:
    - "foo.csv: bar" -> "bar"
    - "x[50/16384]" -> "x.{50,16384}"
    - "x[*]" -> "x.*"
    - top-level comma lists outside braces are split into separate tokens.
    """
    t = str(token).strip()
    if not t:
        return []

    if ":" in t:
        left, right = t.split(":", 1)
        if ".csv" in left.lower():
            t = right.strip()

    def _bracket_repl(m: re.Match[str]) -> str:
        inner = str(m.group(1)).strip()
        if inner == "*":
            return ".*"
        if "/" in inner:
            opts = [x.strip() for x in inner.split("/") if x.strip()]
            if opts:
                return ".{" + ",".join(opts) + "}"
        return "[" + inner + "]"

    t = re.sub(r"\[([^\]]+)\]", _bracket_repl, t)
    return _split_top_level_csv(t)


def _brace_expand(token: str) -> list[str]:
    m = _BRACE_RE.search(token)
    if m is None:
        return [token]
    options = [opt.strip() for opt in m.group(1).split(",") if opt.strip()]
    expanded: list[str] = []
    for opt in options:
        expanded.extend(_brace_expand(token[: m.start()] + opt + token[m.end() :]))
    return expanded


def _required_field_patterns(
    evidence_id: str, rows_by_id: dict[str, ContractRow], visiting: set[str] | None = None
) -> list[str]:
    if visiting is None:
        visiting = set()
    if evidence_id in visiting:
        raise ValueError(
            "Cycle in 'Same columns as' references: " + " -> ".join([*sorted(visiting), evidence_id])
        )
    visiting.add(evidence_id)

    row = rows_by_id[evidence_id]
    tokens = _extract_backticked(row.metric_cell)

    non_id_tokens = [t for t in tokens if _EVIDENCE_ID_RE.fullmatch(t) is None]
    if non_id_tokens:
        patterns: list[str] = []
        for t in non_id_tokens:
            for normalized in _normalize_metric_token(t):
                patterns.extend(_brace_expand(normalized))
        return patterns

    # Handle shorthand like: "Same columns as `E5a`"
    if "Same columns as" in row.metric_cell:
        ref_ids = [t for t in tokens if _EVIDENCE_ID_RE.fullmatch(t) is not None]
        if ref_ids:
            return _required_field_patterns(ref_ids[0], rows_by_id, visiting)

    return []


def _artifact_tokens(row: ContractRow) -> list[str]:
    return [str(p) for p in _extract_backticked(row.artifacts_cell)]


def _read_csv_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        return {str(h) for h in reader.fieldnames if h is not None}


def _read_csv_values(path: Path, *, max_unique_values: int = 20000) -> set[str]:
    vals: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return vals
        for row in reader:
            for v in row.values():
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                vals.add(s)
                if len(vals) >= int(max_unique_values):
                    return vals
    return vals


def _collect_json_keys(obj: Any, out: set[str], *, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            skey = str(key)
            path = f"{prefix}.{skey}" if prefix else skey
            out.add(skey)
            out.add(path)
            _collect_json_keys(value, out, prefix=path)
    elif isinstance(obj, list):
        if prefix:
            out.add(prefix + "[]")
        for item in obj:
            _collect_json_keys(item, out, prefix=prefix)
    else:
        if prefix:
            out.add(prefix)


def _read_json_keys(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    keys: set[str] = set()
    _collect_json_keys(obj, keys)
    return keys


def _pattern_satisfied(pattern: str, available: set[str], available_values: set[str]) -> bool:
    if "*" in pattern:
        if any(fnmatch.fnmatchcase(name, pattern) for name in available):
            return True
    if pattern in available:
        return True
    # Some contract rows intentionally name categorical value checks (e.g. topk/randomk arms).
    return pattern in available_values


def check_contract_field_lists(contract_path: Path, repo_root: Path) -> dict[str, list[str]]:
    contract_text = contract_path.read_text(encoding="utf-8")
    rows = _iter_contract_rows(contract_text)
    rows_by_id = {r.evidence_id: r for r in rows}

    problems: dict[str, list[str]] = {}
    for row in rows:
        if not row.evidence_id.startswith("E"):
            continue

        required = _required_field_patterns(row.evidence_id, rows_by_id)
        if not required:
            continue

        artifact_tokens = _artifact_tokens(row)
        available: set[str] = set()
        available_values: set[str] = set()
        row_problems: list[str] = []

        for tok in artifact_tokens:
            expanded = _brace_expand(str(tok))
            matched_any = False
            for token in expanded:
                token = str(token).strip()
                if not token:
                    continue
                if any(ch in token for ch in ("*", "?", "[")):
                    matches = sorted(repo_root.glob(token))
                else:
                    p = repo_root / token
                    matches = [p] if p.exists() else []
                if not matches:
                    continue
                matched_any = True
                for path in matches:
                    if not path.is_file():
                        continue
                    if path.suffix.lower() == ".csv":
                        available |= _read_csv_header(path)
                        available_values |= _read_csv_values(path)
                    elif path.suffix.lower() == ".json":
                        available |= _read_json_keys(path)
            if not matched_any:
                row_problems.append(f"missing artifact: {tok}")

        missing_fields = [p for p in required if not _pattern_satisfied(p, available, available_values)]
        if missing_fields:
            row_problems.extend(f"missing field: {p}" for p in sorted(set(missing_fields)))

        if row_problems:
            problems[row.evidence_id] = row_problems

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Evidence Contract `E#` field lists against artifact columns/JSON keys."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("MoM_evidence_contract.md"),
        help="Path to evidence contract markdown.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repo root used to resolve artifact paths (default: cwd).",
    )
    args = parser.parse_args(argv)

    problems = check_contract_field_lists(args.contract, args.repo_root)
    if problems:
        for evidence_id, issues in sorted(problems.items()):
            print(f"[error] {evidence_id}:", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
        return 2

    print("[ok] All `E#` contract field lists match at least one listed artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
