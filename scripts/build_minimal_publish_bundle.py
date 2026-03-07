#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANUSCRIPT = ROOT / "paper" / "MoM_paper.md"
DEFAULT_EVIDENCE_CONTRACT = ROOT / "MoM_evidence_contract.md"
DEFAULT_ARCHIVE = ROOT / "mom_minimal_publish_bundle.tar.gz"


ID_RE = re.compile(r"\b([ECR]\d+[a-z]?)\b")
ROW_ID_RE = re.compile(r"^\|\s*`([ECR]\d+[a-z]?)`\s*\|")
BACKTICK_RE = re.compile(r"`([^`]+)`")


STATIC_INCLUDE_PATHS: tuple[str, ...] = (
    "LICENSE",
    "CITATION.cff",
    "README.md",
    "paper/MoM_paper.md",
    "MoM_evidence_contract.md",
    "PAPER_MODES.md",
    "PAPER_VERIFICATION_GUIDE.md",
    "Makefile",
    "requirements.txt",
    "requirements.lock.txt",
    "data/README.md",
    "data_paper_hardened_v2",
    "scripts/generate_data.py",
    "scripts/run_paper.py",
    "scripts/run_scaling_study.py",
    "scripts/run_submission_full_strong.sh",
    "scripts/make_tables.py",
    "scripts/report_results.py",
    "scripts/build_minimal_publish_bundle.py",
    "configs/scaling_study.yaml",
    "tables/table_aom_eval.py",
    "tables/table_cf_patching.py",
    "tables/table_coh_patching.py",
)

OPTIONAL_INCLUDE_PATHS: tuple[str, ...] = (
    ".zenodo.json",
    "docs/ARCHIVAL_RELEASE_MODEL.md",
    "docs/PUBLIC_ARTIFACT_FORMAT.md",
    "docs/RELEASE_CHECKLIST.md",
    "reports/archive_readiness_check.md",
    "reports/readme_reproduction_report.md",
    "reports/readme_reproduction_log.json",
    "reports/archival_readiness_final.md",
)


CORE_ENTRYPOINTS: tuple[str, ...] = (
    "aom_eval.py",
    "aom_cf_patching.py",
    "aom_coh_patching.py",
    "aom_disamb_path_decomp.py",
    "aom_why_fetch.py",
    "aom_feature_families.py",
    "aom_completeness.py",
    "scripts/run_paper.py",
    "scripts/run_scaling_study.py",
)


@dataclass(frozen=True)
class BuildConfig:
    manuscript: Path
    evidence_contract: Path
    archive: Path
    dry_run: bool
    allow_missing_evidence: bool
    publication_root: Path | None
    extra_paths: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveItem:
    source: Path
    arcname: str


def _parse_args() -> BuildConfig:
    p = argparse.ArgumentParser(description="Build a minimal publish bundle from manuscript-cited evidence + code deps.")
    p.add_argument("--manuscript", type=str, default=str(DEFAULT_MANUSCRIPT))
    p.add_argument("--evidence-contract", type=str, default=str(DEFAULT_EVIDENCE_CONTRACT))
    p.add_argument("--archive", type=str, default=str(DEFAULT_ARCHIVE))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--publication_root",
        type=str,
        default="",
        help="Optional directory containing publication-safe copies mirrored by repo-relative path.",
    )
    p.add_argument(
        "--allow_missing_evidence",
        action="store_true",
        help="Do not fail if an evidence artifact referenced by the manuscript is missing.",
    )
    p.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional repo-relative path or glob to include (can be passed multiple times).",
    )
    args = p.parse_args()
    return BuildConfig(
        manuscript=Path(args.manuscript).expanduser().resolve(),
        evidence_contract=Path(args.evidence_contract).expanduser().resolve(),
        archive=Path(args.archive).expanduser().resolve(),
        dry_run=bool(args.dry_run),
        allow_missing_evidence=bool(args.allow_missing_evidence),
        publication_root=Path(args.publication_root).expanduser().resolve() if args.publication_root else None,
        extra_paths=tuple(str(x) for x in (args.extra_path or [])),
    )


def _rel_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError as e:
        raise ValueError(f"Path must be inside repo root ({ROOT}): {path}") from e


def _collect_manuscript_ids(manuscript: Path) -> set[str]:
    text = manuscript.read_text(encoding="utf-8")
    return {m.group(1) for m in ID_RE.finditer(text)}


def _collect_evidence_rows(contract_path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for raw in contract_path.read_text(encoding="utf-8").splitlines():
        m = ROW_ID_RE.match(raw.strip())
        if not m:
            continue
        evidence_id = m.group(1)
        parts = raw.split("|")
        # Table format: | ID | Claim | Metric | Artifact(s) | ...
        if len(parts) < 6:
            continue
        artifact_cell = parts[4]
        artifacts = [s.strip() for s in BACKTICK_RE.findall(artifact_cell) if s.strip()]
        rows[evidence_id] = artifacts
    return rows


def _expand_artifact_tokens(tokens: Iterable[str]) -> tuple[set[str], list[str]]:
    out: set[str] = set()
    missing: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token.startswith("http://") or token.startswith("https://"):
            continue
        if any(ch in token for ch in ("*", "?", "[")):
            matches = sorted(ROOT.glob(token))
            if not matches:
                missing.append(token)
                continue
            for m in matches:
                if m.is_file():
                    out.add(_rel_to_root(m))
            continue
        p = ROOT / token
        if p.is_file():
            out.add(_rel_to_root(p))
        elif p.is_dir():
            out.add(_rel_to_root(p))
        else:
            missing.append(token)
    return out, missing


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if rel.parts[0] == "aom":
        parts = list(rel.parts)
        parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            return ".".join(parts[:-1])
        return ".".join(parts)
    return rel.as_posix()[:-3].replace("/", ".")


def _build_module_map() -> dict[str, Path]:
    module_to_path: dict[str, Path] = {}
    for p in ROOT.rglob("*.py"):
        if any(part in {".git", ".venv", "__pycache__"} for part in p.parts):
            continue
        rel = p.relative_to(ROOT)
        if rel.parts[0] == "aom":
            parts = list(rel.parts)
            parts[-1] = parts[-1][:-3]
            if parts[-1] == "__init__":
                mod = ".".join(parts[:-1])
            else:
                mod = ".".join(parts)
        else:
            mod = rel.as_posix()[:-3].replace("/", ".")
        module_to_path[mod] = p
    return module_to_path


def _resolve_relative(base_package: str, level: int, module: str) -> str:
    if level <= 0:
        return module
    pkg = base_package.split(".") if base_package else []
    up = max(0, len(pkg) - (level - 1))
    prefix = ".".join(pkg[:up])
    if module:
        return f"{prefix}.{module}" if prefix else module
    return prefix


def _import_candidates(node: ast.AST, cur_package: str) -> list[str]:
    candidates: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            candidates.append(alias.name)
        return candidates
    if isinstance(node, ast.ImportFrom):
        base = _resolve_relative(cur_package, int(node.level), str(node.module or ""))
        if base:
            candidates.append(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            if base:
                candidates.append(f"{base}.{alias.name}")
        return candidates
    return candidates


def _collect_local_code_paths(entrypoints: Iterable[str]) -> set[str]:
    module_map = _build_module_map()
    queue: list[Path] = []
    seen: set[Path] = set()

    for rel in entrypoints:
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing entrypoint: {p}")
        queue.append(p)

    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        cur_mod = _module_name_for_path(path)
        if path.name == "__init__.py":
            cur_package = cur_mod
        else:
            cur_package = ".".join(cur_mod.split(".")[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for candidate in _import_candidates(node, cur_package):
                mod_path = module_map.get(candidate)
                if mod_path is not None and mod_path not in seen:
                    queue.append(mod_path)

    return {_rel_to_root(p) for p in seen}


def _iter_files(relpath: str) -> Iterable[Path]:
    p = ROOT / relpath
    if p.is_file():
        yield p
        return
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if not child.is_file():
                continue
            if "__pycache__" in child.parts:
                continue
            if child.name.endswith(".pyc") or child.name == ".DS_Store":
                continue
            yield child
        return
    raise FileNotFoundError(f"Missing path: {p}")


def _resolve_archive_items(include_relpaths: Iterable[str], *, publication_root: Path | None) -> list[ArchiveItem]:
    items: list[ArchiveItem] = []
    seen: set[str] = set()
    for rel in sorted(set(include_relpaths)):
        for f in _iter_files(rel):
            arcname = _rel_to_root(f)
            if arcname in seen:
                continue
            seen.add(arcname)
            source = publication_root / arcname if publication_root and (publication_root / arcname).is_file() else f
            items.append(ArchiveItem(source=source, arcname=arcname))

    if publication_root is not None:
        for extra_arcname in (
            "RELEASE_MANIFEST.json",
            "audit/json_audit.json",
            "audit/csv_audit.json",
            "audit/portability_scan.json",
            "audit/summary.md",
        ):
            extra_source = publication_root / extra_arcname
            if not extra_source.is_file() or extra_arcname in seen:
                continue
            seen.add(extra_arcname)
            items.append(ArchiveItem(source=extra_source, arcname=extra_arcname))
    return items


def _write_bundle(
    archive: Path,
    include_relpaths: Iterable[str],
    *,
    dry_run: bool,
    publication_root: Path | None,
) -> list[str]:
    items = _resolve_archive_items(include_relpaths, publication_root=publication_root)
    all_files = [item.arcname for item in items]

    if dry_run:
        for rel in all_files:
            print(rel)
        return all_files

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        for item in items:
            tf.add(item.source, arcname=item.arcname, recursive=False)
    return all_files


def _write_checksum(path: Path) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    out = path.with_suffix(path.suffix + ".sha256")
    out.write_text(f"{h.hexdigest()}  {path.name}\n", encoding="utf-8")


def main() -> None:
    cfg = _parse_args()

    if not cfg.manuscript.is_file():
        raise FileNotFoundError(f"Missing manuscript: {cfg.manuscript}")
    if not cfg.evidence_contract.is_file():
        raise FileNotFoundError(f"Missing evidence contract: {cfg.evidence_contract}")

    manuscript_ids = _collect_manuscript_ids(cfg.manuscript)
    rows = _collect_evidence_rows(cfg.evidence_contract)

    evidence_tokens: list[str] = []
    missing_ids: list[str] = []
    for evidence_id in sorted(manuscript_ids):
        row_tokens = rows.get(evidence_id)
        if row_tokens is None:
            missing_ids.append(evidence_id)
            continue
        evidence_tokens.extend(row_tokens)

    evidence_paths, missing_tokens = _expand_artifact_tokens(evidence_tokens)
    if missing_ids:
        print(f"[warn] IDs used in manuscript but not found in evidence table: {missing_ids}")

    if missing_tokens and not cfg.allow_missing_evidence:
        raise FileNotFoundError(
            "Missing evidence artifacts referenced by manuscript IDs:\n"
            + "\n".join(f"- {t}" for t in sorted(set(missing_tokens)))
        )
    if missing_tokens:
        print("[warn] Missing evidence artifacts (allowed by flag):")
        for tok in sorted(set(missing_tokens)):
            print(f"  - {tok}")

    extra_paths, missing_extra = _expand_artifact_tokens(cfg.extra_paths)
    if missing_extra:
        raise FileNotFoundError(
            "Missing --extra-path artifacts:\n" + "\n".join(f"- {t}" for t in sorted(set(missing_extra)))
        )

    code_paths = _collect_local_code_paths(CORE_ENTRYPOINTS)
    include_paths = set(STATIC_INCLUDE_PATHS) | evidence_paths | extra_paths | code_paths
    include_paths.update(path for path in OPTIONAL_INCLUDE_PATHS if (ROOT / path).exists())
    include_paths.add(_rel_to_root(cfg.manuscript))
    include_paths.add(_rel_to_root(cfg.evidence_contract))

    paths_file = cfg.archive.with_suffix(cfg.archive.suffix + ".paths.txt")
    files = _write_bundle(cfg.archive, include_paths, dry_run=cfg.dry_run, publication_root=cfg.publication_root)

    print(f"[info] manuscript evidence IDs: {len(manuscript_ids)}")
    print(f"[info] resolved files: {len(files)}")

    if cfg.dry_run:
        return

    paths_file.write_text("".join(f"{p}\n" for p in files), encoding="utf-8")
    _write_checksum(cfg.archive)
    meta = {
        "archive": str(cfg.archive),
        "manuscript": _rel_to_root(cfg.manuscript),
        "evidence_contract": _rel_to_root(cfg.evidence_contract),
        "n_files": len(files),
        "allow_missing_evidence": cfg.allow_missing_evidence,
        "publication_root": (
            cfg.publication_root.relative_to(ROOT).as_posix()
            if cfg.publication_root is not None and cfg.publication_root.is_relative_to(ROOT)
            else (cfg.publication_root.name if cfg.publication_root is not None else "")
        ),
    }
    meta_path = cfg.archive.with_suffix(cfg.archive.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote archive: {cfg.archive}")
    print(f"[ok] wrote digest:  {cfg.archive}.sha256")
    print(f"[ok] wrote file list: {paths_file}")
    print(f"[ok] wrote metadata:  {meta_path}")


if __name__ == "__main__":
    main()
