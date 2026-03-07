# Archival Release Model

## Scope

This repository now distinguishes four surfaces:

1. **Source artifacts**
   - Checked-in development artifacts under `results/`, `data_paper_hardened_v2/`, and manuscript-side evidence files.
   - These files remain the scientific record used for repository-side verification.
   - Publication tooling must not rewrite these tracked sources in place.

2. **Publication artifacts**
   - Sanitized JSON/CSV copies emitted into a separate tree such as `release_public/` or a temp publication directory.
   - These copies are the only JSON/CSV artifacts intended for an external archive or release bundle.
   - Publication copies mirror repo-relative paths so bundle builders can substitute them without changing archive layout.

3. **Release metadata**
   - `LICENSE`
   - `README.md`
   - `CITATION.cff`
   - `.zenodo.json`
   - `RELEASE_MANIFEST.json`
   - release-checklist and archival-format docs

4. **Verification outputs**
   - publication audits under `release_public/audit/`
   - README reproduction reports under `reports/`
   - archival-readiness summaries under `reports/`

## Non-Mutating Rule

- `scripts/release_json_artifacts.py publish` writes publication-safe copies into a separate output tree.
- Source JSON/CSV artifacts under version control are never rewritten as the publication mechanism.
- Audit reports are computed from source-to-publication diffs and emitted alongside the publication copies.

## Path Model

- Repo-local absolute paths are rewritten to repo-relative paths in publication copies.
- Hugging Face cache snapshot paths are rewritten to stable `hf://org/name@rev/...` identifiers.
- Publication manifests and audit reports record paths relative to the publication root, not machine-local absolute paths.

## Bundle Model

- `scripts/build_minimal_publish_bundle.py` accepts `--publication_root`.
- When a mirrored publication copy exists, the bundle consumes that sanitized copy while preserving the original repo-relative archive name.
- Static metadata (`LICENSE`, `README.md`, `CITATION.cff`, `.zenodo.json`) comes from the repository root.
- Bundle dry-runs are part of archival readiness and must succeed before release tagging.

## Verification Model

- Portability checks run on publication copies, not on raw tracked artifacts.
- JSON/CSV audits classify path-only, provenance-semantic, portability-only, and metric-semantic changes.
- Metric-semantic changes in publication generation are treated as failures and must be diagnosed before release.
- README reproduction runs write into a fresh external output directory and are compared against the tracked numeric reference artifacts with a scripted verifier.
