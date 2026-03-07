# Public Artifact Format

## JSON Policy

- Publication JSON must be valid portable JSON.
- Bare `NaN`, `Infinity`, and `-Infinity` tokens are not allowed.
- Non-finite numeric values are normalized to `null` in publication copies.
- JSON is emitted deterministically with sorted keys and stable indentation.

## CSV Policy

- Publication CSV keeps the original schema and row order.
- Repo-local absolute paths are rewritten to repo-relative paths.
- Hugging Face snapshot paths are rewritten to `hf://org/name@rev/...`.
- Non-finite numeric-looking text is normalized to the string sentinel `null`.
- Textual policy fields that literally use strings such as `nan` for non-numeric metadata are allowlisted and preserved.

## Path Policy

- Absolute paths under the repository root become repo-relative.
- Repository-root values become `.`.
- Hugging Face cache snapshot paths become stable `hf://` identifiers.
- Machine-local prefixes such as `/Users/`, `/home/`, and Windows drive paths are forbidden in publication outputs.

## Validation

- `scripts/release_json_artifacts.py publish` emits:
  - `RELEASE_MANIFEST.json`
  - `audit/json_audit.json`
  - `audit/csv_audit.json`
  - `audit/portability_scan.json`
  - `audit/summary.md`
- `scripts/release_json_artifacts.py scan-portability` fails on:
  - bare `NaN`
  - bare `Infinity`
  - bare `-Infinity`
  - `/Users/`
  - `/home/`
  - Windows absolute paths

## Release Rule

- External archives and release bundles must consume only publication copies plus release metadata.
- Tracked source artifacts remain in the repository for evidence verification, but they are not the archive payload.
