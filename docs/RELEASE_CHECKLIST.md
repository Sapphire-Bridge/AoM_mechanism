# Release Checklist

## Freeze

- Confirm the release surface is frozen before adding or editing `.zenodo.json`.
- Confirm the working branch/worktree is based on a committed checkpoint.
- Confirm publication generation does not mutate tracked scientific artifacts in place.

## Verify

- Run `python scripts/archive_readiness_check.py`.
- Run one uninterrupted README reproduction into a fresh external output directory:
  - `python scripts/run_readme_reproduction.py --report_path reports/readme_reproduction_report.md --json_log_path reports/readme_reproduction_log.json`
- Review `reports/readme_reproduction_report.md` for numeric pass/fail against the tracked reference artifacts.
- Review `reports/archive_readiness_check.md` and `reports/archival_readiness_final.md`.

## Package

- Generate publication copies into a clean output tree:
  - `python scripts/release_json_artifacts.py publish --out_dir /tmp/release_public`
- Review:
  - `/tmp/release_public/RELEASE_MANIFEST.json`
  - `/tmp/release_public/audit/json_audit.json`
  - `/tmp/release_public/audit/csv_audit.json`
  - `/tmp/release_public/audit/portability_scan.json`
  - `/tmp/release_public/audit/summary.md`
- Dry-run the minimal publish bundle:
  - `python scripts/build_minimal_publish_bundle.py --dry_run --publication_root /tmp/release_public`

## Release Metadata

- Verify `CITATION.cff` parses and matches `.zenodo.json` license metadata.
- Verify the release bundle includes:
  - `LICENSE`
  - `README.md`
  - `CITATION.cff`
  - `.zenodo.json`
  - `RELEASE_MANIFEST.json`

## Finalize

- Build the final publish bundle from publication copies.
- Tag the release only after the file list is frozen.
- Create the DOI-backed archive from the frozen release bundle.
- Record the final command log and readiness summary in `reports/archival_readiness_final.md`.
