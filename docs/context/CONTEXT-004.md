# CONTEXT-004

## Scope

Files covered here:

- `paper/MoM_paper.md`
- `MoM_evidence_contract.md`
- `PAPER_VERIFICATION_GUIDE.md`
- `scripts/check_evidence_contract.py`
- `scripts/check_evidence_contract_fields.py`
- `scripts/verify_mom_paper.py`
- `scripts/release_json_artifacts.py`
- `scripts/build_minimal_publish_bundle.py`
- `public_artifacts/RELEASE_MANIFEST.json`

## Architecture Overview

This repo treats publication support as a governed workflow, not just a folder of output files.

- `paper/MoM_paper.md` is the manuscript claim surface.
- `MoM_evidence_contract.md` maps evidence IDs and contract fields onto those claims.
- `PAPER_VERIFICATION_GUIDE.md` is the reviewer-oriented explanation of how to inspect support artifacts.
- `scripts/check_evidence_contract.py` and `scripts/check_evidence_contract_fields.py` lint the contract and its references.
- `scripts/verify_mom_paper.py` checks generated or checked-in support artifacts against expected references.
- `scripts/release_json_artifacts.py` sanitizes and audits public JSON/CSV release outputs before publication.

## Key Flows

### 1. Contract integrity

- The evidence contract is the bridge between manuscript claims and computational artifacts.
- Contract-check scripts parse the paper and contract together to ensure evidence IDs, references, and required fields remain aligned.
- This is the first line of defense against citation drift between the manuscript and the supporting repo.

### 2. Support-artifact verification

- `scripts/verify_mom_paper.py` validates support artifacts against checked-in references.
- It can prefer manifest-aware inputs when available, and use CSV-based fallback behavior when necessary.
- This matters because the repo mixes older artifact conventions with newer manifest-bound outputs.

### 3. Public release sanitization

- `scripts/release_json_artifacts.py` constructs a sanitized publication tree from audited result files.
- It emits multiple audit products, including JSON audit, CSV audit, portability scan, summary markdown, and a publication manifest.
- The release path is therefore not just copying files. It is performing a policy check on what is safe and portable to expose.

### 4. Minimal publish bundle support

- `scripts/build_minimal_publish_bundle.py` and `public_artifacts/RELEASE_MANIFEST.json` belong to the outward-facing packaging story.
- These files matter when the task is not reproducing numbers locally but assembling a verifiable distribution.

## Research Invariants

- Evidence IDs in `paper/MoM_paper.md` and `MoM_evidence_contract.md` must stay synchronized.
- A paper-support artifact is not publication-ready merely because it exists in `results/`. It must survive verification and release checks.
- Manifest-aware verification is preferable to raw filename matching because filenames alone do not encode all provenance.
- Public release should remain sanitized and audited. Do not add raw internal artifacts to publication trees without updating the release policy.

## Interfaces

Main verification/release scripts:

- `scripts/check_evidence_contract.py`
- `scripts/check_evidence_contract_fields.py`
- `scripts/verify_mom_paper.py`
- `scripts/release_json_artifacts.py`
- `scripts/build_minimal_publish_bundle.py`

Primary human-facing docs:

- `paper/MoM_paper.md`
- `MoM_evidence_contract.md`
- `PAPER_VERIFICATION_GUIDE.md`

Typical outputs from this layer:

- contract field checks
- verification pass/fail reports
- audit JSON and CSV reports
- release summary markdown
- public release manifest

## Cached Artifacts And Provenance

- Verification often consumes checked-in result files, manifests, or both.
- Release tooling should preserve enough provenance for outside reviewers to understand what each exported artifact represents.
- The publication manifest is the public-facing provenance summary for sanitized outputs.

## Operational Notes / Gotchas

- Paper verification is downstream of experiment correctness. If verification fails, first determine whether the artifact is wrong or the contract/reference is stale.
- Contract scripts can fail for clerical drift, not only for scientific errors. Keep that distinction clear during triage.
- Release tooling is a late-stage safety boundary. Small schema or naming changes upstream can break it unexpectedly.
- Do not assume all checked-in artifacts are meant for public release. Some are internal support files or intermediate products.

## Claim Traceability

- Claim wording lives in `paper/MoM_paper.md`.
- Evidence bindings live in `MoM_evidence_contract.md`.
- Reviewer inspection workflow lives in `PAPER_VERIFICATION_GUIDE.md`.
- Contract correctness is checked by `scripts/check_evidence_contract.py` and `scripts/check_evidence_contract_fields.py`.
- Support-artifact correctness is checked by `scripts/verify_mom_paper.py`.
- Public-release correctness is governed by `scripts/release_json_artifacts.py` and the release manifest.

If a reviewer asks "where does this claim come from?", the intended route is:

- Find the paper claim and its evidence ID.
- Resolve that ID in `MoM_evidence_contract.md`.
- Use `PAPER_VERIFICATION_GUIDE.md` plus verification scripts to locate the underlying support artifact.
- Trace the generating runner and metric logic through `CONTEXT-003` and `CONTEXT-002`.
