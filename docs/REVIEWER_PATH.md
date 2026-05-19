# Reviewer Path

This file gives the shortest paths for evaluating the AoM / MoM repository at different depths.

## 10 Minutes

Use this path if you want to confirm that the repository is structurally serious.

```bash
make check
```

What this covers:
- local test suite
- evidence-contract checks
- canonical smoke runner

Then inspect:
- `paper/MoM_paper.md`
- `MoM_evidence_contract.md`

Goal:
- verify that the repo boots
- verify that the manuscript/evidence linkage is enforced
- verify that the public review surface is real

## 1 Hour

Use this path if you want a stronger reviewer-facing validation.

```bash
make reviewer-assets
make reviewer-check
make one-result-check
```

If an accelerator is available:

```bash
make reviewer-assets
make reviewer-check
make one-result-check-gpu
```

What this covers:
- local reviewer asset validation
- smoke route
- paper-runner dry-run
- one substantive claim-level result check against tracked public reference artifacts

Goal:
- verify that the repo is not only structurally clean
- verify that a real result path is reproducible from the reviewer surface

## Full Review

Use this path if you want the canonical paper-facing reproduction route.

```bash
make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"
```

What this covers:
- strict paper-facing MoM package
- core comparability / endpoint-decomposition path
- support analyses referenced in the manuscript

Goal:
- evaluate the repository as a paper-reproduction system rather than only a codebase

## Claim-to-Artifact Navigation

Canonical manuscript:
- `paper/MoM_paper.md`

Canonical evidence table:
- `MoM_evidence_contract.md`

Key validation scripts:
- `scripts/check_evidence_contract.py`
- `scripts/check_evidence_contract_fields.py`

Key reproduction / reviewer entry points:
- `scripts/run_paper.py`
- `scripts/run_mom_paper.py`
- `scripts/reviewer_quickcheck.py`
- `scripts/run_one_result_check.py`

Key artifact surfaces:
- `results/`
- `public_artifacts/`

Use the Evidence IDs in `MoM_evidence_contract.md` and the Appendix A.2 claim-to-artifact map in `paper/MoM_paper.md` to trace manuscript statements back to checked-in artifacts.
