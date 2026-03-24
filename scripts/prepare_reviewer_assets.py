#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reviewer_assets import (
    local_asset_results,
    materialize_reviewer_clt_bundle,
    prepare_fixed_layer_sae_cache,
    prepare_model_snapshot,
)


def _best_accelerator() -> str:
    try:
        import torch
    except Exception:
        return ""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return ""


def _next_commands_snippet() -> str:
    commands = ["make reviewer-check", "make one-result-check"]
    if _best_accelerator():
        commands.append("make one-result-check-gpu")
    commands.append('make reproduction MOM_PAPER_ARGS="--run_root /tmp/mom_paper_review_run"')
    return "\n".join(commands)


def _print_step(header: str, detail: str) -> None:
    print(f"{header} {detail}")


def main() -> int:
    try:
        _print_step("[1/4] Model cache:", "downloading pinned Gemma 2 2B snapshot")
        model_snapshot = prepare_model_snapshot()
        _print_step("[ok]", f"model snapshot ready at {model_snapshot}")

        _print_step("[2/4] CLT bundle:", "materializing reviewer CLT bundle")
        bundle_path = materialize_reviewer_clt_bundle()
        _print_step("[ok]", f"CLT bundle ready at {bundle_path}")

        _print_step("[3/4] SAE support:", "caching fixed-layer SAE support files")
        sae_path = prepare_fixed_layer_sae_cache()
        _print_step("[ok]", f"fixed-layer SAE support ready at {sae_path}")
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    _print_step("[4/4] Verifying:", "checking local reviewer assets")
    results = local_asset_results()
    ready = all(result.ok for result in results)

    print("# MoM Reviewer Assets")
    print("")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"- {result.name}: `{status}`; {result.detail}")

    print("")
    print(f"- ready_for_offline_paper_run: `{'PASS' if ready else 'FAIL'}`")
    print("")
    print("## Next Commands")
    print("")
    print("```bash")
    print(_next_commands_snippet())
    print("```")

    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
