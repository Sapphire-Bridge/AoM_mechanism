from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        import aom  # type: ignore

        aom_file = str(Path(aom.__file__).resolve()) if getattr(aom, "__file__", None) else ""
    except Exception as e:  # pragma: no cover - diagnostic script
        aom_file = f"<import failed: {type(e).__name__}: {e}>"

    out = {
        "repo_root": str(repo_root),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "aom_file": aom_file,
        "sys_path_first_10": sys.path[:10],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
