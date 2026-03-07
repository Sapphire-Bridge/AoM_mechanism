from __future__ import annotations

import json
from pathlib import Path

from scripts import archive_readiness_check as arc


def test_check_citation_accepts_required_fields(tmp_path: Path) -> None:
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'message: "cite this"',
                'title: "demo"',
                "authors:",
                '  - family-names: "Doe"',
                '    given-names: "Jane"',
                'license: "Apache-2.0"',
                "preferred-citation:",
                "  type: manuscript",
                '  title: "demo paper"',
                "  year: 2026",
                "  authors:",
                '    - family-names: "Doe"',
                '      given-names: "Jane"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = arc._check_citation(citation)

    assert result.status == "ok"


def test_check_archive_metadata_rejects_license_mismatch(tmp_path: Path) -> None:
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "\n".join(
            [
                "cff-version: 1.2.0",
                'message: "cite this"',
                'title: "demo"',
                "authors:",
                '  - family-names: "Doe"',
                '    given-names: "Jane"',
                'license: "Apache-2.0"',
                "preferred-citation:",
                "  type: manuscript",
                '  title: "demo paper"',
                "  year: 2026",
                "  authors:",
                '    - family-names: "Doe"',
                '      given-names: "Jane"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    zenodo = tmp_path / ".zenodo.json"
    zenodo.write_text(
        json.dumps(
            {
                "title": "demo",
                "upload_type": "software",
                "description": "desc",
                "creators": [{"name": "Doe, Jane"}],
                "license": "MIT",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = arc._check_archive_metadata(zenodo, citation)

    assert result.status == "fail"
    assert "License mismatch" in result.detail
