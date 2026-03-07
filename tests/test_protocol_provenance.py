from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aom.provenance.protocol import resolve_protocol_provenance
from aom_cf_patching import _resolve_protocol_provenance as resolve_cf
from aom_coh_patching import _resolve_protocol_provenance as resolve_coh
from aom_eval import _resolve_protocol_provenance as resolve_eval


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_empty(fn):
    path, sha = fn(protocol_path_raw="", protocol_sha256_raw="")
    assert path == ""
    assert sha == ""


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_autohash(tmp_path: Path, fn):
    p = tmp_path / "proto.yaml"
    p.write_text("k: v\n", encoding="utf-8")
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    path, sha = fn(protocol_path_raw=str(p), protocol_sha256_raw="")
    assert path == str(p)
    assert sha == expected


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_matching_hash_ok(tmp_path: Path, fn):
    p = tmp_path / "proto.yaml"
    p.write_text("k: v\n", encoding="utf-8")
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    path, sha = fn(protocol_path_raw=str(p), protocol_sha256_raw=str(expected))
    assert path == str(p)
    assert sha == expected


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_mismatch_hash_raises(tmp_path: Path, fn):
    p = tmp_path / "proto.yaml"
    p.write_text("k: v\n", encoding="utf-8")
    bad = "0" * 64
    if bad == hashlib.sha256(p.read_bytes()).hexdigest():
        bad = "1" * 64
    with pytest.raises(ValueError, match="mismatch"):
        fn(protocol_path_raw=str(p), protocol_sha256_raw=str(bad))


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_invalid_hash_raises(fn):
    with pytest.raises(ValueError, match="protocol_sha256"):
        fn(protocol_path_raw="", protocol_sha256_raw="xyz")


@pytest.mark.parametrize("fn", [resolve_eval, resolve_cf, resolve_coh])
def test_resolve_protocol_provenance_missing_path_raises(tmp_path: Path, fn):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        fn(protocol_path_raw=str(missing), protocol_sha256_raw="")


def test_resolve_protocol_provenance_sha_without_path_unverified():
    prov = resolve_protocol_provenance(protocol_path_raw="", protocol_sha256_raw="a" * 64, require_path_for_sha=False)
    assert prov.protocol_path == ""
    assert prov.protocol_sha256 == "a" * 64
    assert prov.protocol_sha256_verified is False
    assert prov.protocol_sha256_source == "provided_unverified"


def test_resolve_protocol_provenance_empty_source_is_explicit():
    prov = resolve_protocol_provenance(protocol_path_raw="", protocol_sha256_raw="", require_path_for_sha=False)
    assert prov.protocol_path == ""
    assert prov.protocol_sha256 == ""
    assert prov.protocol_sha256_verified is False
    assert prov.protocol_sha256_source == "empty"


def test_resolve_protocol_provenance_sha_without_path_can_be_forbidden():
    with pytest.raises(ValueError, match="requires --protocol_path"):
        _ = resolve_protocol_provenance(protocol_path_raw="", protocol_sha256_raw="a" * 64, require_path_for_sha=True)


def test_resolve_protocol_provenance_path_source_is_computed(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text("protocol:\n  name: demo\n", encoding="utf-8")
    prov = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="")
    assert prov.protocol_path == str(p.resolve())
    assert prov.protocol_sha256_verified is True
    assert prov.protocol_sha256_source == "computed_from_path"


def test_resolve_protocol_provenance_non_mapping_config_fails_closed(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        _ = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="")


def test_resolve_protocol_provenance_frozen_requires_protocol_block(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text("bootstrap:\n  n: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'protocol' mapping"):
        _ = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="", require_frozen=True)


def test_resolve_protocol_provenance_frozen_requires_required_fields(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text("protocol:\n  name: demo\n  status: frozen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen protocol is missing required fields"):
        _ = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="", require_frozen=True)


def test_resolve_protocol_provenance_frozen_requires_frozen_status(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text(
        "\n".join(
            [
                "protocol:",
                "  name: demo",
                "  version: v1",
                "  prereg_tag: proto_v1",
                "  status: draft",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires protocol.status='frozen'"):
        _ = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="", require_frozen=True)


def test_resolve_protocol_provenance_frozen_happy_path(tmp_path: Path):
    p = tmp_path / "proto.yaml"
    p.write_text(
        "\n".join(
            [
                "protocol:",
                "  name: demo",
                "  version: v1",
                "  prereg_tag: proto_v1",
                "  status: frozen",
                "bootstrap:",
                "  n: 10",
                "",
            ]
        ),
        encoding="utf-8",
    )
    prov = resolve_protocol_provenance(protocol_path_raw=str(p), protocol_sha256_raw="", require_frozen=True)
    assert prov.protocol_name == "demo"
    assert prov.protocol_version == "v1"
    assert prov.protocol_prereg_tag == "proto_v1"
    assert prov.protocol_sha256_source == "computed_from_path"


def test_resolve_protocol_provenance_frozen_requires_path():
    with pytest.raises(ValueError, match="required when require_frozen=True"):
        _ = resolve_protocol_provenance(protocol_path_raw="", protocol_sha256_raw="", require_frozen=True)
