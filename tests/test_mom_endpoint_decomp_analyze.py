from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

from aom.metrics.primary_logodds import PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT
from scripts.clt_raw_comparability import _write_csv
from scripts.mom_endpoint_decomp_analyze import (
    PRIMARY_FIELDS,
    REQUIRED_PRIMARY_METADATA_FIELDS,
    _build_pair_aggregates,
    _load_rows,
    parse_args,
)


def _base_row() -> dict[str, object]:
    row: dict[str, object] = {
        "analysis_included": True,
        "pair_id": "p1",
        "layer": 4,
        "effect_A": 0.3,
        "effect_C": 0.5,
        "decomp_exp_delta_logz_mean_A": 0.2,
        "decomp_other_delta_logz_mean_A": 0.2,
        "decomp_exp_delta_logz_mean_C": 0.1,
        "decomp_other_delta_logz_mean_C": 0.1,
        "primary_labels_binary": True,
        "primary_expected_all_single": True,
        "primary_other_all_single": True,
        "primary_token_sets_disjoint": True,
        "primary_candidate_count_equal": True,
        "primary_static_applicable": True,
        "primary_logodds_applicable": True,
        "primary_cancellation_pass": True,
    }
    for field in PRIMARY_FIELDS:
        row[field] = 0.0

    row["dlogPE_A"] = 0.4
    row["dlogPO_A"] = 0.1
    row["dlogPE_C"] = 0.9
    row["dlogPO_C"] = 0.4
    row["d_CA_logPE"] = 0.5
    row["d_CA_logPO"] = 0.3
    row["primary_delta_m_from_logodds_A"] = 0.3
    row["primary_delta_m_from_logodds_C"] = 0.5
    row["primary_delta_m_residual_A"] = 0.0
    row["primary_delta_m_residual_C"] = 0.0
    return row


def test_schema_roundtrip_writer_and_analyzer_loader(tmp_path: Path) -> None:
    out_csv = tmp_path / "comparability_endpoint_v2.csv"
    _write_csv([_base_row()], out_csv)
    rows = _load_rows(out_csv)
    assert len(rows) == 1
    row = rows[0]
    for field in REQUIRED_PRIMARY_METADATA_FIELDS:
        assert field in row
    for field in PRIMARY_FIELDS:
        assert field in row


def test_build_pair_aggregates_preserves_logodds_identity(tmp_path: Path) -> None:
    out_csv = tmp_path / "comparability_endpoint_v2.csv"
    _write_csv([_base_row()], out_csv)
    rows = _load_rows(out_csv)
    aggs = _build_pair_aggregates(rows, ratio_den_eps=1e-8)
    assert len(aggs) == 1
    agg = aggs[0]
    assert agg.primary_static_pair is True
    assert agg.d_ca_diag_logz == pytest.approx(0.0, abs=1e-12)
    assert agg.d_ca_logodds_residual == pytest.approx(0.0, abs=1e-12)


def test_non_primary_pair_sets_primary_metrics_nan(tmp_path: Path) -> None:
    row_a = _base_row()
    row_b = dict(_base_row())
    row_b["primary_logodds_applicable"] = False
    row_b["primary_static_applicable"] = False
    row_b["primary_expected_all_single"] = False
    out_csv = tmp_path / "comparability_endpoint_v2.csv"
    _write_csv([row_a, row_b], out_csv)
    rows = _load_rows(out_csv)
    aggs = _build_pair_aggregates(rows, ratio_den_eps=1e-8)
    agg = aggs[0]
    assert agg.primary_static_pair is False
    assert agg.n_primary_rows == 1
    assert agg.d_ca_diag_logz != agg.d_ca_diag_logz  # nan
    assert agg.d_ca_logodds_residual != agg.d_ca_logodds_residual  # nan
    assert agg.metrics["d_CA_logPE"] != agg.metrics["d_CA_logPE"]  # nan


def test_loader_requires_new_schema_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "old_schema.csv"
    with bad_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["analysis_included", "pair_id", "layer", "effect_A", "effect_C"])
        writer.writeheader()
        writer.writerow(
            {
                "analysis_included": True,
                "pair_id": "p1",
                "layer": 4,
                "effect_A": 0.1,
                "effect_C": 0.2,
            }
        )
    with pytest.raises(ValueError, match="Missing required columns"):
        _load_rows(bad_csv)


def test_parse_args_uses_pair_aggregate_output_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mom_endpoint_decomp_analyze.py"])
    args = parse_args()
    assert args.out_pair_csv.name == "endpoint_pair_aggregates_v2.csv"
    assert args.out_pair_csv.name != "endpoint_candidate_terms_v2.csv"
    assert args.primary_residual_tol == pytest.approx(float(PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT), abs=1e-12)
    assert args.comparability_summary is None
    assert args.disamb_path is None
    assert args.tokenizer_name_or_path == ""
