from __future__ import annotations

from aom.run_summary import RunSummary


def test_run_summary_fails_if_no_successes():
    s = RunSummary()
    status, reasons = s.evaluate()
    assert status == "FAIL"
    assert reasons == ["succeeded == 0"]

    s.record_failure(ValueError("boom"))
    status, reasons = s.evaluate()
    assert status == "FAIL"
    assert reasons == ["succeeded == 0"]


def test_run_summary_warns_on_failures_or_skips():
    s = RunSummary()
    s.record_success()
    status, reasons = s.evaluate()
    assert status == "PASS"
    assert reasons == []

    s.record_failure(ValueError("boom"))
    status, reasons = s.evaluate()
    assert status == "WARN"
    assert reasons == ["nonzero failures/skips"]

    s2 = RunSummary()
    s2.record_success()
    s2.record_skip(ValueError("skip"))
    status, reasons = s2.evaluate()
    assert status == "WARN"
    assert reasons == ["nonzero failures/skips"]


def test_run_summary_warns_on_invalid_results():
    s = RunSummary()
    s.record_success()
    s.record_invalid("missing/invalid primary metrics: disamb.accuracy")
    status, reasons = s.evaluate()
    assert status == "WARN"
    assert "invalid_results > 0" in reasons


def test_run_summary_thresholds_trigger_fail():
    s = RunSummary()
    s.record_success()
    s.record_failure(ValueError("boom"))
    s.record_failure(RuntimeError("oops"))

    status, reasons = s.evaluate(max_fail_rate=0.50)
    assert status == "FAIL"
    assert any("fail_rate" in r for r in reasons)

    s2 = RunSummary()
    s2.record_success()
    s2.record_skip(ValueError("skip1"))
    s2.record_skip(ValueError("skip2"))
    status, reasons = s2.evaluate(max_skips=1)
    assert status == "FAIL"
    assert any("max_skips" in r for r in reasons)


def test_run_summary_as_dict_contains_top_exception_types():
    s = RunSummary()
    s.record_success()
    s.record_failure(ValueError("bad"))
    s.record_failure(ValueError("worse"))
    s.record_failure(RuntimeError("oops"))
    s.record_skip(ValueError("skip"))

    d = s.as_dict(top_k=2)
    assert d["attempted"] == 5
    assert d["succeeded"] == 1
    assert d["failed"] == 3
    assert d["skipped"] == 1

    top_fail = {row["type"]: row["count"] for row in d["top_failure_types"]}
    assert top_fail.get("ValueError") == 2
    assert top_fail.get("RuntimeError") == 1

    top_skip = {row["type"]: row["count"] for row in d["top_skip_types"]}
    assert top_skip.get("ValueError") == 1


def test_run_summary_invariant_violations_fail():
    s = RunSummary(attempted=1, succeeded=1, failed=1, skipped=0)
    status, reasons = s.evaluate()
    assert status == "FAIL"
    assert any("invariant:" in r for r in reasons)

    s2 = RunSummary(attempted=1, succeeded=1, failed=0, skipped=0, invalid=2)
    status, reasons = s2.evaluate()
    assert status == "FAIL"
    assert any("invalid > succeeded" in r for r in reasons)
