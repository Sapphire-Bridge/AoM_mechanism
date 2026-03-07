from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal


ErrorPolicy = Literal["raise", "warn_skip", "skip_silent"]
RunStatus = Literal["PASS", "WARN", "FAIL"]

_MAX_EXAMPLE_LEN = 500


def normalize_error_thresholds(
    *,
    max_fail_rate: float,
    max_skips: int,
    strict_errors: bool,
) -> tuple[float, int | None]:
    fail_rate = float(max_fail_rate)
    skips: int | None = None if int(max_skips) < 0 else int(max_skips)
    if bool(strict_errors):
        fail_rate = 0.0
        skips = 0
    return fail_rate, skips


def _cap(s: str, *, max_len: int) -> str:
    s = str(s)
    if len(s) <= int(max_len):
        return s
    if int(max_len) <= 1:
        return s[: int(max_len)]
    return s[: int(max_len) - 1] + "…"


@dataclass
class ExceptionCounts:
    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, str] = field(default_factory=dict)

    def add(self, exc: BaseException) -> None:
        name = type(exc).__name__
        self.counts[name] += 1
        if name not in self.examples:
            self.examples[name] = _cap(str(exc), max_len=_MAX_EXAMPLE_LEN)

    def top(self, k: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, count in self.counts.most_common(int(k)):
            out.append({"type": str(name), "count": int(count), "example": str(self.examples.get(name, ""))})
        return out


@dataclass
class ReasonCounts:
    counts: Counter[str] = field(default_factory=Counter)

    def add(self, reason: str) -> None:
        key = _cap(str(reason), max_len=_MAX_EXAMPLE_LEN)
        self.counts[key] += 1

    def top(self, k: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for reason, count in self.counts.most_common(int(k)):
            out.append({"reason": str(reason), "count": int(count)})
        return out


@dataclass
class RunSummary:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    invalid: int = 0
    failures: ExceptionCounts = field(default_factory=ExceptionCounts)
    skips: ExceptionCounts = field(default_factory=ExceptionCounts)
    invalid_reasons: ReasonCounts = field(default_factory=ReasonCounts)

    def _check_invariants(self) -> list[str]:
        problems: list[str] = []
        if int(self.attempted) != int(self.succeeded) + int(self.failed) + int(self.skipped):
            problems.append("attempted != succeeded+failed+skipped")
        if int(self.invalid) > int(self.succeeded):
            problems.append("invalid > succeeded")
        return problems

    def record_success(self) -> None:
        self.attempted += 1
        self.succeeded += 1

    def record_failure(self, exc: BaseException) -> None:
        self.attempted += 1
        self.failed += 1
        self.failures.add(exc)

    def record_skip(self, exc: BaseException) -> None:
        self.attempted += 1
        self.skipped += 1
        self.skips.add(exc)

    def record_invalid(self, reason: str) -> None:
        self.invalid += 1
        self.invalid_reasons.add(str(reason))

    def as_dict(self, *, top_k: int = 5) -> dict[str, Any]:
        invariant_problems = self._check_invariants()
        attempted = int(self.attempted)
        failed = int(self.failed)
        skipped = int(self.skipped)
        invalid = int(self.invalid)
        fail_rate = float(failed / attempted) if attempted > 0 else float("nan")
        skip_rate = float(skipped / attempted) if attempted > 0 else float("nan")
        invalid_rate = float(invalid / int(self.succeeded)) if int(self.succeeded) > 0 else float("nan")
        return {
            "attempted": attempted,
            "succeeded": int(self.succeeded),
            "failed": failed,
            "skipped": skipped,
            "invalid": invalid,
            "fail_rate": fail_rate if math.isfinite(fail_rate) else None,
            "skip_rate": skip_rate if math.isfinite(skip_rate) else None,
            "invalid_rate": invalid_rate if math.isfinite(invalid_rate) else None,
            "top_failure_types": self.failures.top(int(top_k)),
            "top_skip_types": self.skips.top(int(top_k)),
            "top_invalid_reasons": self.invalid_reasons.top(int(top_k)),
            "invariant_problems": list(invariant_problems),
        }

    def evaluate(
        self,
        *,
        max_fail_rate: float | None = None,
        max_skips: int | None = None,
    ) -> tuple[RunStatus, list[str]]:
        reasons: list[str] = []
        invariant_problems = self._check_invariants()
        if invariant_problems:
            reasons.extend([f"invariant: {p}" for p in invariant_problems])
        if int(self.succeeded) <= 0:
            reasons.append("succeeded == 0")
            return "FAIL", reasons

        attempted = int(self.attempted)
        failed = int(self.failed)
        skipped = int(self.skipped)
        if max_skips is not None and int(max_skips) >= 0 and skipped > int(max_skips):
            reasons.append(f"skipped {skipped} > max_skips {int(max_skips)}")

        if max_fail_rate is not None:
            if attempted <= 0:
                reasons.append("attempted == 0")
            else:
                fail_rate = float(failed / attempted)
                if math.isfinite(fail_rate) and fail_rate > float(max_fail_rate):
                    reasons.append(f"fail_rate {fail_rate:.3f} > max_fail_rate {float(max_fail_rate):.3f}")

        if reasons:
            return "FAIL", reasons

        warn_reasons: list[str] = []
        if failed > 0 or skipped > 0:
            warn_reasons.append("nonzero failures/skips")
        if int(self.invalid) > 0:
            warn_reasons.append("invalid_results > 0")
        if warn_reasons:
            return "WARN", warn_reasons
        return "PASS", []
