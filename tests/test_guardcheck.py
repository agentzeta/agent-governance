#!/usr/bin/env python3
"""Checks for guardcheck. Pure stdlib, self-contained, content-agnostic.

    python3 -I tests/test_guardcheck.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guardcheck as g

PASS = 0
FAIL: list[str] = []


def ok(cond: bool, label: str) -> None:
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(label)
        # Under pytest, fail loudly: a recorded-but-not-raised failure would
        # otherwise let pytest report this test as passing (a false green).
        if "pytest" in sys.modules:
            raise AssertionError(label)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def _stream(n_allow, n_block, wrongly_blocked, let_through):
    """Build a synthetic, content-free labelled stream and a guardrail's
    outcomes on it. Cases are bare ids; no content of any kind."""
    cases, outs = [], []
    for i in range(n_allow):
        cases.append(g.Case(f"ok{i}", should_allow=True))
        outs.append(g.Outcome(f"ok{i}", allowed=(i >= wrongly_blocked)))
    for i in range(n_block):
        cases.append(g.Case(f"bk{i}", should_allow=False))
        outs.append(g.Outcome(f"bk{i}", allowed=(i < let_through)))
    return cases, outs


# ==========================================================================
# trivial baselines
# ==========================================================================

def test_trivial_baselines() -> None:
    cases, _ = _stream(970, 30, 0, 0)
    b = g.trivial_baselines(cases)
    ok(close(b.block_base_rate, 0.03), "base rate")
    ok(close(b.allow_everything_accuracy, 0.97), "allow-everything accuracy = allow share")
    ok(close(b.deny_everything_accuracy, 0.03), "deny-everything accuracy = block share")
    ok("skewed base rate" in b.note, "skew flagged")
    ok(b.allow_everything_accuracy + b.deny_everything_accuracy == 1.0,
       "the two trivial accuracies partition the cases")

    # no cases should be blocked: the security number is unmeasurable
    allpos = [g.Case(f"ok{i}", True) for i in range(50)]
    b2 = g.trivial_baselines(allpos)
    ok(close(b2.block_base_rate, 0.0), "no block cases")
    ok("NO CASES SHOULD BE BLOCKED" in b2.note, "unmeasurable security number flagged")
    ok(close(b2.allow_everything_accuracy, 1.0), "allow-everything is perfect on an all-allow sample")

    # everything should be blocked
    allneg = [g.Case(f"bk{i}", False) for i in range(50)]
    b3 = g.trivial_baselines(allneg)
    ok("EVERY CASE SHOULD BE BLOCKED" in b3.note, "no utility measurement possible")

    # balanced base rate: still warns off accuracy
    bal = [g.Case(f"a{i}", True) for i in range(50)] + [g.Case(f"b{i}", False) for i in range(50)]
    b4 = g.trivial_baselines(bal)
    ok(close(b4.block_base_rate, 0.5), "balanced base rate")
    ok("read the error rates" in b4.note, "still points away from accuracy")

    ok(raises(lambda: g.trivial_baselines([]), ValueError), "empty rejected")


# ==========================================================================
# the headline: a do-nothing guardrail cannot hide behind accuracy
# ==========================================================================

def test_do_nothing_guardrail() -> None:
    # allow-everything guardrail on a 3%-block stream
    cases, _ = _stream(970, 30, 0, 0)
    outs = [g.Outcome(c.case_id, allowed=True) for c in cases]  # allows everything
    rep = g.evaluate(cases, outs)

    ok(close(rep.accuracy, 0.97), "allow-everything scores 97% accuracy")
    ok(rep.false_allow_rate is not None and close(rep.false_allow_rate, 1.0),
       "but its false-allow rate is 1.0 — it caught nothing that mattered")
    ok(rep.false_deny_rate is not None and close(rep.false_deny_rate, 0.0),
       "and its false-deny rate is 0.0")
    ok(rep.verdict == "NO_BETTER_THAN_DOING_NOTHING", "verdict is blunt")
    ok("allow-everything with extra steps" in rep.note, "named for what it is")
    ok("do not quote this alone" in rep.summary(), "accuracy is labelled unusable")

    # deny-everything guardrail: perfect security, zero utility
    outs2 = [g.Outcome(c.case_id, allowed=False) for c in cases]
    rep2 = g.evaluate(cases, outs2)
    ok(rep2.false_allow_rate is not None and close(rep2.false_allow_rate, 0.0),
       "deny-everything lets nothing through")
    ok(rep2.false_deny_rate is not None and close(rep2.false_deny_rate, 1.0),
       "but blocks every legitimate case")
    ok(rep2.verdict == "NO_BETTER_THAN_DOING_NOTHING", "deny-everything is also not a guardrail")
    ok("deny-everything is a competitor" in rep2.note, "utility collapse named")


def test_working_guardrail() -> None:
    # catches 24/30, lets 6 through, wrongly blocks 20/970
    cases, outs = _stream(970, 30, wrongly_blocked=20, let_through=6)
    rep = g.evaluate(cases, outs)
    ok(rep.blocked_correctly == 24, "true positives")
    ok(rep.let_through == 6, "false allows")
    ok(rep.blocked_wrongly == 20, "false denies")
    ok(rep.allowed_correctly == 950, "true negatives")
    ok(rep.false_allow_rate is not None and close(rep.false_allow_rate, 6 / 30),
       "false-allow rate = let-through / should-block")
    ok(rep.false_deny_rate is not None and close(rep.false_deny_rate, 20 / 970),
       "false-deny rate = wrongly-blocked / should-allow")
    ok(rep.block_recall is not None and close(rep.block_recall, 24 / 30), "block recall")
    ok(rep.verdict == "WORKS", "a real guardrail is recognised")
    ok("not optimal" in rep.note, "WORKS is carefully scoped")

    # intervals bracket the point estimates and are wide for the rare error
    lo, hi = rep.false_allow_ci
    ok(lo < rep.false_allow_rate < hi, "false-allow interval brackets estimate")
    ok(hi - lo > 0.2, "and is wide — 30 block cases is a small denominator")


def test_over_blocking_guardrail() -> None:
    # catches everything it should, but also blocks 300/970 legitimate cases
    cases, outs = _stream(970, 30, wrongly_blocked=300, let_through=0)
    rep = g.evaluate(cases, outs, max_false_deny=0.20)
    ok(rep.false_allow_rate is not None and close(rep.false_allow_rate, 0.0),
       "perfect catch rate")
    ok(rep.false_deny_rate is not None and rep.false_deny_rate > 0.20,
       "but false-deny exceeds tolerance")
    ok(rep.verdict == "NO_BETTER_THAN_DOING_NOTHING",
       "a guardrail that blocks too much legit traffic fails despite perfect security")
    ok("deny-everything is a competitor" in rep.note, "the comparison is named")

    # the same guardrail passes if the operator's tolerance is higher
    rep2 = g.evaluate(cases, outs, max_false_deny=0.50)
    ok(rep2.verdict == "WORKS", "tolerance is the operator's to set")


def test_lets_most_through() -> None:
    # catches only 5 of 30
    cases, outs = _stream(970, 30, wrongly_blocked=5, let_through=25)
    rep = g.evaluate(cases, outs)
    ok(rep.false_allow_rate is not None and rep.false_allow_rate > 0.5,
       "lets more than half through")
    ok(rep.verdict == "NO_BETTER_THAN_DOING_NOTHING", "and is judged so")
    ok("survived it" in rep.note, "the failure is stated plainly")


def test_unmeasurable() -> None:
    # a stream with nothing that should be blocked cannot measure a guardrail
    cases = [g.Case(f"ok{i}", True) for i in range(100)]
    outs = [g.Outcome(f"ok{i}", allowed=True) for i in range(100)]
    rep = g.evaluate(cases, outs)
    ok(rep.false_allow_rate is None, "no false-allow rate without block cases")
    ok(rep.verdict == "UNMEASURABLE", "verdict is UNMEASURABLE, not WORKS")
    ok("the whole point — is unmeasured" in rep.note, "reason given")
    ok("UNMEASURABLE" in rep.summary(), "surfaced in summary")

    # all-block stream: utility unmeasured
    cases2 = [g.Case(f"bk{i}", False) for i in range(100)]
    outs2 = [g.Outcome(f"bk{i}", allowed=False) for i in range(100)]
    rep2 = g.evaluate(cases2, outs2)
    ok(rep2.verdict == "UNMEASURABLE", "all-block is also unmeasurable")
    ok("utility damage is unmeasured" in rep2.note, "utility reason given")


def test_input_validation() -> None:
    cases, outs = _stream(10, 2, 0, 0)
    ok(raises(lambda: g.evaluate([], outs), ValueError), "no cases rejected")
    ok(raises(lambda: g.evaluate(cases, []), ValueError), "no outcomes rejected")
    # a case with no outcome is an error, not a silent drop
    ok(raises(lambda: g.evaluate(cases, outs[:-1]), ValueError), "missing outcome rejected")
    try:
        g.evaluate(cases, outs[:-1])
    except ValueError as e:
        ok("did not decide" in str(e), "missing-outcome message explains why")


# ==========================================================================
# operating points — only when the guardrail scores
# ==========================================================================

def test_operating_points() -> None:
    # a fixed-decision guardrail has no sweep
    cases, outs = _stream(50, 10, 2, 2)
    ok(g.sweep_operating_points(cases, outs) is None,
       "no scores means no sweep — not a fabricated one")

    # a scoring guardrail: higher score = more likely should-block.
    # Construct a clean separation so the sweep is checkable.
    cases2, outs2 = [], []
    for i in range(80):  # should-allow, low scores
        cases2.append(g.Case(f"ok{i}", True))
        outs2.append(g.Outcome(f"ok{i}", allowed=True, score=0.1 + 0.001 * i))
    for i in range(20):  # should-block, high scores
        cases2.append(g.Case(f"bk{i}", False))
        outs2.append(g.Outcome(f"bk{i}", allowed=False, score=0.8 + 0.001 * i))
    pts = g.sweep_operating_points(cases2, outs2)
    ok(pts is not None, "a scoring guardrail yields a sweep")
    ok(pts[0].false_allow_rate == 1.0, "highest threshold blocks nothing: all let through")
    ok(pts[0].false_deny_rate == 0.0, "and denies nothing")
    # somewhere in the middle there is a clean operating point
    best = min(pts, key=lambda p: p.false_allow_rate + p.false_deny_rate)
    ok(close(best.false_allow_rate, 0.0) and close(best.false_deny_rate, 0.0),
       "clean separation yields a zero-error operating point")
    ok(all(0.0 <= p.false_allow_rate <= 1.0 for p in pts), "rates stay in range")

    # partial scores are rejected rather than half-swept
    mixed = outs2[:]
    mixed[0] = g.Outcome("ok0", allowed=True, score=None)
    ok(raises(lambda: g.sweep_operating_points(cases2, mixed), ValueError),
       "mixed scored/unscored rejected")

    # a scored stream with no block cases: no sweep
    allpos = [g.Case(f"ok{i}", True) for i in range(10)]
    allpos_out = [g.Outcome(f"ok{i}", allowed=True, score=0.5) for i in range(10)]
    ok(g.sweep_operating_points(allpos, allpos_out) is None,
       "no block cases means no meaningful sweep")


# ==========================================================================

def main() -> int:
    for fn in (
        test_trivial_baselines,
        test_do_nothing_guardrail,
        test_working_guardrail,
        test_over_blocking_guardrail,
        test_lets_most_through,
        test_unmeasurable,
        test_input_validation,
        test_operating_points,
    ):
        fn()
    if FAIL:
        print(f"{PASS} passed, {len(FAIL)} FAILED:")
        for f in FAIL:
            print(f"  FAIL: {f}")
        return 1
    print(f"all green ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
