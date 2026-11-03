#!/usr/bin/env python3
"""Checks for agentloop, permcheck, selective and budget.

    python3 -I tests/test_agentstack.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import permcheck
import selective

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


# ==========================================================================
# agentloop
# ==========================================================================

class _T:
    """Minimal stand-in for a tool: permcheck only reads .category."""
    def __init__(self, category):
        self.category = category


_FakeTools = [_T("read"), _T("read"), _T("write"), _T("comms"), _T("money")]


def _dec(i, cat, allowed, should):
    return permcheck.Decision(f"a{i}", cat, allowed, should)


def test_permcheck_decisions() -> None:
    # a gate that allows everything, on data where 1% should be denied
    ds = [_dec(i, "read", True, True) for i in range(99)] + [
        _dec(99, "money", True, False)
    ]
    q = permcheck.decision_quality(ds)
    ok(close(q.accuracy, 0.99), "allow-everything scores 99% accuracy")
    ok(q.false_allow == 1, "one security failure")
    ok(q.false_allow_rate is not None and close(q.false_allow_rate, 1.0),
       "false-allow rate is 1.0 — it missed every case that mattered")
    ok(q.false_deny_rate is not None and close(q.false_deny_rate, 0.0), "no false denies")
    ok("do not use this" in q.summary(), "accuracy labelled unusable")
    ok("security failure" in q.note, "security failure named")

    # no negative cases at all: the number that matters is unmeasured
    allpos = [_dec(i, "read", True, True) for i in range(50)]
    q2 = permcheck.decision_quality(allpos)
    ok(q2.false_allow_rate is None, "no negatives means no false-allow rate")
    ok("NO NEGATIVE CASES" in q2.note, "absence of negatives called out")
    ok(close(q2.accuracy, 1.0), "perfect accuracy on a sample that tests nothing")

    # a gate that works
    good = (
        [_dec(i, "read", True, True) for i in range(80)]
        + [_dec(80 + i, "money", False, False) for i in range(18)]
        + [_dec(98, "money", True, False)]
        + [_dec(99, "read", False, True)]
    )
    q3 = permcheck.decision_quality(good)
    ok(q3.false_allow == 1 and q3.false_deny == 1, "one of each error")
    ok(q3.false_allow_rate is not None and close(q3.false_allow_rate, 1 / 19),
       "false-allow rate over should-deny cases")
    ok(q3.false_allow_ci is not None and q3.false_allow_ci[0] < q3.false_allow_rate,
       "interval brackets below")
    ok(raises(lambda: permcheck.decision_quality([]), ValueError), "empty rejected")


def test_permcheck_prompt_burden() -> None:
    # THE headline: a prompt with a 1% deny base rate, answered perfectly
    b = permcheck.prompt_burden(
        n_prompts=1000, n_should_deny=10, n_correctly_denied=10
    )
    ok(close(b.always_approve_accuracy, 0.99), "clicking yes scores 0.99")
    ok(close(b.reviewer_accuracy, 1.0), "the reviewer was perfect")
    ok(close(b.lift, 0.01), "perfect review buys one point")
    ok(b.prompts_per_catch is not None and close(b.prompts_per_catch, 100.0),
       "100 prompts per catch")
    ok(b.is_a_control, "it does beat clicking yes, so it is a control")
    ok("habituation" in b.note, "habituation flagged at 100 prompts per catch")

    # the same prompt, with a reviewer who clicks yes
    b2 = permcheck.prompt_burden(
        n_prompts=1000, n_should_deny=10, n_correctly_denied=0
    )
    ok(not b2.is_a_control, "a reviewer who approves everything is not a control")
    ok(close(b2.lift, 0.0), "zero lift")
    ok(b2.prompts_per_catch is None, "no catches means no prompts-per-catch")
    ok("training the user to approve" in b2.note, "the real effect named")

    # a reviewer who catches most but also over-denies can score NEGATIVE lift
    b3 = permcheck.prompt_burden(
        n_prompts=1000, n_should_deny=10, n_correctly_denied=8,
        n_incorrectly_denied=40,
    )
    ok(b3.lift < 0, "over-denial can make review worse than clicking yes")
    ok(not b3.is_a_control, "negative lift is not a control")

    # nothing to catch
    b4 = permcheck.prompt_burden(n_prompts=500, n_should_deny=0, n_correctly_denied=0)
    ok(not b4.is_a_control, "a prompt with nothing to catch is not a control")
    ok("nothing to catch" in b4.note, "explained")
    ok(close(b4.always_approve_accuracy, 1.0), "clicking yes is perfect here")

    # a well-targeted prompt: high base rate, few prompts per catch
    b5 = permcheck.prompt_burden(
        n_prompts=50, n_should_deny=20, n_correctly_denied=19
    )
    ok(b5.is_a_control, "a targeted prompt is a control")
    ok(b5.prompts_per_catch is not None and b5.prompts_per_catch < 10,
       "few prompts per catch")
    ok("earning its cost" in b5.note, "positive verdict")

    ok(raises(lambda: permcheck.prompt_burden(0, 0, 0), ValueError), "n_prompts >= 1")
    ok(raises(lambda: permcheck.prompt_burden(10, 20, 0), ValueError), "deny <= prompts")
    ok(raises(lambda: permcheck.prompt_burden(10, 2, 5), ValueError), "caught <= should deny")
    ok(raises(lambda: permcheck.prompt_burden(10, 2, 1, 50), ValueError),
       "over-denials bounded by should-allow count")


def test_permcheck_coverage() -> None:
    space = permcheck.ActionSpace.from_tools(_FakeTools)
    ok(space.categories == ("read", "write", "comms", "money"),
       "categories taken from tools, deduplicated, in declaration order")
    ok(raises(lambda: permcheck.ActionSpace(()), ValueError), "empty space rejected")
    ok(raises(lambda: permcheck.ActionSpace(("a", "a")), ValueError), "duplicates rejected")
    ok(raises(lambda: permcheck.ActionSpace.from_tools([object()]), ValueError),
       "uncategorised tool rejected")

    # THE stance finding: the same gap means opposite things
    deny = permcheck.coverage(space, ["read", "write", "comms"], stance="default_deny")
    allow = permcheck.coverage(space, ["read", "write", "comms"], stance="default_allow")
    ok(deny.uncovered == ("money",), "money has no rule")
    ok(allow.uncovered == ("money",), "same gap under either stance")
    ok(close(deny.coverage, 3 / 4), "three of four covered")
    ok(deny.silent_permissions == (), "default-deny: the gap blocks")
    ok(allow.silent_permissions == ("money",), "default-allow: the gap PERMITS")
    ok("utility bug" in deny.note, "default-deny gap is a utility bug")
    ok("permitted by omission" in allow.note, "default-allow gap is a silent permission")
    ok("SILENT PERMISSIONS" in allow.summary(), "summary shouts about it")
    ok("NO RULE" in deny.summary(), "summary marks the uncovered category")

    full = permcheck.coverage(space, ["read", "write", "comms", "money"])
    ok(close(full.coverage, 1.0), "complete coverage")
    ok(full.uncovered == (), "nothing uncovered")
    ok("every declared action category has a rule" in full.note, "clean note")

    # dead policy
    stale = permcheck.coverage(space, ["read", "write", "comms", "money", "ftp"])
    ok(stale.undeclared_rules == ("ftp",), "rule for a category no tool declares")
    ok("dead policy" in stale.note, "dead policy flagged")


def test_permcheck_audit() -> None:
    R = permcheck.ApprovalRecord
    clean = [R(f"r{i}", "pay(amount=10)", "pay(amount=10)") for i in range(5)]
    rep = permcheck.approval_execution_audit(clean)
    ok(rep.mismatch_rate is not None and close(rep.mismatch_rate, 0.0), "no mismatches")
    ok(rep.exact_matches == 5, "all exact")
    ok("renderer is the same on both sides" in rep.note, "renderer caveat stated")

    # approved one thing, executed another
    mixed = clean + [R("bad", "pay(amount=10)", "pay(amount=10000)")]
    rep2 = permcheck.approval_execution_audit(mixed)
    ok(rep2.mismatches == 1, "one mismatch")
    ok(rep2.mismatch_rate is not None and close(rep2.mismatch_rate, 1 / 6), "rate")
    ok(len(rep2.examples) == 1, "example surfaced")
    ok("10000" in rep2.examples[0][2], "the executed value is shown")
    ok("renderer differs" in rep2.note, "alternative explanation offered first")

    # executed after denial — the worst case, reported first
    denied = clean + [R("x", "pay(amount=5)", "pay(amount=5)", approved=False)]
    rep3 = permcheck.approval_execution_audit(denied)
    ok(rep3.executed_without_approval == 1, "execution after denial counted")
    ok("EXECUTED WITHOUT APPROVAL" in rep3.summary(), "summary shouts")
    ok("Investigate these before" in rep3.note, "prioritised over mismatch rate")

    ok(raises(lambda: permcheck.approval_execution_audit([]), ValueError), "empty rejected")

    # audit trail
    t = permcheck.audit_trail(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"])
    ok(t.reconstructable, "fully replayable")
    ok(close(t.replayable_fraction, 1.0), "fraction 1")

    t2 = permcheck.audit_trail(["a", "b", "c"], ["a", "b", "c"], ["a"])
    ok(not t2.reconstructable, "logged but not replayable is not reconstructable")
    ok(close(t2.logged_fraction, 1.0), "everything logged")
    ok("counting incidents, not" in t2.note, "the distinction stated")

    t3 = permcheck.audit_trail(["a", "b", "c"], ["a"], ["a"])
    ok(close(t3.logged_fraction, 1 / 3), "partial logging")
    ok("no log entry at all" in t3.note, "missing entries named")
    ok(not t3.reconstructable, "not reconstructable")

    ok(raises(lambda: permcheck.audit_trail([], [], []), ValueError), "empty rejected")
    ok(raises(lambda: permcheck.audit_trail(["a"], [], ["a"]), ValueError),
       "replayable must be logged")


# ==========================================================================
# selective
# ==========================================================================

def test_selective() -> None:
    # a perfect ranker: all errors have the lowest scores
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    correct = [True, True, True, False, False]
    curve = selective.risk_coverage_curve(scores, correct)
    ok(len(curve) == 5, "one point per coverage level")
    ok(close(curve[0].risk, 0.0), "risk 0 at minimum coverage")
    ok(close(curve[2].risk, 0.0), "risk still 0 after three correct")
    ok(close(curve[-1].risk, 0.4), "full coverage gives the base error rate")
    ok(close(curve[-1].coverage, 1.0), "coverage reaches 1")
    ok(selective.aurc(scores, correct) < 0.4, "a good ranker beats the base rate")

    # THE finding: a perfectly calibrated constant predictor cannot abstain.
    # Every score identical, so no ordering exists.
    flat = [0.6] * 100
    flat_correct = [True] * 60 + [False] * 40
    rep = selective.assess(flat, flat_correct, trials=50)
    ok(close(rep.base_error_rate, 0.4), "base error rate 0.4")
    ok(close(rep.baseline_sd, 0.0), "shuffling constant scores changes nothing")
    ok(rep.z is None, "no z when the baseline has no spread")
    ok(rep.baseline_sd == 0.0,
       "constant scores give exactly zero spread on every Python version "
       "(before 3.12, float summation leaves a ~1e-17 residue)")
    ok(not rep.usable_for_abstention, "a constant score cannot support abstention")
    ok("do not rank correctness" in rep.note, "the reason given")

    # a genuinely informative signal
    good_scores = [0.95 - 0.001 * i for i in range(90)] + [0.1 - 0.001 * i for i in range(10)]
    good_correct = [True] * 90 + [False] * 10
    rep2 = selective.assess(good_scores, good_correct, trials=50)
    ok(rep2.usable_for_abstention, "a perfect ranker is usable")
    ok(rep2.aurc < rep2.baseline_aurc, "AURC beats the shuffled baseline")
    ok(rep2.z is not None and rep2.z >= 2.0, "comfortably significant")
    ok(rep2.high_conf_error is not None and close(rep2.high_conf_error, 0.0),
       "no confidently-wrong items")

    # Errors concentrated at the TOP of the confidence range. Note which check
    # catches this: the AURC advantage is destroyed outright, because risk at
    # low coverage is computed over few items so each early error moves it a lot.
    inverted = [0.99] * 10 + [0.5] * 90
    inv_correct = [False] * 10 + [True] * 80 + [False] * 10
    rep3 = selective.assess(inverted, inv_correct, trials=50)
    ok(rep3.high_conf_error is not None and close(rep3.high_conf_error, 1.0),
       "the top decile is wrong every time")
    ok(not rep3.usable_for_abstention, "confidently wrong at the top is not usable")
    ok(not rep3.beats_baseline, "ten errors at the top destroy the AURC advantage")
    ok(rep3.z is not None and rep3.z < 0, "AURC is worse than shuffling")

    # The case that exercises the SECOND guard: the ranking is genuinely
    # informative (z=2.7) and the top decile is still worse than the base
    # rate. A single AURC figure would have passed this.
    ranked = [1.0 - 0.005 * i for i in range(200)]
    r_correct = [False] * 5 + [True] * 160 + [False] * 35
    rep5 = selective.assess(ranked, r_correct, trials=100)
    ok(rep5.beats_baseline, "the ranking does beat shuffling here")
    ok(rep5.z is not None and rep5.z > 2.0, "and significantly so")
    ok(rep5.high_conf_error is not None and close(rep5.high_conf_error, 0.25),
       "top decile errs at 0.25")
    ok(close(rep5.base_error_rate, 0.20), "against a base rate of 0.20")
    ok(not rep5.usable_for_abstention,
       "good AURC plus a bad top decile is still not usable")
    ok("not at the top" in rep5.note, "the second guard's reason given")

    # high-confidence error rate
    hce = selective.high_confidence_error_rate(scores, correct, top_fraction=0.4)
    ok(hce is not None and close(hce[0], 0.0) and hce[1] == 2, "top 40% all correct")
    ok(selective.high_confidence_error_rate([0.5], [True], 0.1) is None,
       "too few items for a top decile returns None")

    # operating points
    rp = selective.risk_at_coverage(scores, correct, 0.6)
    ok(rp is not None and close(rp.risk, 0.0) and rp.n_answered == 3, "risk at 60%")
    cp = selective.coverage_at_risk(scores, correct, 0.0)
    ok(cp is not None and close(cp.coverage, 0.6), "most coverage at zero risk")
    ok(selective.coverage_at_risk([0.5, 0.4], [False, False], 0.0) is None,
       "no feasible operating point returns None")

    # abstention is pointless when it costs as much as being wrong
    ok(selective.optimal_threshold(scores, correct, 1.0, 1.0) is None,
       "equal costs: abstention never pays")
    ok(selective.optimal_threshold(scores, correct, 1.0, 2.0) is None,
       "abstaining dearer than erring: never abstain")
    opt = selective.optimal_threshold(scores, correct, 10.0, 1.0)
    ok(opt is not None and opt[2] < 1.0, "costly errors justify some abstention")
    ok(opt is not None and opt[1] >= 0.0, "cost is non-negative")

    # validation
    ok(raises(lambda: selective.aurc([], []), ValueError), "empty rejected")
    ok(raises(lambda: selective.aurc([1.0], [True, False]), ValueError),
       "length mismatch rejected")
    ok(raises(lambda: selective.risk_at_coverage(scores, correct, 0.0), ValueError),
       "zero coverage rejected")
    ok(raises(lambda: selective.high_confidence_error_rate(scores, correct, 0.0), ValueError),
       "zero top fraction rejected")
    ok(raises(lambda: selective.optimal_threshold(scores, correct, 0.0, 0.0), ValueError),
       "non-positive cost_wrong rejected")

    # a sample with no errors cannot demonstrate ranking
    rep4 = selective.assess([0.9, 0.8, 0.7], [True, True, True], trials=20)
    ok(not rep4.usable_for_abstention, "no errors means nothing to abstain on")
    ok("nothing to abstain on" in rep4.note, "explained")


# ==========================================================================
# budget
# ==========================================================================

def main() -> int:
    for fn in (
        test_permcheck_decisions,
        test_permcheck_prompt_burden,
        test_permcheck_coverage,
        test_permcheck_audit,
        test_selective,
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
