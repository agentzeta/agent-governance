#!/usr/bin/env python3
"""Checks for superagent (cascade filtering, speculative-execution safety).

    python3 -I tests/test_superagent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import superagent as sa
from superagent import Reversibility as R
from superagent import SpeculativeAction as A
from superagent import Visibility as V

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
# cascade
# ==========================================================================

def test_cascade_recall() -> None:
    ok(close(sa.cascade_recall(1.0, 1.0), 1.0), "perfect cascade")
    ok(close(sa.cascade_recall(0.9, 0.9), 0.81), "recalls multiply")
    ok(close(sa.cascade_recall(0.0, 1.0), 0.0), "a stage that drops all caps at 0")
    ok(close(sa.cascade_recall(0.98, 1.0), 0.98),
       "a perfect model cannot exceed the Fast Path")
    # monotone: total never exceeds either stage
    for f in (0.5, 0.8, 0.98):
        for s in (0.5, 0.8, 0.98):
            t = sa.cascade_recall(f, s)
            ok(t <= f + 1e-12 and t <= s + 1e-12, f"cascade <= min stage ({f},{s})")
    ok(raises(lambda: sa.cascade_recall(1.5, 0.9), ValueError), "out of range rejected")


def test_stage_audit() -> None:
    ok(sa.StageAudit(50, 2).n_actionable == 2, "audit constructs")
    ok(raises(lambda: sa.StageAudit(10, 20), ValueError), "actionable <= sampled")
    ok(raises(lambda: sa.StageAudit(-1, 0), ValueError), "negative sample rejected")


def test_audit_fast_path() -> None:
    # THE default state: nobody adjudicates the dropped stream
    r, ci, note = sa.audit_fast_path(300, 40, None)
    ok(r is None, "no dropped sample means unknown recall")
    ok(ci is None, "and no interval")
    ok("NO DROPPED SAMPLE ADJUDICATED" in note, "stated loudly")
    ok("not the same as high" in note, "unknown is distinguished from fine")
    ok("absence of reported misses is not evidence" in note,
       "the invisibility argument is in the note")

    # an empty sample is not a sample
    r2, _, note2 = sa.audit_fast_path(300, 40, sa.StageAudit(0, 0))
    ok(r2 is None, "empty sample means unmeasured")
    ok("unmeasured" in note2, "said so")

    # a Fast Path that drops nothing caps nothing, and filters nothing
    r3, ci3, note3 = sa.audit_fast_path(100, 100, sa.StageAudit(0, 0))
    ok(r3 is not None and close(r3, 1.0), "dropping nothing gives recall 1")
    ok(ci3 == (1.0, 1.0), "degenerate interval")
    ok("doing no filtering" in note3, "and the point is made")

    # THE amplification: a small miss RATE on a large dropped stream is a
    # large absolute loss, because it scales with how much was dropped
    r4, ci4, note4 = sa.audit_fast_path(300, 40, sa.StageAudit(50, 2))
    ok(r4 is not None and 0.78 < r4 < 0.80, "4% miss rate gives ~0.79 recall")
    ok(ci4 is not None and ci4[0] < r4 < ci4[1], "interval brackets the estimate")
    ok("caps the whole cascade" in note4, "the cap is named")
    ok("implying about 10 missed" in note4, "absolute miss count given")

    # the same miss rate with a SMALLER dropped stream costs far less recall
    r5, _, _ = sa.audit_fast_path(300, 250, sa.StageAudit(50, 2))
    ok(r5 is not None and r4 is not None and r5 > r4,
       "the same miss rate hurts less when less is dropped")
    ok(r5 is not None and r5 > 0.99, "dropping only 50 events at 4% barely matters")

    # a clean sample still leaves a loose bound
    r6, ci6, note6 = sa.audit_fast_path(300, 40, sa.StageAudit(20, 0))
    ok(r6 is not None and close(r6, 1.0), "no misses found gives point estimate 1")
    ok(ci6 is not None and ci6[0] < 0.5,
       "but 20 clean samples bound the miss rate very loosely")
    ok("bounds the miss rate loosely" in note6, "the looseness is stated")

    ok(raises(lambda: sa.audit_fast_path(0, 0, None), ValueError), "n_events >= 1")
    ok(raises(lambda: sa.audit_fast_path(10, 20, None), ValueError), "passed <= events")


def test_assess_cascade() -> None:
    rep = sa.assess_cascade(300, 40, sa.StageAudit(50, 2), slow_recall=0.95)
    ok(close(rep.pass_rate, 40 / 300), "pass rate")
    ok(close(rep.compute_saved, 260 / 300), "compute saved")
    ok(rep.ceiling is not None and rep.ceiling == rep.fast_recall,
       "the ceiling IS the Fast Path's recall")
    ok(rep.total_recall is not None and rep.total_recall < 0.76, "total recall")
    ok(rep.attainment is not None and close(rep.attainment, 0.95),
       "attainment equals slow recall when fast is the ceiling")
    ok("binding constraint" in rep.note, "the Fast Path named as binding")
    ok("invisible to every downstream metric" in rep.note, "invisibility restated")
    ok("no model improvement passes this" in rep.summary(), "summary is blunt")

    # unknown ceiling
    rep2 = sa.assess_cascade(300, 40, None, slow_recall=0.95)
    ok(rep2.ceiling is None, "no ceiling without a dropped sample")
    ok(rep2.total_recall is None, "and no total recall")
    ok("UNKNOWN" in rep2.summary(), "summary says unknown")
    ok("does not bound the system" in rep2.note, "slow recall demoted")

    # a near-perfect Fast Path shifts the binding constraint to the model
    rep3 = sa.assess_cascade(1000, 900, sa.StageAudit(200, 0), slow_recall=0.9)
    ok(rep3.fast_recall is not None and rep3.fast_recall > 0.99, "fast recall high")
    ok("Slow Path is the binding constraint" in rep3.note, "constraint moves")

    ok(raises(lambda: sa.assess_cascade(300, 40, sa.StageAudit(50, 2), slow_recall=1.5),
              ValueError), "slow_recall range checked")


def test_filter_baseline() -> None:
    # THE trivial baseline, using the architecture paper's own workload:
    # ~300 messages per run, ~20 of them work requests
    b = sa.filter_baseline(300, 20, 20, 0)
    ok(close(b.actionable_base_rate, 20 / 300), "6.67% actionable")
    ok(close(b.ignore_everything_accuracy, 280 / 300), "ignoring scores 93.3%")
    ok(close(b.filter_accuracy, 1.0), "a perfect filter scores 100%")
    ok(close(b.lift, 20 / 300), "perfect filtering buys 6.67 points")
    ok(b.beats_ignoring, "it does beat ignoring")
    ok("dominated by the noise class" in b.note, "accuracy framing warned about")

    # a filter that flags nothing scores 93.3% and is useless
    b2 = sa.filter_baseline(300, 20, 0, 0)
    ok(close(b2.filter_accuracy, b2.ignore_everything_accuracy), "ties the baseline")
    ok(close(b2.lift, 0.0), "zero lift")
    ok(not b2.beats_ignoring, "tying the baseline does not beat it")
    ok("no better than silence" in b2.note, "the verdict is plain")

    # an over-eager filter can score BELOW ignoring everything
    b3 = sa.filter_baseline(300, 20, 18, 60)
    ok(b3.lift < 0, "over-flagging can be worse than silence")
    ok(not b3.beats_ignoring, "negative lift fails")

    # nothing actionable
    b4 = sa.filter_baseline(300, 0, 0, 0)
    ok(not b4.beats_ignoring, "no actionable events means nothing demonstrated")
    ok("nothing to find" in b4.note, "explained")
    ok(close(b4.ignore_everything_accuracy, 1.0), "ignoring is perfect here")

    ok(raises(lambda: sa.filter_baseline(0, 0, 0), ValueError), "n_events >= 1")
    ok(raises(lambda: sa.filter_baseline(10, 2, 5), ValueError), "flagged <= actionable")
    ok(raises(lambda: sa.filter_baseline(10, 2, 1, 50), ValueError),
       "false flags bounded by the noise count")


# ==========================================================================
# speculation
# ==========================================================================

def test_speculative_action() -> None:
    ok(A("read", "r", R.FULL, V.NONE).name == "read", "constructs")
    ok(raises(lambda: A("", "r", R.FULL, V.NONE), ValueError), "needs a name")
    ok(raises(lambda: A("x", "", R.FULL, V.NONE), ValueError), "needs a category")


def test_may_speculate() -> None:
    # the only permitted shape: fully reversible AND confined
    ok(may := sa.may_speculate(A("read_file", "read", R.FULL, V.NONE)),
       "a local read is permitted")
    ok(may.permitted, "permitted flag set")
    ok("Safe to run ahead" in may.reason, "reason given")

    draft = sa.may_speculate(A("draft_reply", "draft", R.FULL, V.PRINCIPAL_ONLY))
    ok(draft.permitted, "a draft only the principal sees is permitted")

    # THE architecture's own example fails
    pr = sa.may_speculate(A("open_pull_request", "write", R.PARTIAL, V.PUBLIC))
    ok(not pr.permitted, "opening a PR speculatively is refused")
    ok("un-notify" in pr.reason, "the trace argument is given")

    # irreversible is refused regardless of visibility
    for vis in (V.NONE, V.PRINCIPAL_ONLY, V.ORGANISATION, V.PUBLIC):
        v = sa.may_speculate(A("pay", "money", R.NONE, vis))
        ok(not v.permitted, f"irreversible refused at visibility {vis.value}")
        ok("irreversible" in v.reason, f"reason names irreversibility ({vis.value})")

    # partial is refused regardless of visibility
    for vis in (V.NONE, V.PRINCIPAL_ONLY, V.PUBLIC):
        v = sa.may_speculate(A("edit_shared_doc", "write", R.PARTIAL, vis))
        ok(not v.permitted, f"partial refused at visibility {vis.value}")

    # fully reversible but visible to others is still refused
    org = sa.may_speculate(A("post_status", "comms", R.FULL, V.ORGANISATION))
    ok(not org.permitted, "reversible but org-visible is refused")
    ok("did not ask for this action" in org.reason, "the reason is about consent")

    pub = sa.may_speculate(A("post_public", "comms", R.FULL, V.PUBLIC))
    ok(not pub.permitted, "reversible but public is refused")

    ok(str(pr).startswith("REFUSE"), "str renders a refusal")
    ok(str(may).startswith("SPECULATE"), "str renders a permission")


def test_audit_surface() -> None:
    actions = [
        A("read_file", "read", R.FULL, V.NONE),
        A("search_web", "read", R.FULL, V.NONE),
        A("draft_reply", "draft", R.FULL, V.PRINCIPAL_ONLY),
        A("open_pull_request", "write", R.PARTIAL, V.PUBLIC),
        A("send_email", "comms", R.NONE, V.PUBLIC),
        A("pay_invoice", "money", R.NONE, V.ORGANISATION),
    ]
    rep = sa.audit_surface(actions)
    ok(rep.n_actions == 6, "all actions scored")
    ok(len(rep.permitted) == 3, "three permitted")
    ok(len(rep.refused) == 3, "three refused")
    ok("open_pull_request" in rep.refused, "the PR is refused")
    ok("send_email" in rep.refused, "email is refused")
    ok("pay_invoice" in rep.refused, "payment is refused")
    ok(close(rep.unapproved_surface, 0.5), "half the surface")
    ok(set(rep.categories_permitted) == {"read", "draft"},
       "only read and draft categories clear")
    ok("never let a speculative result become an external effect" in rep.note,
       "the delivery rule is stated")

    # an all-write inventory clears nothing
    none_ok = sa.audit_surface([
        A("send_email", "comms", R.NONE, V.PUBLIC),
        A("deploy", "write", R.PARTIAL, V.ORGANISATION),
    ])
    ok(close(none_ok.unapproved_surface, 0.0), "nothing qualifies")
    ok(none_ok.categories_permitted == (), "no categories clear")
    ok("buys nothing here" in none_ok.note, "and that is said")
    ok("still worth having" in none_ok.note, "while noting the safe cases")

    # a suspiciously permissive inventory gets a warning
    permissive = sa.audit_surface([
        A(f"read_{i}", "read", R.FULL, V.NONE) for i in range(5)
    ] + [A("pay", "money", R.NONE, V.PUBLIC)])
    ok(permissive.unapproved_surface > 0.5, "over half permitted")
    ok("over-assigned" in permissive.note, "warns that FULL is over-assigned")

    # the permitted set never contains a non-FULL action: the invariant
    for acts in (actions, permissive.verdicts and actions):
        r = sa.audit_surface(actions)
        by_name = {a.name: a for a in actions}
        ok(
            all(by_name[n].reversibility is R.FULL for n in r.permitted),
            "invariant: every permitted action is fully reversible",
        )
        ok(
            all(
                by_name[n].visibility in (V.NONE, V.PRINCIPAL_ONLY)
                for n in r.permitted
            ),
            "invariant: every permitted action is confined to the principal",
        )
    ok("INCONSISTENT" not in rep.note, "no inconsistency reported on valid input")

    ok(raises(lambda: sa.audit_surface([]), ValueError), "empty inventory rejected")


# ==========================================================================

def main() -> int:
    for fn in (
        test_cascade_recall,
        test_stage_audit,
        test_audit_fast_path,
        test_assess_cascade,
        test_filter_baseline,
        test_speculative_action,
        test_may_speculate,
        test_audit_surface,
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
