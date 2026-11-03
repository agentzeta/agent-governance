#!/usr/bin/env python3
"""Measuring a guardrail without being fooled by its accuracy.

    python3 -I labs/lab_guardrails.py

Every case here is synthetic and content-free: an id and a ground-truth label.
The "guardrails" are simulated decision functions, not real detectors. Nothing
in this file is, or produces, the kind of input a guardrail is meant to catch.
The subject is measurement.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guardcheck as g


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def stream(n_allow, n_block, wrongly_blocked, let_through, scored=False):
    """A synthetic labelled stream and a guardrail's outcomes on it."""
    cases, outs = [], []
    for i in range(n_allow):
        cases.append(g.Case(f"ok{i}", should_allow=True))
        s = {"score": 0.1 + 0.3 * (i / max(1, n_allow))} if scored else {}
        outs.append(g.Outcome(f"ok{i}", allowed=(i >= wrongly_blocked), **s))
    for i in range(n_block):
        cases.append(g.Case(f"bk{i}", should_allow=False))
        s = {"score": 0.6 + 0.3 * (i / max(1, n_block))} if scored else {}
        outs.append(g.Outcome(f"bk{i}", allowed=(i < let_through), **s))
    return cases, outs


# --------------------------------------------------------------------------
rule("1. A guardrail that blocks nothing is 97% accurate")

print("""Base rate matters. On a stream where 3% of cases should be blocked, a
guardrail that lets everything through scores 97% accuracy -- and catches
nothing it was installed to catch.""")
print()
cases, _ = stream(970, 30, 0, 0)
nothing = [g.Outcome(c.case_id, allowed=True) for c in cases]
print(g.evaluate(cases, nothing).summary())


# --------------------------------------------------------------------------
rule("2. The opposite failure: perfect security, no utility")

print("""A guardrail that blocks everything has a 0% false-allow rate. It also
blocks every legitimate request. 'Deny everything' is the other trivial
baseline, and a guardrail must beat it on utility just as it must beat
'allow everything' on security.""")
print()
everything = [g.Outcome(c.case_id, allowed=False) for c in cases]
print(g.evaluate(cases, everything).summary())


# --------------------------------------------------------------------------
rule("3. A guardrail that actually works")

print("""Catches 24 of 30, lets 6 through, wrongly blocks 20 of 970 legitimate
cases. Accuracy barely moves from the do-nothing baseline -- the whole story
is in the two error rates, each with an interval.""")
print()
cases, outs = stream(970, 30, wrongly_blocked=20, let_through=6)
print(g.evaluate(cases, outs).summary())

print("""
Note the false-allow interval: wide, because 30 block cases is a small
denominator. To tighten the security number you need more labelled cases that
should be blocked -- the expensive kind.""")


# --------------------------------------------------------------------------
rule("4. Perfect catch rate, and still not worth deploying")

print("""This guardrail catches everything it should -- false-allow 0.0 -- by
blocking 300 of 970 legitimate cases. At that utility cost, deny-everything is
a competitor, and the harness says so against a default 20% tolerance.""")
print()
cases, outs = stream(970, 30, wrongly_blocked=300, let_through=0)
print(g.evaluate(cases, outs, max_false_deny=0.20).summary())

print("""
The tolerance is yours to set. The same guardrail is judged WORKS at a 50%
tolerance -- which is the point: the harness measures, you decide the trade.""")


# --------------------------------------------------------------------------
rule("5. The number you cannot get from a clean sample")

print("""A guardrail evaluated only on cases that should be allowed cannot be
measured on the thing that matters. The verdict is UNMEASURABLE, and
UNMEASURABLE is not a pass.""")
print()
clean = [g.Case(f"ok{i}", should_allow=True) for i in range(100)]
clean_out = [g.Outcome(f"ok{i}", allowed=True) for i in range(100)]
print(g.evaluate(clean, clean_out).summary())


# --------------------------------------------------------------------------
rule("6. Operating points, when the guardrail emits a score")

print("""A fixed-decision guardrail has one operating point. One that emits a
risk score has a curve, and the operator should pick the threshold their cost
ratio justifies rather than accept a vendor default. Here, false-allow against
false-deny as the block threshold moves:""")
print()
cases, outs = stream(80, 20, wrongly_blocked=0, let_through=0, scored=True)
pts = g.sweep_operating_points(cases, outs)
if pts:
    for p in pts[::max(1, len(pts) // 12)]:
        print(p)


# --------------------------------------------------------------------------
rule("What this harness refuses to do")

print("""It never reports a single accuracy number as a guardrail's score.
It returns None rather than invent a false-allow rate from a sample that has no
cases to block. It calls a guardrail UNMEASURABLE rather than let it pass by
default. And it is content-agnostic: it measures a defence against your own
labelled cases, and it neither contains nor produces the inputs that defence
exists to stop.

A guardrail you could not measure is not a guardrail you may rely on.""")
print()
