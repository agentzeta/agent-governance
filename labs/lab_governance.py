#!/usr/bin/env python3
"""Permissions, speculation and confident wrongness, measured.

    python3 -I labs/lab_governance.py

Three results, each computed here: a permission prompt with a low deny rate
cannot be a control; under default-allow an unruled action is a silent
permission; and a perfectly calibrated model can be useless for abstention.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import permcheck
import selective
import superagent as sa
from superagent import Reversibility as R
from superagent import SpeculativeAction as A
from superagent import Visibility as V


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


class _Tool:
    """Minimal stand-in: permcheck only reads .category off a tool."""

    def __init__(self, name: str, category: str) -> None:
        self.name = name
        self.category = category


TOOLS = [
    _Tool("search", "read"),
    _Tool("read_file", "read"),
    _Tool("write_file", "write"),
    _Tool("send_email", "comms"),
    _Tool("pay", "money"),
]


# ------------------------------------------------------------------------
rule("6. A permission prompt with a 1% deny rate cannot be a control")

print("""The gate is one boolean, decided from a description of an action that
has not happened yet. Here is what asking a human achieves when 10 of 1,000
actions should be denied and the human catches every one of them:""")
print()
print(permcheck.prompt_burden(
    n_prompts=1000, n_should_deny=10, n_correctly_denied=10
).summary())

print("""
Perfect review buys one point of accuracy over clicking yes, at the price of
a thousand interruptions. Habituation is not carelessness here; it is the
correct response to a signal with that base rate.

The same prompt with a human who has learned to click yes:""")
print()
print(permcheck.prompt_burden(
    n_prompts=1000, n_should_deny=10, n_correctly_denied=0
).summary())

print("""
And a prompt worth showing — fired on fewer actions, so the base rate is
high enough for attention to pay:""")
print()
print(permcheck.prompt_burden(
    n_prompts=50, n_should_deny=20, n_correctly_denied=19
).summary())


# ------------------------------------------------------------------------
rule("7. Under default-allow, an unruled action is a silent permission")

space = permcheck.ActionSpace.from_tools(TOOLS)
print(f"action categories declared by the tools: {space.categories}")
print("""
Policy has rules for 'read', 'write' and 'comms'. Nobody wrote one for
'money'. The same gap, under the two stances:""")
print()
print(permcheck.coverage(space, ["read", "write", "comms"],
                        stance="default_deny").summary())
print()
print(permcheck.coverage(space, ["read", "write", "comms"],
                        stance="default_allow").summary())

print("""
Default-deny: a utility bug, and someone will hit it and complain.
Default-allow: the action is permitted, no rule authorises it, no log records
a decision about it, and nothing in the system will ever mention it.

The categories have to come from the tools, not the policy. A policy that
enumerates its own scope is unfalsifiable — which is why Tool requires a
category and refuses to construct without one.""")


# ------------------------------------------------------------------------
rule("8. Knowing when the agent is confidently wrong")

print("""Accuracy tells you how often it is wrong. Abstention needs something
else: whether confidence RANKS correctness. A model emitting the base rate
for every item is perfectly calibrated and useless — shuffling its scores
changes nothing, because there is no order to destroy.""")
print()
print(selective.assess([0.6] * 100, [True] * 60 + [False] * 40, trials=50).summary())

print("""
A signal that does rank:""")
print()
good = [0.95 - 0.001 * i for i in range(90)] + [0.1 - 0.001 * i for i in range(10)]
print(selective.assess(good, [True] * 90 + [False] * 10, trials=50).summary())

print("""
And the case a single AURC figure would have passed — a genuinely
informative ranking whose top decile is still worse than the overall error
rate. If high-confidence outputs go unreviewed, the top decile is the only
part that matters:""")
print()
ranked = [1.0 - 0.005 * i for i in range(200)]
print(selective.assess(ranked, [False] * 5 + [True] * 160 + [False] * 35,
                       trials=100).summary())

print("""
Finally: abstention is not free. When a refusal costs the user as much as a
wrong answer, the optimal policy is to answer everything, and the function
says so rather than returning a threshold of 0:""")
print()
scores = [0.9, 0.8, 0.7, 0.2, 0.1]
correct = [True, True, True, False, False]
print(f"  cost_wrong=1, cost_abstain=1  -> "
      f"{selective.optimal_threshold(scores, correct, 1.0, 1.0)}")
print(f"  cost_wrong=10, cost_abstain=1 -> "
      f"{selective.optimal_threshold(scores, correct, 10.0, 1.0)}")



# --------------------------------------------------------------------------
rule("Speculative execution is an unapproved action")

print("""Speculation runs before anyone decides. So every speculatively-
executable action is, by construction, an action taken without approval. A
mispredicted CPU branch is discarded by throwing away a register file; a
mispredicted agent action has already happened in the world.

Classifying a real tool inventory on reversibility and visibility:""")
print()
print(sa.audit_surface([
    A("read_file", "read", R.FULL, V.NONE),
    A("search_web", "read", R.FULL, V.NONE),
    A("draft_reply", "draft", R.FULL, V.PRINCIPAL_ONLY),
    A("open_pull_request", "write", R.PARTIAL, V.PUBLIC),
    A("send_email", "comms", R.NONE, V.PUBLIC),
    A("pay_invoice", "money", R.NONE, V.ORGANISATION),
]).summary())

print("""
The permitted set is the agent's unapproved-action surface. Sizing it is the
whole governance question for speculative execution, and any team can compute
it from its own tool inventory in an afternoon.""")


# --------------------------------------------------------------------------
rule("A deterministic pre-filter caps the whole system, invisibly")

print("""Cascade recall is multiplicative: R_total = R_fast * R_slow. However
good the semantic stage gets, total recall cannot exceed what the deterministic
stage passed -- and a dropped event reaches no model, no worker, and no log, so
the system cannot know it happened.

A 4% miss rate on the dropped stream, where 87% of events are dropped:""")
print()
print(sa.assess_cascade(300, 40, sa.StageAudit(50, 2), slow_recall=0.95).summary())

print("""
And the default state of every deployment: nobody adjudicates the drops.""")
print()
print(sa.assess_cascade(300, 40, None, slow_recall=0.95).summary())


# --------------------------------------------------------------------------
rule("What these share")

print("""Each is a case where a control exists, looks like it is working, and
is not -- and in each case what reveals it is a second number: the accuracy of
clicking yes, the stance that decides what an omission means, the AURC of
shuffled scores, the recall of what the filter threw away.

A function that cannot honestly answer returns None and reports the shortfall.
That rule is what makes a measurement a control rather than a dashboard.""")
print()
