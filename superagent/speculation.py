"""Speculative execution for agents, and the actions it must never cover.

The Super-agent architecture borrows speculative execution from computer
systems: the agent starts work in the background before knowing whether the
result will be used, and "results that turn out to be useful are delivered,
and the rest are discarded."

That sentence is sound for a CPU and unsound for an agent, for one reason.

> **Speculation runs before anyone decides. So every speculatively-executable
> action is, by construction, an action taken without approval.**

A mispredicted branch in a CPU is discarded by throwing away a register file.
A mispredicted agent action has already happened in the world. You cannot
discard a sent email, an un-notify a reviewer, or a deleted row.

So the set of actions eligible for speculation is not a performance tuning
knob. It is a permission decision, made once, in advance, for an entire class
of actions — and it must be a strict subset of actions that are both
reversible and invisible outside the principal.

The architecture's own worked example fails this test. From the paper: while
a group was still discussing a proposed change, the agent "has already opened
a Pull Request with a fix in the background." A Pull Request notifies
reviewers, appears in repository activity, and triggers CI. Closing it does
not retract any of that. It is a write, it is externally visible, and it was
performed before anybody approved it.

`may_speculate` classifies an action on two axes and refuses on anything that
is not plainly safe. It is deliberately conservative: the cost of a wrong
"no" is a slower agent, and the cost of a wrong "yes" is an action nobody
authorised.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

__all__ = [
    "Reversibility",
    "Visibility",
    "SpeculativeAction",
    "Verdict",
    "may_speculate",
    "SurfaceReport",
    "audit_surface",
]


class Reversibility(Enum):
    """Can the effect be undone, completely, by the agent, without a trace?"""

    NONE = "none"
    """Cannot be undone at all: a payment, a deletion, a sent message."""
    PARTIAL = "partial"
    """The state can be restored but the trace cannot: a closed PR, an
    edited-then-reverted file in a watched repo, a retracted message."""
    FULL = "full"
    """No effect outside the agent's own scratch space: a read, a local
    computation, a draft held in memory."""


class Visibility(Enum):
    """Who, besides the principal, can observe that this happened?"""

    PUBLIC = "public"
    """Third parties: reviewers, a group chat, customers, a public repo."""
    ORGANISATION = "organisation"
    """Colleagues who did not ask for this action."""
    PRINCIPAL_ONLY = "principal_only"
    """Only the person the agent is acting for."""
    NONE = "none"
    """Nobody; the effect is confined to the agent's own workspace."""


@dataclass(frozen=True)
class SpeculativeAction:
    """A candidate for background execution before anyone has decided."""

    name: str
    category: str
    reversibility: Reversibility
    visibility: Visibility
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an action needs a name")
        if not self.category:
            raise ValueError(
                f"action {self.name!r} must declare a category; an "
                f"uncategorised action cannot be covered by a policy"
            )


@dataclass(frozen=True)
class Verdict:
    action: str
    permitted: bool
    reason: str

    def __str__(self) -> str:
        mark = "SPECULATE" if self.permitted else "REFUSE   "
        return f"{mark}  {self.action:<28} {self.reason}"


def may_speculate(action: SpeculativeAction) -> Verdict:
    """May this action be run speculatively, before anyone has approved it?

    Permitted only when the effect is fully reversible AND invisible beyond
    the principal. Everything else is refused, with the reason stated.

    The asymmetry is deliberate. A refused speculation costs latency. A
    permitted one that should not have been is an unapproved action with an
    external trace, and there is no undo.
    """
    rev, vis = action.reversibility, action.visibility

    if rev is Reversibility.NONE:
        return Verdict(
            action.name,
            False,
            "irreversible. Speculation means acting before a decision; an "
            "irreversible action taken before a decision cannot be discarded.",
        )
    if rev is Reversibility.PARTIAL:
        return Verdict(
            action.name,
            False,
            "partially reversible: the state can be restored but the trace "
            "cannot. Closing a Pull Request does not un-notify its reviewers.",
        )
    if vis in (Visibility.PUBLIC, Visibility.ORGANISATION):
        return Verdict(
            action.name,
            False,
            f"reversible but visible to {vis.value}. Someone who did not ask "
            f"for this action will see it happen, which is itself an effect.",
        )
    return Verdict(
        action.name,
        True,
        f"fully reversible and visible to {vis.value} only. Safe to run "
        f"ahead; discard costs nothing.",
    )


# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceReport:
    n_actions: int
    verdicts: tuple[Verdict, ...]
    permitted: tuple[str, ...]
    refused: tuple[str, ...]
    unapproved_surface: float
    categories_permitted: tuple[str, ...]
    note: str

    def summary(self) -> str:
        lines = [f"{self.n_actions} candidate actions for speculative execution", ""]
        for v in self.verdicts:
            lines.append(f"  {v}")
        lines.append("")
        lines.append(
            f"  speculation surface: {len(self.permitted)}/{self.n_actions} "
            f"actions ({self.unapproved_surface:.1%})"
        )
        if self.categories_permitted:
            lines.append(
                f"  categories cleared for speculation: "
                f"{', '.join(self.categories_permitted)}"
            )
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def audit_surface(actions: Sequence[SpeculativeAction]) -> SurfaceReport:
    """What fraction of an agent's action set may run before anyone decides?

    The permitted set is the agent's **unapproved-action surface**. Sizing it
    is the whole governance question for speculative execution, and it is a
    number a team can compute from its own tool inventory in an afternoon.
    """
    if not actions:
        raise ValueError("no actions supplied")
    verdicts = tuple(may_speculate(a) for a in actions)
    permitted = tuple(v.action for v in verdicts if v.permitted)
    refused = tuple(v.action for v in verdicts if not v.permitted)
    by_name = {a.name: a for a in actions}
    cats = tuple(
        dict.fromkeys(by_name[n].category for n in permitted)
    )
    surface = len(permitted) / len(actions)

    write_like = [
        v.action
        for v, a in zip(verdicts, actions)
        if v.permitted and a.reversibility is not Reversibility.FULL
    ]
    if write_like:
        note = (
            "INCONSISTENT: an action was permitted that is not fully "
            "reversible. This should be unreachable; check the classification."
        )
    elif surface == 0.0:
        note = (
            "No action qualifies. Speculative execution buys nothing here "
            "beyond pre-reading context, which is still worth having — "
            "gathering background and drafting in memory are the safe cases."
        )
    elif surface > 0.5:
        note = (
            f"Over half the action set may run unapproved. Check the "
            f"reversibility classifications before accepting this; in our "
            f"experience 'fully reversible' is over-assigned, and anything "
            f"touching a shared system is at best partial."
        )
    else:
        note = (
            f"{len(permitted)} action(s) may run ahead of a decision, all "
            f"fully reversible and confined to the principal. Deliver the "
            f"results for approval; never let a speculative result become an "
            f"external effect without one."
        )
    return SurfaceReport(
        n_actions=len(actions),
        verdicts=verdicts,
        permitted=permitted,
        refused=refused,
        unapproved_surface=surface,
        categories_permitted=cats,
        note=note,
    )
