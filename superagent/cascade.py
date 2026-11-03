"""Two-stage event filtering, and the recall its first stage silently caps.

The Super-agent architecture filters environmental events in two stages: a
deterministic Fast Path of programmatic policies (source allowlists, group
rules, bot filters) and a Slow Path where a model judges semantically. The
Fast Path exists to drop noise before any model is called, and it does.

It also sets a ceiling nobody measures.

**Cascade recall is multiplicative: R_total = R_fast x R_slow.** However good
the semantic stage becomes, total recall cannot exceed what the Fast Path
passed through. A programmatic rule that drops 2% of genuinely actionable
events caps the whole system at 98% and no model improvement recovers it.

The sharper problem is observability:

> **A Fast Path false negative is structurally invisible.** The event never
> reaches a model, never reaches a Worker, never reaches a log that anyone
> reviews, and produces no artefact at all. The system cannot know it
> happened. Nothing in the architecture will ever mention it.

This is the same shape as a silent permission under a default-allow policy,
with the sign flipped: there, omission grants; here, omission suppresses. In
both cases the omission is the one event class that generates no record.

Measuring Fast Path recall therefore requires something nobody does by
default: sample the events the Fast Path **dropped**, adjudicate them by
hand, and estimate the miss rate from that sample. `audit_fast_path` does the
arithmetic and refuses to report a ceiling when no dropped sample was
adjudicated, because the ceiling is unknown rather than high.

The module also applies the trivial-baseline test to event filtering. With a
low actionable base rate, "ignore everything" is a highly accurate filter,
and accuracy is therefore the wrong frame — exactly as it is for a permission
prompt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "cascade_recall",
    "StageAudit",
    "audit_fast_path",
    "CascadeReport",
    "assess_cascade",
    "filter_baseline",
    "FilterBaseline",
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------

def cascade_recall(fast_recall: float, slow_recall: float) -> float:
    """R_total = R_fast * R_slow. Stage recalls compose multiplicatively."""
    for name, v in (("fast_recall", fast_recall), ("slow_recall", slow_recall)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    return fast_recall * slow_recall


@dataclass(frozen=True)
class StageAudit:
    """Result of hand-adjudicating a sample of events a stage dropped.

    `n_sampled` is how many dropped events were reviewed. `n_actionable`
    is how many of those should have passed. Both must come from an
    adjudicated sample of the DROPPED stream — a sample of what passed
    tells you nothing about what was missed.
    """

    n_sampled: int
    n_actionable: int

    def __post_init__(self) -> None:
        if self.n_sampled < 0:
            raise ValueError("n_sampled must be non-negative")
        if not 0 <= self.n_actionable <= self.n_sampled:
            raise ValueError("n_actionable must be in [0, n_sampled]")


def audit_fast_path(
    n_events: int,
    n_passed: int,
    dropped_sample: StageAudit | None,
) -> tuple[float | None, tuple[float, float] | None, str]:
    """Estimate Fast Path recall from an adjudicated sample of what it dropped.

    Returns (recall, interval, note). Recall is None when no dropped sample
    was adjudicated — which is the default state of every deployment we have
    seen, and the honest report is "unknown", not "presumably fine".

    The estimate extrapolates the sample's actionable rate to the whole
    dropped stream, so it inherits that sample's representativeness. A sample
    drawn only from one noisy group chat will understate misses elsewhere.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")
    if not 0 <= n_passed <= n_events:
        raise ValueError("n_passed must be in [0, n_events]")

    n_dropped = n_events - n_passed
    if dropped_sample is None:
        return (
            None,
            None,
            "NO DROPPED SAMPLE ADJUDICATED. Fast Path recall is unknown, so "
            "the cascade's recall ceiling is unknown. This is not the same as "
            "high: nothing in the system records a Fast Path miss, so the "
            "absence of reported misses is not evidence of their absence.",
        )
    # Order matters: if nothing was dropped there was nothing to sample, so an
    # empty sample is complete rather than missing. Checking emptiness first
    # reports a fully-measured Fast Path as unmeasured.
    if n_dropped == 0:
        return (
            1.0,
            (1.0, 1.0),
            "the Fast Path dropped nothing, so it caps nothing. It is also "
            "doing no filtering, which was its purpose.",
        )
    if dropped_sample.n_sampled == 0:
        return (None, None, "dropped sample is empty; recall is unmeasured.")

    miss_rate = dropped_sample.n_actionable / dropped_sample.n_sampled
    est_missed = miss_rate * n_dropped
    # recall = actionable passed / total actionable. Treat everything passed as
    # a candidate: this is the optimistic reading and is stated as such.
    denom = n_passed + est_missed
    recall = 1.0 if denom == 0 else n_passed / denom
    lo_m, hi_m = _wilson(dropped_sample.n_actionable, dropped_sample.n_sampled)
    r_hi_denom = n_passed + lo_m * n_dropped
    r_lo_denom = n_passed + hi_m * n_dropped
    interval = (
        n_passed / r_lo_denom if r_lo_denom else 1.0,
        n_passed / r_hi_denom if r_hi_denom else 1.0,
    )

    if dropped_sample.n_actionable == 0:
        note = (
            f"no actionable events found in {dropped_sample.n_sampled} "
            f"adjudicated drops. The interval still runs down to "
            f"{interval[0]:.3f}, because a clean small sample bounds the miss "
            f"rate loosely."
        )
    else:
        note = (
            f"{dropped_sample.n_actionable}/{dropped_sample.n_sampled} "
            f"adjudicated drops were actionable, implying about "
            f"{est_missed:.0f} missed events across {n_dropped:,} drops. This "
            f"caps the whole cascade."
        )
    return recall, interval, note


# --------------------------------------------------------------------------
# the cascade report
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CascadeReport:
    n_events: int
    n_passed: int
    pass_rate: float
    fast_recall: float | None
    fast_recall_ci: tuple[float, float] | None
    slow_recall: float | None
    total_recall: float | None
    ceiling: float | None
    attainment: float | None
    compute_saved: float
    note: str
    fast_note: str

    def summary(self) -> str:
        lines = [
            f"{self.n_events:,} events, {self.n_passed:,} passed the Fast Path "
            f"({self.pass_rate:.4f})",
            f"  compute avoided by the Fast Path   {self.compute_saved:.1%} "
            f"of model calls",
        ]
        if self.fast_recall is None:
            lines.append("  Fast Path recall                   UNKNOWN")
            lines.append("  cascade recall ceiling             UNKNOWN")
        else:
            lo, hi = self.fast_recall_ci or (0.0, 1.0)
            lines.append(
                f"  Fast Path recall                   {self.fast_recall:.4f} "
                f"[{lo:.4f}, {hi:.4f}]"
            )
            lines.append(
                f"  cascade recall ceiling             {self.ceiling:.4f}"
                f"   <- no model improvement passes this"
            )
        if self.slow_recall is not None and self.total_recall is not None:
            lines.append(f"  Slow Path recall (given passed)    {self.slow_recall:.4f}")
            lines.append(f"  total cascade recall               {self.total_recall:.4f}")
            if self.attainment is not None:
                lines.append(
                    f"  attainment against the ceiling     {self.attainment:.4f}"
                    f"   <- how much of the headroom the model is using"
                )
        lines.append("")
        lines.append(f"  {self.fast_note}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def assess_cascade(
    n_events: int,
    n_passed: int,
    dropped_sample: StageAudit | None = None,
    slow_recall: float | None = None,
) -> CascadeReport:
    """Score a two-stage event filter, with the first stage's cap made explicit.

    `slow_recall` is the semantic stage's recall among events that reached it,
    which is measurable the ordinary way because those events produce records.
    `dropped_sample` is the only route to the Fast Path's recall, and without
    it the cascade's ceiling is unknown.
    """
    fast_recall, ci, fast_note = audit_fast_path(n_events, n_passed, dropped_sample)
    n_dropped = n_events - n_passed
    saved = n_dropped / n_events

    total = None
    attain = None
    if fast_recall is not None and slow_recall is not None:
        if not 0.0 <= slow_recall <= 1.0:
            raise ValueError(f"slow_recall must be in [0, 1], got {slow_recall}")
        total = cascade_recall(fast_recall, slow_recall)
        attain = None if fast_recall == 0 else total / fast_recall

    if fast_recall is None:
        note = (
            "Report the Slow Path's recall if you like; it is a conditional "
            "figure and does not bound the system. Until a dropped sample is "
            "adjudicated, the number that governs is unmeasured."
        )
    elif fast_recall < 0.99:
        note = (
            f"The Fast Path is the binding constraint. Improving the model "
            f"can raise total recall to at most {fast_recall:.4f}; the "
            f"remaining {1 - fast_recall:.2%} is lost in deterministic code "
            f"and is invisible to every downstream metric."
        )
    else:
        note = (
            "The Fast Path passes nearly all actionable events, so the Slow "
            "Path is the binding constraint and ordinary evaluation applies "
            "to it."
        )
    return CascadeReport(
        n_events=n_events,
        n_passed=n_passed,
        pass_rate=n_passed / n_events,
        fast_recall=fast_recall,
        fast_recall_ci=ci,
        slow_recall=slow_recall,
        total_recall=total,
        ceiling=fast_recall,
        attainment=attain,
        compute_saved=saved,
        note=note,
        fast_note=fast_note,
    )


# --------------------------------------------------------------------------
# the trivial baseline, for event filters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterBaseline:
    n_events: int
    n_actionable: int
    actionable_base_rate: float
    ignore_everything_accuracy: float
    filter_accuracy: float
    lift: float
    beats_ignoring: bool
    note: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.n_events:,} events, {self.n_actionable} actionable",
                f"  actionable base rate              "
                f"{self.actionable_base_rate:.4f}",
                f"  accuracy of ignoring everything   "
                f"{self.ignore_everything_accuracy:.4f}   <- trivial baseline",
                f"  accuracy of the filter            {self.filter_accuracy:.4f}",
                f"  lift                              {self.lift:+.4f}",
                "",
                f"  {'BEATS IGNORING' if self.beats_ignoring else 'DOES NOT BEAT IGNORING'}"
                f": {self.note}",
            ]
        )


def filter_baseline(
    n_events: int,
    n_actionable: int,
    n_correctly_flagged: int,
    n_false_flags: int = 0,
) -> FilterBaseline:
    """Does the filter beat ignoring every event?

    With a low actionable base rate, "ignore everything" is a very accurate
    filter. Any accuracy figure quoted for an event filter has to be read
    against it, exactly as a permission prompt's approval accuracy has to be
    read against clicking yes.
    """
    if n_events < 1:
        raise ValueError(f"n_events must be >= 1, got {n_events}")
    if not 0 <= n_actionable <= n_events:
        raise ValueError("n_actionable must be in [0, n_events]")
    if not 0 <= n_correctly_flagged <= n_actionable:
        raise ValueError("n_correctly_flagged must be in [0, n_actionable]")
    n_noise = n_events - n_actionable
    if not 0 <= n_false_flags <= n_noise:
        raise ValueError("n_false_flags must be in [0, n_events - n_actionable]")

    base = n_actionable / n_events
    ignore_acc = 1.0 - base
    correct = n_correctly_flagged + (n_noise - n_false_flags)
    filter_acc = correct / n_events
    lift = filter_acc - ignore_acc

    if n_actionable == 0:
        beats = False
        note = (
            "nothing in this sample was actionable, so the filter had nothing "
            "to find and cannot be shown to work."
        )
    elif lift <= 0:
        beats = False
        note = (
            f"the filter scores {filter_acc:.4f} and ignoring everything "
            f"scores {ignore_acc:.4f}. It is spending compute to do no better "
            f"than silence."
        )
    else:
        beats = True
        note = (
            f"the filter beats ignoring everything by {lift:+.4f}. Read this "
            f"next to its recall: accuracy is dominated by the noise class."
        )
    return FilterBaseline(
        n_events=n_events,
        n_actionable=n_actionable,
        actionable_base_rate=base,
        ignore_everything_accuracy=ignore_acc,
        filter_accuracy=filter_acc,
        lift=lift,
        beats_ignoring=beats,
        note=note,
    )
