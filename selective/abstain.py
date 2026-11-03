"""Knowing when the agent is confidently wrong.

An agent that is wrong 10% of the time is usable if you can tell which 10%.
It is unusable if you cannot. The property that distinguishes those two
situations is not accuracy and it is not calibration — it is whether the
confidence signal **ranks** correctness.

This is the distinction the module exists to make measurable:

    calibration     do the stated probabilities match observed frequencies?
    discrimination  does a higher score mean more likely correct?

A model can be perfectly calibrated and useless for abstention. Emit the
base rate as the confidence for every item and you are perfectly calibrated
by construction, and your risk-coverage curve is flat — abstaining on the
lowest-confidence items removes a random sample, so the error rate among the
answered never improves. Calibration is a property of the marginal
distribution; abstention needs an ordering.

So every report here is scored **against the random-ordering baseline**, the
same discipline the leakage work uses. The discriminating question is not
"what is the AURC" but "did the AURC beat shuffling the scores."

The other number this module reports, because it is the one the question
actually asks for: the **error rate in the top confidence decile.** That is
"confidently wrong", quantified.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "RiskPoint",
    "risk_coverage_curve",
    "aurc",
    "random_baseline_aurc",
    "risk_at_coverage",
    "coverage_at_risk",
    "high_confidence_error_rate",
    "optimal_threshold",
    "SelectiveReport",
    "assess",
]


def _check(scores: Sequence[float], correct: Sequence[bool]) -> None:
    if len(scores) != len(correct):
        raise ValueError("scores and correct must be the same length")
    if not scores:
        raise ValueError("no items supplied")


# --------------------------------------------------------------------------
# the curve
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskPoint:
    coverage: float
    """Fraction of items answered."""
    risk: float
    """Error rate AMONG THE ANSWERED. Not over all items."""
    n_answered: int
    threshold: float


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[RiskPoint]:
    """Sort by confidence descending; answer the top m for each m.

    `risk` is the error rate among answered items, which is what a user
    experiences when the system abstains on the rest. It is not the error
    rate over the whole set.
    """
    _check(scores, correct)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = len(scores)
    out: list[RiskPoint] = []
    errors = 0
    for m, idx in enumerate(order, start=1):
        if not correct[idx]:
            errors += 1
        out.append(
            RiskPoint(
                coverage=m / n,
                risk=errors / m,
                n_answered=m,
                threshold=scores[idx],
            )
        )
    return out


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve. Lower is better.

    A perfect ranker puts every error last, so risk stays at 0 until the
    errors begin. A useless ranker holds risk at the base error rate
    throughout, giving an AURC equal to that rate.
    """
    curve = risk_coverage_curve(scores, correct)
    return sum(p.risk for p in curve) / len(curve)


def random_baseline_aurc(
    scores: Sequence[float],
    correct: Sequence[bool],
    trials: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """AURC under shuffled scores: mean and standard deviation.

    This is the number any real AURC must beat. Without it, an AURC of 0.09
    looks good and may be exactly what a coin would produce on data with a
    10% error rate.
    """
    _check(scores, correct)
    rng = random.Random(seed)
    shuffled = list(scores)
    vals: list[float] = []
    for _ in range(trials):
        rng.shuffle(shuffled)
        vals.append(aurc(shuffled, correct))
    # If every shuffle gave the same AURC there is no spread, and the answer
    # must be exactly zero. Computing it arithmetically is not safe: before
    # Python 3.12, sum() over identical floats can leave a rounding residue
    # (~1e-17), which a caller would then divide by.
    if max(vals) == min(vals):
        return vals[0], 0.0
    mean = math.fsum(vals) / len(vals)
    var = math.fsum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


# --------------------------------------------------------------------------
# operating points
# --------------------------------------------------------------------------

def risk_at_coverage(
    scores: Sequence[float], correct: Sequence[bool], target_coverage: float
) -> RiskPoint | None:
    """Error rate among answered when answering `target_coverage` of items.

    Returns None when the requested coverage is below one item — the honest
    answer at that point is that the sample cannot support the question.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    curve = risk_coverage_curve(scores, correct)
    eligible = [p for p in curve if p.coverage <= target_coverage]
    return eligible[-1] if eligible else None


def coverage_at_risk(
    scores: Sequence[float], correct: Sequence[bool], max_risk: float
) -> RiskPoint | None:
    """The most items answerable while keeping risk at or below `max_risk`.

    Returns None when no non-empty answered set meets the target, which is
    the useful answer: the system cannot operate at that risk level at any
    coverage, and no threshold choice will fix it.
    """
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError(f"max_risk must be in [0, 1], got {max_risk}")
    curve = risk_coverage_curve(scores, correct)
    feasible = [p for p in curve if p.risk <= max_risk]
    return max(feasible, key=lambda p: p.coverage) if feasible else None


def high_confidence_error_rate(
    scores: Sequence[float], correct: Sequence[bool], top_fraction: float = 0.10
) -> tuple[float, int] | None:
    """Error rate among the most confident items. "Confidently wrong."

    Returns None when the top fraction rounds to zero items. This figure is
    the one to put in front of anyone deciding whether to let an agent act
    unsupervised on its own high-confidence outputs, because those are
    exactly the actions that will not be reviewed.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    _check(scores, correct)
    n = len(scores)
    m = int(n * top_fraction)
    if m < 1:
        return None
    order = sorted(range(n), key=lambda i: -scores[i])[:m]
    errs = sum(1 for i in order if not correct[i])
    return errs / m, m


def optimal_threshold(
    scores: Sequence[float],
    correct: Sequence[bool],
    cost_wrong: float,
    cost_abstain: float,
) -> tuple[float, float, float] | None:
    """The threshold minimising expected cost. Returns (threshold, cost, coverage).

    Returns None when `cost_abstain >= cost_wrong`, because then abstaining
    is never worth it and the optimal policy is to answer everything — a
    conclusion worth reaching explicitly rather than by picking a threshold
    that happens to be 0. Teams routinely build abstention into systems where
    a refusal costs the user as much as an error, and the arithmetic says not
    to bother.
    """
    _check(scores, correct)
    if cost_wrong <= 0 or cost_abstain < 0:
        raise ValueError("cost_wrong must be positive and cost_abstain non-negative")
    if cost_abstain >= cost_wrong:
        return None

    n = len(scores)
    candidates = sorted(set(scores), reverse=True)
    best: tuple[float, float, float] | None = None
    for thr in candidates:
        answered = [i for i in range(n) if scores[i] >= thr]
        wrong = sum(1 for i in answered if not correct[i])
        abstained = n - len(answered)
        cost = (wrong * cost_wrong + abstained * cost_abstain) / n
        if best is None or cost < best[1]:
            best = (thr, cost, len(answered) / n)
    return best


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectiveReport:
    n: int
    base_error_rate: float
    aurc: float
    baseline_aurc: float
    baseline_sd: float
    z: float | None
    beats_baseline: bool
    high_conf_error: float | None
    high_conf_n: int | None
    usable_for_abstention: bool
    note: str

    def summary(self) -> str:
        lines = [
            f"{self.n} items, base error rate {self.base_error_rate:.3f}",
            f"  AURC                   {self.aurc:.4f}  (lower is better)",
            f"  AURC, shuffled scores  {self.baseline_aurc:.4f} "
            f"+/- {self.baseline_sd:.4f}   <- the baseline to beat",
        ]
        if self.z is None:
            lines.append("  z vs baseline          undefined (no baseline spread)")
        else:
            lines.append(f"  z vs baseline          {self.z:.2f}")
        if self.high_conf_error is None:
            lines.append("  top-decile error rate  undefined (too few items)")
        else:
            lines.append(
                f"  top-decile error rate  {self.high_conf_error:.3f} "
                f"over {self.high_conf_n} items   <- confidently wrong"
            )
        lines.append("")
        lines.append(
            f"  {'USABLE FOR ABSTENTION' if self.usable_for_abstention else 'NOT USABLE FOR ABSTENTION'}"
            f": {self.note}"
        )
        return "\n".join(lines)


def assess(
    scores: Sequence[float],
    correct: Sequence[bool],
    trials: int = 200,
    seed: int = 0,
    min_z: float = 2.0,
) -> SelectiveReport:
    """Is this confidence signal good enough to abstain on?

    `usable_for_abstention` is True only when the AURC beats the shuffled
    baseline by `min_z` standard deviations. A confidence signal that does not
    beat shuffling is not a weak signal, it is not a signal, and no threshold
    choice recovers one.
    """
    _check(scores, correct)
    base = sum(1 for c in correct if not c) / len(correct)
    a = aurc(scores, correct)
    b_mean, b_sd = random_baseline_aurc(scores, correct, trials=trials, seed=seed)
    # A spread that is pure rounding noise is no spread: dividing by it would
    # turn floating-point residue into a z-score.
    z = None if b_sd <= 1e-12 * max(1.0, abs(b_mean)) else (b_mean - a) / b_sd
    beats = z is not None and z >= min_z

    hce = high_confidence_error_rate(scores, correct)
    hc_err, hc_n = hce if hce is not None else (None, None)

    if base == 0.0:
        usable = False
        note = (
            "nothing is wrong in this sample, so there is nothing to abstain "
            "on and no ordering can be demonstrated. Score this on data "
            "containing errors."
        )
    elif not beats:
        usable = False
        note = (
            f"AURC {a:.4f} against a shuffled baseline of {b_mean:.4f}. The "
            f"scores do not rank correctness. Abstaining on the lowest "
            f"confidence items would remove a roughly random sample, so it "
            f"would not reduce the error rate among the rest."
        )
    elif hc_err is not None and hc_err >= base:
        usable = False
        note = (
            f"AURC beats the baseline, but the top-decile error rate "
            f"{hc_err:.3f} is at or above the overall rate {base:.3f}. The "
            f"signal ranks somewhere in the distribution and not at the top, "
            f"which is the only part that matters if high-confidence outputs "
            f"go unreviewed."
        )
    else:
        usable = True
        note = (
            f"AURC {a:.4f} beats the shuffled baseline {b_mean:.4f} at "
            f"z={z:.2f}"
            + (
                f", and the top decile errs at {hc_err:.3f} against "
                f"{base:.3f} overall."
                if hc_err is not None
                else "."
            )
        )
    return SelectiveReport(
        n=len(scores),
        base_error_rate=base,
        aurc=a,
        baseline_aurc=b_mean,
        baseline_sd=b_sd,
        z=z,
        beats_baseline=beats,
        high_conf_error=hc_err,
        high_conf_n=hc_n,
        usable_for_abstention=usable,
        note=note,
    )
