"""selective — knowing when the agent is confidently wrong.

Calibration asks whether stated probabilities match observed frequencies.
Abstention needs something else: whether the score RANKS correctness. A
model emitting the base rate for every item is perfectly calibrated and
useless for abstention. Everything here is scored against the
random-ordering baseline.
"""

from __future__ import annotations

from .abstain import (
    RiskPoint,
    SelectiveReport,
    assess,
    aurc,
    coverage_at_risk,
    high_confidence_error_rate,
    optimal_threshold,
    random_baseline_aurc,
    risk_at_coverage,
    risk_coverage_curve,
)

__all__ = [
    "RiskPoint", "SelectiveReport", "assess", "aurc", "coverage_at_risk",
    "high_confidence_error_rate", "optimal_threshold", "random_baseline_aurc",
    "risk_at_coverage", "risk_coverage_curve",
]
__version__ = "0.1.0"
