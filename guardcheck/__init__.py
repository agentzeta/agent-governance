"""guardcheck — measuring whether a guardrail does any work.

A guardrail (an output validator, an approval policy, a PII filter, a content
classifier) is a classifier with asymmetric error costs and a skewed base
rate, which makes accuracy meaningless and makes two numbers decisive:

    false-allow rate  — of what should have been blocked, how much got through
    false-deny  rate  — of what should have been allowed, how much was blocked

Both are reported against the two trivial baselines — allow-everything and
deny-everything — so a guardrail that does nothing cannot hide behind accuracy.

Content-agnostic: a case is an id and a ground-truth label. The harness
measures outcomes, never content. It contains no examples of what a guardrail
is meant to catch, generates none, and needs none. It evaluates defences; it
does not produce attacks.

House rule throughout: a function that cannot honestly answer returns None and
reports the shortfall. A guardrail you could not measure is not one you may
rely on.
"""

from __future__ import annotations

from .evaluate import (
    Case,
    GuardrailReport,
    OperatingPoint,
    Outcome,
    TrivialBaselines,
    evaluate,
    sweep_operating_points,
    trivial_baselines,
)

__all__ = [
    "Case", "GuardrailReport", "OperatingPoint", "Outcome", "TrivialBaselines",
    "evaluate", "sweep_operating_points", "trivial_baselines",
]
__version__ = "0.1.0"
