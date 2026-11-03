"""superagent — measuring the two mechanisms a Super-agent architecture adds.

Event-driven agents introduce two things ordinary eval practice has no frame
for, and both are measurable without a model.

    cascade.py      two-stage event filtering. Cascade recall is
                    multiplicative, so the deterministic Fast Path caps the
                    whole system — and its false negatives are structurally
                    invisible, because a dropped event produces no artefact.

    speculation.py  speculative execution. Speculation runs before anyone
                    decides, so every speculatively-executable action is an
                    unapproved action. The eligible set must be a strict
                    subset of actions that are fully reversible AND invisible
                    beyond the principal — which the architecture's own
                    worked example (opening a Pull Request) is not.

Measurement and policy classification only. Nothing here issues, binds,
scopes or validates an agent's mandate, computes enforcement coverage or any
minimal enforcement set, or evaluates whether an authority is in force. Those
are separate mechanisms held back from publication; see permcheck's scope
boundary, which applies here unchanged.
"""

from __future__ import annotations

from .cascade import (
    CascadeReport,
    FilterBaseline,
    StageAudit,
    assess_cascade,
    audit_fast_path,
    cascade_recall,
    filter_baseline,
)
from .speculation import (
    Reversibility,
    SpeculativeAction,
    SurfaceReport,
    Verdict,
    Visibility,
    audit_surface,
    may_speculate,
)

__all__ = [
    "CascadeReport", "FilterBaseline", "StageAudit", "assess_cascade",
    "audit_fast_path", "cascade_recall", "filter_baseline",
    "Reversibility", "SpeculativeAction", "SurfaceReport", "Verdict",
    "Visibility", "audit_surface", "may_speculate",
]
__version__ = "0.1.0"
