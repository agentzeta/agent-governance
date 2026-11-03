"""permcheck — measuring whether a permission gate does any work.

Measurement only. It scores decisions a policy already made and audits logs
of actions already taken. It does not compute enforcement coverage or any
minimal enforcement set, does not issue, bind, scope or validate a mandate,
and does not evaluate whether an authority is currently in force. See the
scope boundary in policy.py.
"""

from __future__ import annotations

from .policy import (
    ActionSpace,
    ApprovalRecord,
    Coverage,
    Decision,
    DecisionQuality,
    MismatchReport,
    PromptBurden,
    TrailReport,
    approval_execution_audit,
    audit_trail,
    coverage,
    decision_quality,
    prompt_burden,
)

__all__ = [
    "ActionSpace", "ApprovalRecord", "Coverage", "Decision", "DecisionQuality",
    "MismatchReport", "PromptBurden", "TrailReport",
    "approval_execution_audit", "audit_trail", "coverage", "decision_quality",
    "prompt_burden",
]
__version__ = "0.1.0"
