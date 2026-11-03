"""Measuring a permission system: is the gate doing any work?

An agent permission gate is one boolean, decided from a description of an
action that has not happened yet. That is the whole interface. This package
measures what such a gate achieves, and the headline result is uncomfortable:

> **A permission prompt whose approval rate is 99% cannot be a control.**
> Approving everything scores 99% on the same data, costs nothing, and takes
> no attention. The prompt is not protecting anybody; it is training them to
> click yes.

This is the trivial-baseline discipline from benchmark leakage, applied to
permissions. The discriminating question is never "how often did the user
approve correctly" — it is "did careful review beat approving everything."
`prompt_burden` computes both and refuses to call a prompt a control when it
does not.

Four other things measured here:

- **Decision quality** as an asymmetric classification problem. False-allow
  is a security failure, false-deny a utility failure, their costs differ by
  orders of magnitude, and a single accuracy figure hides both.
- **Action-space coverage**: which actions the policy has a rule for. Under
  default-allow, an uncovered action is a silent permission.
- **Approval/execution mismatch**: whether what was executed matches what was
  approved, audited from the log after the fact.
- **Audit-trail completeness**: whether the log can reconstruct what happened.

---

## Scope boundary — deliberate, and stated rather than implied

This package is **measurement only**. It scores decisions that a policy
already made and audits logs of actions already taken. It does **not**:

- compute which enforcement points must be reached for a permission decision
  to be effective anywhere, or any minimal such set;
- compute, bound or certify coverage of an enforcement surface;
- issue, bind, scope or validate an agent's mandate;
- evaluate whether an authority is currently in force.

Those are separate mechanisms held back from publication. The line matters:
scoring a gate's decisions is an audit, and constructing the authority the
gate is supposed to be enforcing is not. Nothing in this package should grow
in that direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

__all__ = [
    "Decision",
    "DecisionQuality",
    "decision_quality",
    "PromptBurden",
    "prompt_burden",
    "ActionSpace",
    "Coverage",
    "coverage",
    "ApprovalRecord",
    "MismatchReport",
    "approval_execution_audit",
    "TrailReport",
    "audit_trail",
]

Stance = Literal["default_allow", "default_deny"]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# 1. decision quality, with the asymmetry kept visible
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """One permission decision, with ground truth.

    `should_allow` is the adjudicated answer — what a careful reviewer with
    full information concluded. Obtaining it is expensive and is the real
    cost of evaluating a permission system; there is no way around it.
    """

    action_id: str
    category: str
    allowed: bool
    should_allow: bool


@dataclass(frozen=True)
class DecisionQuality:
    n: int
    false_allow: int
    false_deny: int
    true_allow: int
    true_deny: int
    false_allow_rate: float | None
    false_allow_ci: tuple[float, float] | None
    false_deny_rate: float | None
    false_deny_ci: tuple[float, float] | None
    accuracy: float
    deny_base_rate: float
    note: str

    def summary(self) -> str:
        lines = [
            f"{self.n} decisions",
            "",
            "                     allowed    denied",
            f"  should allow     {self.true_allow:>8}  {self.false_deny:>8}",
            f"  should deny      {self.false_allow:>8}  {self.true_deny:>8}",
            "",
        ]
        if self.false_allow_rate is None:
            lines.append(
                "  false-allow rate   undefined — no action in this sample "
                "should have been denied"
            )
        else:
            lo, hi = self.false_allow_ci or (0.0, 1.0)
            lines.append(
                f"  false-allow rate   {self.false_allow_rate:.3f} "
                f"[{lo:.3f}, {hi:.3f}]   <- security failures"
            )
        if self.false_deny_rate is None:
            lines.append(
                "  false-deny rate    undefined — no action in this sample "
                "should have been allowed"
            )
        else:
            lo, hi = self.false_deny_ci or (0.0, 1.0)
            lines.append(
                f"  false-deny rate    {self.false_deny_rate:.3f} "
                f"[{lo:.3f}, {hi:.3f}]   <- utility failures"
            )
        lines.append(f"  accuracy           {self.accuracy:.3f}   <- do not use this")
        lines.append(f"  deny base rate     {self.deny_base_rate:.3f}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def decision_quality(decisions: Sequence[Decision]) -> DecisionQuality:
    """Score a policy's decisions, with the two error types kept apart.

    Accuracy is computed and labelled as unusable. When 1% of actions should
    be denied, a policy allowing everything is 99% accurate, and quoting that
    figure is how a gate that does nothing survives review.
    """
    if not decisions:
        raise ValueError("no decisions supplied")
    ta = sum(1 for d in decisions if d.allowed and d.should_allow)
    fa = sum(1 for d in decisions if d.allowed and not d.should_allow)
    fd = sum(1 for d in decisions if not d.allowed and d.should_allow)
    td = sum(1 for d in decisions if not d.allowed and not d.should_allow)
    n = len(decisions)

    n_should_deny = fa + td
    n_should_allow = ta + fd
    far = None if n_should_deny == 0 else fa / n_should_deny
    fdr = None if n_should_allow == 0 else fd / n_should_allow

    if far is None:
        note = (
            "NO NEGATIVE CASES. This sample contains nothing that should have "
            "been denied, so it cannot measure whether the gate denies anything. "
            "The false-allow rate is the number that matters and it is "
            "unmeasured here."
        )
    elif far > 0:
        note = (
            f"{fa} security failure(s). Weight these against false denials by "
            f"their actual cost ratio, not by count."
        )
    else:
        note = (
            "no false allows in this sample; check the sample size before "
            "reading that as a guarantee."
        )
    return DecisionQuality(
        n=n,
        false_allow=fa,
        false_deny=fd,
        true_allow=ta,
        true_deny=td,
        false_allow_rate=far,
        false_allow_ci=None if far is None else _wilson(fa, n_should_deny),
        false_deny_rate=fdr,
        false_deny_ci=None if fdr is None else _wilson(fd, n_should_allow),
        accuracy=(ta + td) / n,
        deny_base_rate=n_should_deny / n,
        note=note,
    )


# --------------------------------------------------------------------------
# 2. the headline result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptBurden:
    n_prompts: int
    deny_base_rate: float
    always_approve_accuracy: float
    reviewer_accuracy: float
    lift: float
    prompts_per_catch: float | None
    is_a_control: bool
    note: str

    def summary(self) -> str:
        lines = [
            f"{self.n_prompts} prompts shown to a human",
            f"  should-deny base rate     {self.deny_base_rate:.4f}",
            f"  accuracy of clicking yes  {self.always_approve_accuracy:.4f}"
            f"   <- the trivial baseline",
            f"  accuracy of the reviewer  {self.reviewer_accuracy:.4f}",
            f"  lift over clicking yes    {self.lift:+.4f}",
        ]
        if self.prompts_per_catch is None:
            lines.append(
                "  prompts per catch         infinite — nothing was caught"
            )
        else:
            lines.append(
                f"  prompts per catch         {self.prompts_per_catch:,.0f}"
            )
        lines.append("")
        lines.append(
            f"  {'IS A CONTROL' if self.is_a_control else 'NOT A CONTROL'}: {self.note}"
        )
        return "\n".join(lines)


def prompt_burden(
    n_prompts: int,
    n_should_deny: int,
    n_correctly_denied: int,
    n_incorrectly_denied: int = 0,
    min_lift: float = 0.0,
) -> PromptBurden:
    """Does asking the human beat not asking the human?

    The comparison is against the trivial policy of approving everything,
    which costs no attention and achieves `1 - deny_base_rate`. A prompt is a
    control only when careful review beats that by more than `min_lift`.

    `prompts_per_catch` is the attention price of each genuine catch. When it
    runs to thousands, habituation is not a user failing — it is the rational
    response to a signal with that base rate, and the design is what needs
    changing.
    """
    if n_prompts < 1:
        raise ValueError(f"n_prompts must be >= 1, got {n_prompts}")
    if not 0 <= n_should_deny <= n_prompts:
        raise ValueError("n_should_deny must be in [0, n_prompts]")
    if not 0 <= n_correctly_denied <= n_should_deny:
        raise ValueError("n_correctly_denied must be in [0, n_should_deny]")
    n_should_allow = n_prompts - n_should_deny
    if not 0 <= n_incorrectly_denied <= n_should_allow:
        raise ValueError("n_incorrectly_denied must be in [0, n_should_allow]")

    base = n_should_deny / n_prompts
    always_yes = 1.0 - base
    correct = n_correctly_denied + (n_should_allow - n_incorrectly_denied)
    reviewer = correct / n_prompts
    lift = reviewer - always_yes
    ppc = None if n_correctly_denied == 0 else n_prompts / n_correctly_denied

    if n_should_deny == 0:
        is_control = False
        note = (
            "nothing in this sample should have been denied, so the prompt had "
            "nothing to catch. It cannot be shown to be a control from data "
            "like this, however many prompts were answered correctly."
        )
    elif lift <= min_lift:
        is_control = False
        note = (
            f"careful review scores {reviewer:.4f} and clicking yes scores "
            f"{always_yes:.4f}. The prompt costs {n_prompts} interruptions and "
            f"buys {lift:+.4f}. It is training the user to approve, not "
            f"protecting them."
        )
    elif ppc is not None and ppc >= 100:
        is_control = True
        note = (
            f"review does beat clicking yes (+{lift:.4f}), but at "
            f"{ppc:,.0f} prompts per catch. Expect habituation; that is the "
            f"rational response to this base rate, not carelessness. Raise the "
            f"base rate by prompting on less, or accept that attention will "
            f"decay."
        )
    else:
        is_control = True
        note = (
            f"review beats clicking yes by {lift:+.4f} at "
            f"{ppc:,.0f} prompts per catch. The interruption is earning its cost."
        )
    return PromptBurden(
        n_prompts=n_prompts,
        deny_base_rate=base,
        always_approve_accuracy=always_yes,
        reviewer_accuracy=reviewer,
        lift=lift,
        prompts_per_catch=ppc,
        is_a_control=is_control,
        note=note,
    )


# --------------------------------------------------------------------------
# 3. action-space coverage
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionSpace:
    """The declared set of action categories an agent can perform.

    It has to come from the tools, not from the policy — a policy enumerating
    its own scope is unfalsifiable. `agentloop.Tool` requires a category for
    exactly this reason.
    """

    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError("an action space cannot be empty")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("duplicate categories in the action space")

    @classmethod
    def from_tools(cls, tools: Sequence[object]) -> "ActionSpace":
        cats: list[str] = []
        for t in tools:
            c = getattr(t, "category", None)
            if not c:
                raise ValueError(f"tool {t!r} has no category")
            if c not in cats:
                cats.append(c)
        return cls(tuple(cats))


@dataclass(frozen=True)
class Coverage:
    stance: Stance
    declared: tuple[str, ...]
    ruled: tuple[str, ...]
    uncovered: tuple[str, ...]
    undeclared_rules: tuple[str, ...]
    coverage: float
    silent_permissions: tuple[str, ...]
    note: str

    def summary(self) -> str:
        lines = [
            f"policy stance: {self.stance}",
            f"  action categories declared by tools: {len(self.declared)}",
            f"  categories with a rule:              {len(self.ruled)}",
            f"  coverage:                            {self.coverage:.3f}",
        ]
        for c in self.declared:
            mark = "rule" if c in self.ruled else "NO RULE"
            lines.append(f"    {c:<30}{mark}")
        if self.undeclared_rules:
            lines.append(
                f"  rules for undeclared categories: "
                f"{', '.join(self.undeclared_rules)}"
            )
        if self.silent_permissions:
            lines.append("")
            lines.append(
                f"  SILENT PERMISSIONS ({len(self.silent_permissions)}): "
                + ", ".join(self.silent_permissions)
            )
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def coverage(
    space: ActionSpace,
    ruled_categories: Sequence[str],
    stance: Stance = "default_deny",
) -> Coverage:
    """Which declared actions does the policy actually have a rule for?

    The stance decides what an uncovered category means, and the difference
    is total. Under default-deny an uncovered action is unavailable: a
    utility bug, visible the first time someone needs it. Under default-allow
    it is permitted with no rule, no record and nobody's decision — a
    **silent permission**, and nothing in the system will ever mention it.
    """
    ruled = tuple(dict.fromkeys(ruled_categories))
    declared = space.categories
    uncovered = tuple(c for c in declared if c not in ruled)
    undeclared = tuple(c for c in ruled if c not in declared)
    cov = (len(declared) - len(uncovered)) / len(declared)

    silent = uncovered if stance == "default_allow" else ()
    if silent:
        note = (
            f"Under default-allow these {len(silent)} categories are permitted "
            f"by omission. No rule authorises them and no log records a decision "
            f"about them."
        )
    elif uncovered:
        note = (
            f"Under default-deny these {len(uncovered)} categories are blocked "
            f"by omission. That is a utility bug and it is at least visible: "
            f"someone will hit it."
        )
    else:
        note = "every declared action category has a rule."
    if undeclared:
        note += (
            f" {len(undeclared)} rule(s) reference categories no tool declares — "
            f"either dead policy or a tool inventory that is out of date."
        )
    return Coverage(
        stance=stance,
        declared=declared,
        ruled=ruled,
        uncovered=uncovered,
        undeclared_rules=undeclared,
        coverage=cov,
        silent_permissions=silent,
        note=note,
    )


# --------------------------------------------------------------------------
# 4. approval / execution mismatch
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovalRecord:
    """What was shown to the approver, and what was then executed.

    `approved_description` is the string a human saw. `executed_description`
    is what the call actually was. These are compared as strings because that
    is what the human had: a description. If the two differ, the approval was
    given for something other than what happened, whatever the reason.
    """

    record_id: str
    approved_description: str
    executed_description: str
    approved: bool = True


@dataclass(frozen=True)
class MismatchReport:
    n: int
    n_approved: int
    exact_matches: int
    mismatches: int
    executed_without_approval: int
    mismatch_rate: float | None
    examples: tuple[tuple[str, str, str], ...]
    note: str

    def summary(self) -> str:
        lines = [f"{self.n} records, {self.n_approved} approved"]
        if self.mismatch_rate is None:
            lines.append("  mismatch rate: undefined (nothing was approved)")
        else:
            lines.append(
                f"  exact matches        {self.exact_matches}/{self.n_approved}"
            )
            lines.append(
                f"  mismatches           {self.mismatches} "
                f"({self.mismatch_rate:.3f})"
            )
        if self.executed_without_approval:
            lines.append(
                f"  EXECUTED WITHOUT APPROVAL  {self.executed_without_approval}"
            )
        for rid, appr, exe in self.examples:
            lines.append(f"    {rid}: approved {appr!r} -> executed {exe!r}")
        lines.append(f"  {self.note}")
        return "\n".join(lines)


def approval_execution_audit(
    records: Sequence[ApprovalRecord],
    max_examples: int = 3,
) -> MismatchReport:
    """Did what ran match what was approved?

    A log audit, after the fact, with one error mode worth naming: this
    compares descriptions, so a mismatch may be a cosmetic difference in how
    the call was rendered rather than a different action. Normalise the
    renderer before reading a high rate as adversarial. A *zero* rate is the
    informative case, and a nonzero one is a prompt to look at the examples.

    This measures a logged mismatch. It does not detect divergence in
    authority state, does not prevent a mismatch, and proposes no mechanism
    for binding an execution to its approval.
    """
    if not records:
        raise ValueError("no records supplied")
    approved = [r for r in records if r.approved]
    exact = sum(
        1 for r in approved if r.approved_description == r.executed_description
    )
    mism = len(approved) - exact
    unapproved_exec = sum(
        1 for r in records if not r.approved and r.executed_description
    )
    rate = None if not approved else mism / len(approved)

    examples = tuple(
        (r.record_id, r.approved_description, r.executed_description)
        for r in approved
        if r.approved_description != r.executed_description
    )[:max_examples]

    if unapproved_exec:
        note = (
            f"{unapproved_exec} action(s) executed after being denied or "
            f"without an approval record. Investigate these before the "
            f"mismatch rate."
        )
    elif rate is None:
        note = "nothing approved in this sample."
    elif rate == 0.0:
        note = (
            "every approved action executed exactly as described. Confirm the "
            "renderer is the same on both sides before relying on this."
        )
    else:
        note = (
            f"{mism} approved action(s) executed differently from their "
            f"description. Check whether the renderer differs before "
            f"concluding anything about intent."
        )
    return MismatchReport(
        n=len(records),
        n_approved=len(approved),
        exact_matches=exact,
        mismatches=mism,
        executed_without_approval=unapproved_exec,
        mismatch_rate=rate,
        examples=examples,
        note=note,
    )


# --------------------------------------------------------------------------
# 5. audit trail completeness
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrailReport:
    n_executed: int
    n_logged: int
    n_replayable: int
    logged_fraction: float
    replayable_fraction: float
    reconstructable: bool
    note: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.n_executed} executed actions",
                f"  logged at all        {self.n_logged} "
                f"({self.logged_fraction:.3f})",
                f"  logged in enough detail to replay  {self.n_replayable} "
                f"({self.replayable_fraction:.3f})",
                f"  {'RECONSTRUCTABLE' if self.reconstructable else 'NOT RECONSTRUCTABLE'}"
                f": {self.note}",
            ]
        )


def audit_trail(
    executed_ids: Sequence[str],
    logged_ids: Sequence[str],
    replayable_ids: Sequence[str],
) -> TrailReport:
    """Can you reconstruct what the agent did, from the log alone?

    Two thresholds, and only the second one matters. A log that records that
    something happened without recording what it was supports an incident
    count and not an incident investigation. `reconstructable` requires every
    executed action to be replayable, with no partial credit, because a
    reconstruction with gaps is not a reconstruction.
    """
    if not executed_ids:
        raise ValueError("no executed actions supplied")
    ex = set(executed_ids)
    lg = set(logged_ids) & ex
    rp = set(replayable_ids) & ex
    if not rp <= lg:
        raise ValueError("replayable actions must also be logged")

    lf = len(lg) / len(ex)
    rf = len(rp) / len(ex)
    ok = rf == 1.0
    if ok:
        note = "every executed action can be replayed from the log."
    elif lf == 1.0:
        note = (
            f"everything was logged, but {len(ex) - len(rp)} entries lack the "
            f"detail to replay. This supports counting incidents, not "
            f"investigating them."
        )
    else:
        note = (
            f"{len(ex) - len(lg)} executed action(s) have no log entry at all. "
            f"The trail cannot bound what happened."
        )
    return TrailReport(
        n_executed=len(ex),
        n_logged=len(lg),
        n_replayable=len(rp),
        logged_fraction=lf,
        replayable_fraction=rf,
        reconstructable=ok,
        note=note,
    )
