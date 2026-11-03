# agent-governance

**A permission prompt with a 1% deny rate cannot be a control. Clicking yes scores 99% on the same data.**

Measurement for the three controls agent systems rely on and nobody has checked:
human approval, event pre-filtering, and the agent's own confidence.

Pure Python standard library. No dependencies, no network, no model.

```bash
git clone https://github.com/<you>/agent-governance && cd agent-governance
python3 -I tests/test_permcheck_selective.py    # 117 checks
python3 -I tests/test_superagent.py             # 117 checks
python3 -I tests/test_guardcheck.py             # 59 checks
python3 -I labs/lab_guardrails.py               # guardrail evaluation, computed live
python3 -I labs/lab_governance.py               # the results, computed live
```

---

## 1. Permission prompts, scored against clicking yes

An agent permission gate is one boolean, decided from a description of an action
that has not happened yet. That is the whole interface. Here is what asking a
human achieves when 10 of 1,000 actions should be denied and the human catches
**every one of them**:

```
1000 prompts shown to a human
  should-deny base rate     0.0100
  accuracy of clicking yes  0.9900   <- the trivial baseline
  accuracy of the reviewer  1.0000
  lift over clicking yes    +0.0100
  prompts per catch         100

  IS A CONTROL: review does beat clicking yes (+0.0100), but at 100 prompts
  per catch. Expect habituation; that is the rational response to this base
  rate, not carelessness.
```

Perfect vigilance buys **one point** of accuracy, for a thousand interruptions.

The same prompt with a human who has learned what it is worth:

```
  accuracy of the reviewer  0.9900
  lift over clicking yes    +0.0000
  NOT A CONTROL: the prompt costs 1000 interruptions and buys +0.0000.
  It is training the user to approve, not protecting them.
```

> **The discriminating question is never "how often did the user approve
> correctly." It is "did careful review beat approving everything?"**

A prompt that does not clear that baseline is not a weak control. It is not a
control, and its real function is to manufacture a record of consent.

Two consequences people resist:

**Habituation is rational.** At 100 prompts per catch, careful attention is
worth one tenth of one percent per prompt. Training, nagging and sterner wording
do not change the arithmetic. **Raising the base rate does** — prompt on less.

**Over-denial can be worse than the reflex.** A reviewer who catches 8 of 10 bad
actions but wrongly blocks 40 good ones scores **−0.0320 below clicking yes** —
while doing the job well by the usual standard. They caught 80% of the real
problems. The base rate defeated them.

### And accuracy is the wrong frame for the gate itself

A policy that allows everything, on data where 1% should be denied:

```
  accuracy           0.990   <- do not use this
  false-allow rate   1.000   [0.207, 1.000]   <- security failures
  false-deny rate    0.000   [0.000, 0.037]   <- utility failures
```

99% accurate, and it missed **every case that mattered**. `decision_quality`
also refuses on a sample containing nothing that should have been denied,
because such a sample cannot measure whether the gate denies anything.

---

## 2. Under default-allow, an unruled action is a silent permission

The action space comes from the **tools**, not the policy — `ActionSpace`
requires each tool to declare a category and refuses to construct without one.
A policy that enumerates its own scope is unfalsifiable; deriving the space from
the tool inventory is what makes coverage measurable.

Tools declare `read`, `write`, `comms`, `money`. Policy has rules for the first
three. The same gap, under the two stances:

| | default-deny | default-allow |
|---|---|---|
| coverage | 0.750 | 0.750 |
| uncovered | `money` | `money` |
| meaning | blocked by omission | **permitted by omission** |
| who finds out | the first person who needs it | **nobody** |

Under default-deny the gap is a utility bug and it self-reports: someone hits it
and complains. Under default-allow the action is permitted, **no rule authorises
it, no log records a decision about it, and nothing in the system will ever
mention it.**

That asymmetry is a stronger argument for default-deny than the usual security
framing, because it is about *observability* rather than risk appetite.

`coverage` also reports rules for categories no tool declares — dead policy, or
a tool inventory that has drifted. Neither shows up in any test.

---

## 3. Speculative execution is an unapproved action

Speculation runs work in the background before anyone knows whether the result
will be used, and the standard framing says useful results are delivered and the
rest discarded. Sound for a CPU, where a mispredicted branch is discarded by
throwing away a register file. Unsound for an agent:

> **Speculation runs before anyone decides. So every speculatively-executable
> action is, by construction, an action taken without approval.**

You cannot discard a sent email, un-notify a reviewer, or un-delete a row.

So eligibility is not a performance knob. It is a permission decision, made once
in advance for a whole class of actions, and it must be a strict subset of
actions that are **fully reversible** and **invisible beyond the principal**:

```
  SPECULATE  read_file            fully reversible and visible to none only.
  SPECULATE  search_web           fully reversible and visible to none only.
  SPECULATE  draft_reply          fully reversible and visible to principal_only.
  REFUSE     open_pull_request    partially reversible: the state can be restored
                                  but the trace cannot. Closing a Pull Request
                                  does not un-notify its reviewers.
  REFUSE     send_email           irreversible.
  REFUSE     pay_invoice          irreversible.

  speculation surface: 3/6 actions (50.0%)
  categories cleared for speculation: read, draft
```

The permitted set is the agent's **unapproved-action surface**, and sizing it is
the governance question. Any team can compute it from its own tool inventory in
an afternoon.

Two practical notes. "Fully reversible" is over-assigned in practice — anything
touching a shared system is at best partial — and `audit_surface` warns when
more than half an inventory clears. And an all-write inventory clears nothing,
which does not make speculation worthless: pre-reading context and drafting in
memory are the safe cases, and they are where most of the latency saving
actually comes from.

---

## 4. A deterministic pre-filter caps the whole system, invisibly

Event-driven agents filter in two stages: deterministic code drops noise, then a
model judges what survives. It saves real compute and it sets a ceiling nobody
measures.

**Cascade recall is multiplicative: `R_total = R_fast × R_slow`.** However good
the semantic stage gets, total recall cannot exceed what the first stage passed.
And:

> **A Fast Path false negative is structurally invisible.** The event reaches no
> model, no worker, and no log anyone reviews, and produces no artefact. The
> system cannot know it happened.

Same shape as a silent permission with the sign flipped — there omission grants,
here omission suppresses — and in both cases the omission is the one event class
that generates no record.

**The amplification is the part that surprises.** A miss *rate* on the dropped
stream is multiplied by *how much* was dropped. 300 events, 40 pass, and a
hand-adjudicated sample of 50 drops finds 2 that should have passed — a 4% miss
rate, which sounds tolerable:

```
300 events, 40 passed the Fast Path (0.1333)
  compute avoided by the Fast Path   86.7% of model calls
  Fast Path recall                   0.7937 [0.5334, 0.9331]
  cascade recall ceiling             0.7937   <- no model improvement passes this
  total cascade recall               0.7540

  2/50 adjudicated drops were actionable, implying about 10 missed events
  across 260 drops. This caps the whole cascade.
```

**A 4% miss rate costs 21% of recall**, because it is 4% of 260 drops against 40
passes. The same rate with only 50 events dropped leaves recall above 0.99.

Measuring it requires the one thing nobody does: **sample what the filter
dropped and adjudicate it by hand.** Sampling what passed tells you nothing.
Without that sample the honest report is *unknown*:

```
  Fast Path recall                   UNKNOWN
  cascade recall ceiling             UNKNOWN
  NO DROPPED SAMPLE ADJUDICATED. [...] This is not the same as high: nothing
  in the system records a Fast Path miss, so the absence of reported misses
  is not evidence of their absence.
```

Note also that 20 clean adjudicated samples bound the miss rate very loosely —
the interval runs below 0.5. Finding no misses in a small sample is weak
evidence.

---

## 5. Confidence must rank correctness, not merely be calibrated

An agent wrong 10% of the time is usable if you can tell **which** 10%.

| | asks |
|---|---|
| calibration | do the stated probabilities match observed frequencies? |
| discrimination | does a higher score mean more likely correct? |

Abstention needs the second. Emit the base rate as the confidence for every item
and you are perfectly calibrated by construction:

```
100 items, base error rate 0.400
  AURC                   0.0955  (lower is better)
  AURC, shuffled scores  0.0955 +/- 0.0000   <- the baseline to beat
  NOT USABLE FOR ABSTENTION: the scores do not rank correctness.
```

Look at that AURC. **0.0955 against a base error rate of 0.400** looks
excellent, and it is entirely an artefact of the order the items arrived in. The
shuffled baseline is the only thing that reveals it, because shuffling a
constant changes nothing.

So every report here is scored against the random-ordering baseline. And there
is a second guard — the **top-decile error rate**, which is "confidently wrong"
quantified — that catches a case a single AURC figure passes:

```
200 items, base error rate 0.200
  AURC                   0.1316
  AURC, shuffled scores  0.2044 +/- 0.0266
  z vs baseline          2.74              <- genuinely informative
  top-decile error rate  0.250 over 20 items   <- worse than the overall rate

  NOT USABLE FOR ABSTENTION: AURC beats the baseline, but the top-decile error
  rate 0.250 is at or above the overall rate 0.200.
```

A ranking can be informative in aggregate and inverted at the top. If
high-confidence outputs go unreviewed, the head is the only part that matters.

**And abstention is not free.** When a refusal costs the user as much as a wrong
answer, the optimal policy is to answer everything, and `optimal_threshold`
returns `None` saying so rather than a threshold of 0.

---

## 6. A guardrail that blocks nothing is 97% accurate

An output validator, an approval policy, a PII filter, a content classifier — a
guardrail is a classifier with asymmetric error costs and a skewed base rate.
Those three facts make accuracy close to meaningless and make two numbers
decisive:

- **false-allow rate** — of what should have been blocked, how much got through
  (the security number)
- **false-deny rate** — of what should have been allowed, how much was blocked
  (the utility number)

An allow-everything guardrail on a stream where 3% of cases should be blocked:

```
  false-allow rate    1.0000 [0.8865, 1.0000]   <- got through; the security number
  false-deny rate     0.0000 [0.0000, 0.0039]   <- blocked wrongly; the utility number
  accuracy            0.9700   <- do not quote this alone

  'allow everything' accuracy 0.9700   (false-allow 1.000, false-deny 0.000)
  'deny everything' accuracy  0.0300   (false-allow 0.000, false-deny 1.000)

  VERDICT: NO_BETTER_THAN_DOING_NOTHING — the guardrail blocked none of what it
  should have. It is allow-everything with extra steps.
```

`guardcheck` reports both error rates with Wilson intervals, against **both**
trivial baselines. A real guardrail must beat allow-everything on false-allow
*without* collapsing utility toward deny-everything — and the verdict enforces
it, including the case of a guardrail with a perfect catch rate that blocks so
much legitimate traffic that deny-everything is a competitor.

When the guardrail emits a risk score, `sweep_operating_points` traces
false-allow against false-deny across thresholds, so an operator picks the
operating point their cost ratio justifies rather than a vendor default:

```
  thr   0.780   false-allow 0.6000   false-deny 0.0000   catches 40.0%
  thr   0.660   false-allow 0.2000   false-deny 0.0000   catches 80.0%
  thr   0.385   false-allow 0.0000   false-deny 0.0500   catches 100.0%
```

**It is content-agnostic by construction.** A case is an identifier and a
ground-truth label; the harness never sees, stores, or generates the prompt or
output being judged — that is the operator's data, labelled by the operator. It
evaluates defences against your own labelled cases. It contains no examples of
what a guardrail is meant to catch, generates none, and needs none. **It
measures guardrails; it does not attack models.**

And it refuses: a guardrail evaluated on a sample with no cases that should be
blocked gets verdict `UNMEASURABLE`, not a pass — because that sample cannot
measure the number that matters, and a guardrail you could not measure is not
one you may rely on.

---

## The design rule

> **A function that cannot honestly answer returns `None` and reports the
> shortfall. It never relaxes a constraint to produce a number.**

A tool that always returns a number is a dashboard. In an organisation where
nobody can say "wait," a dashboard is the thing that gets pointed at.

---

## Scope boundary — deliberate, and stated rather than implied

This is **measurement and policy classification only.** It scores decisions a
policy already made and audits logs of actions already taken. It does **not**:

- compute which enforcement points must be reached for a permission decision to
  be effective anywhere, or any minimal such set;
- compute, bound or certify coverage of an enforcement surface;
- issue, bind, scope or validate an agent's mandate;
- evaluate whether an authority is currently in force.

And `guardcheck` is **content-agnostic**: it measures a defence against the
operator's own labelled outcomes, and it neither contains nor produces the
inputs a guardrail exists to stop. It evaluates defences; it does not generate
attacks, adversarial payloads, or the content a guardrail is meant to catch.

Scoring a gate's decisions is an audit. Constructing the authority the gate is
supposed to be enforcing is a different thing, and nothing here should grow in
that direction. The boundary is in the module docstrings as well as here, so it
travels with the code.

---

## An error this code caught in its own author

**An ordering bug in `audit_fast_path`.** The empty-sample check ran before the
nothing-was-dropped check, so a filter that dropped nothing — and therefore had
nothing to sample — was reported as *unmeasured* rather than as fully measured
with recall 1.0. Caught by a test asserting the degenerate case.

Worth noting what the degenerate case was: "the filter is doing nothing," which
is exactly the state a team is in on day one.

**A z-score computed from floating-point noise, on older Pythons only.** For a
model that emits one constant confidence, every shuffle gives the same AURC, so
the baseline's spread should be exactly zero and no z-score should exist. On
Python 3.12+ it is. On 3.10 and 3.11, summing fifty identical floats and
dividing by fifty does not return the original number, so the spread came out
as `2.8e-17` — and the code divided by it. Found by running the suite on every
Python version the CI matrix targets rather than only the one at hand. Fixed in
the code, not the test: identical values now return an exactly-zero spread,
summation is compensated, and a spread that is pure rounding residue is treated
as none. A regression check pins the exact zero; the old code fails it on 3.10.

## Layout

```
permcheck/policy.py        decision quality, prompt burden, action-space coverage,
                           approval/execution audit, trail completeness
superagent/cascade.py      cascade recall ceiling, Fast Path audit, filter baseline
superagent/speculation.py  reversibility × visibility, unapproved-action surface
selective/abstain.py       risk-coverage, AURC vs shuffled baseline, top-decile error
guardcheck/evaluate.py     false-allow/false-deny rates, trivial baselines,
                           operating-point sweep — content-agnostic
labs/lab_governance.py     permissions, speculation, abstention, computed live
labs/lab_guardrails.py     guardrail evaluation, computed live
tests/                     293 checks
```

## Licence

MIT.
