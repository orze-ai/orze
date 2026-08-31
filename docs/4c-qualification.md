# 4C qualification contract

This document defines how Orze may be evaluated under the
[4C Harness Evaluation Standard](https://github.com/warlockee/4c-harness).
It is a qualification contract, not a self-awarded score or rank.

## Terrain

The terrain is long-running, GPU-constrained ML research where a harness must
turn a finite experiment budget into independently verified model decisions.
The representative task starts from a preregistered set of candidate
configurations and ends only when the external evaluation contract accepts or
rejects every candidate and the resulting artifact lineage is reproducible.

Allowed effects are limited to the declared project paths, services, and
physical GPUs in the campaign manifest. Dataset identities, evaluation code,
environment identity, model lineage, allocation receipts, and externally
checkable result receipts are required evidence. Each campaign supplies its
own time, attempt, provider, and compute limits.

All four pressures are active:

- **Cost:** GPU-hours, wall time, provider calls, retries, duplicate work, and
  operator attention change which experiments can be attempted.
- **Compatibility:** trainers, evaluators, providers, model formats, and
  physical schedulers have semantics that must survive their adapters.
- **Continuity:** campaigns cross process failures, controller restarts,
  handoffs, and code or environment changes.
- **Cognition:** prior outcomes must change a versioned future research policy;
  retaining more logs or prompt context alone does not qualify.

## Anti-gaming rule

Any comparative score must use an exam frozen in the independent 4C repository
before candidate answers are opened. The exam must use the same questions,
weights, half/full-credit rules, and evidence cutoff for every candidate.
Missing evidence scores zero. Orze may challenge a score with pinned
counterevidence, but it must not receive candidate-specific questions or
weights.

A source score can establish only a source-predicted advantage. Competitive
status requires repeated paired trials under the same model/provider,
workload, limits, validity checks, and authority boundary. Those trials must
report the full yield vector and tails; a local proxy score, activity count,
or one successful run is insufficient.

## Competitive finish line

Orze is competitive for this terrain only when all of the following are true:

1. An independent, candidate-blind source exam places Orze in the leading
   cohort under the frozen contract.
2. A preregistered runtime comparison reaches `Task-proven`: more qualified
   outcomes per limiting GPU-hour or lower time to the same qualified outcome,
   with the paired confidence interval excluding no improvement.
3. The advantage survives cold/warm paths, p95 tails, a controller restart,
   an injected worker failure, and an invalid-candidate rejection.
4. Epistemic Access, Validity, and Authority all pass. Any failed boundary
   blocks promotion regardless of the 4C score.

The default Orze campaign targets remain minimum gates: allocation duty cycle
at least 90% while eligible work exists, zero duplicate launches, 100% invalid
work rejected before GPU allocation, first valid decision within four hours,
all declared decisions within 24 hours, qualified success rate at least 25%,
and at most eight GPU-hours per qualified success. A comparison may tighten
these thresholds but may not weaken them after registration.

## Boundary requirements

- **Epistemic Access:** exact data/access manifests must prove that training
  did not observe forbidden evaluation or private-label material. Missing
  access evidence is `UNVERIFIED`, not evidence of absence.
- **Validity:** an independent checker must qualify outputs. Official rank or
  private-set claims require the official receipt; local evaluation does not
  prove them. When a task requires one standalone model, composite, routing,
  or ensemble output is ineligible.
- **Authority:** campaign launch receipts must contain only the preregistered
  physical GPU allowlist. For the current qualification campaign that allowlist
  is GPUs 4–7; GPUs 0–3 are forbidden. External publication or submission is a
  separate authorized effect.

## Current evidence stage

The checked-in CPU control-plane receipt is `VERIFIED` at acceptance scale:
50,000 ideas and 20 steady-state samples, with no model execution or
accelerator access. It establishes only the measured control-plane properties
in that receipt. It does not establish GPU duty cycle, research yield,
standalone-model eligibility, absence of data leakage, or public rank.

- [CPU control-plane receipt](evidence/2026-08-30-cpu-control-plane.json)
- [Harness efficiency contract](harness-efficiency.md)
- [Research harness acceptance contract](research-harness-acceptance.md)

Until the frozen comparative exam and paired campaign finish, the public
status is **Mapped / partially runtime-observed / not yet Task-proven**. No 4C
rank is claimed here.
