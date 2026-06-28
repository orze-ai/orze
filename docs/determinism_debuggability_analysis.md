# Determinism & Debuggability — Input & Objective Layer

Status: design / proposal
Owner: erik@boson.ai
Scope: research-role config generation, project kwarg schema, eval-objective
noise, structured event observability, engine-vs-project boundary
Out of scope: mechanical-glue races / silent skips / implicit lifecycle state —
those are covered by the companion doc [`determinism_hardening.md`](determinism_hardening.md)

## How this doc relates to `determinism_hardening.md`

`determinism_hardening.md` hardens the **mechanical layer underneath the LLM**:
filesystem-as-message-queue races, silent skips, and idea-lifecycle state
inferred from file presence (failure modes F1–F3). It explicitly leaves the
LLM/prompt and eval layers out of scope.

This doc covers exactly those two left-out surfaces, which are the other half of
"make the system more deterministic and debuggable":

| Surface | Owner doc |
|---|---|
| FS races, silent skips, implicit lifecycle state | `determinism_hardening.md` (F1–F3) |
| **LLM config generation (made-up / mis-shaped configs)** | this doc (G1) |
| **Eval objective noise (shaky scoreboard)** | this doc (G2) |
| **Structured event observability (log archaeology)** | this doc (G3), overlaps hardening #2 |
| **Engine-vs-project boundary discipline** | this doc (cross-cutting) |

The two docs share the same thesis: **the leverage is NOT a LangChain/LangGraph
migration.** It is hardening the deterministic layers around appropriately
LLM-driven judgment.

## Goal (plain statement)

Make the research loop **predictable** (same input → same decision) and
**inspectable** (one queryable place to see what happened and why), without
replacing the SOP/role system that is doing its job.

## Three failure surfaces (input & objective layer)

### G1. Free-form LLM config generation — the #1 input-side stochastic surface

The research role emits experiment configs as free-form YAML against a project
`train.py` kwarg contract. With no schema gate at generation time, it routinely
emits keys that don't exist or are mis-shaped. Observed in the 1.7b (ASR) run
over ~16h after the v4.4.4 restart:

| Rejected shape | Count | Verdict |
|---|---|---|
| `decoding:` wrapper block | 72 | hallucinated wrapper around a mix of real + fake keys |
| `eval_fix_ngram_loops`, `oversample_short`, `rejection_gate`, … | ~10 | scalar/strategy values wrongly nested, or non-existent keys |
| `per_dataset_audio_min_duration_s` (nested) | 1 (then fixed) | **legit** key, was a false rejection — now whitelisted |

The `decoding:` block is the instructive case — it mixes **real** flat args
(`enable_thinking`, `max_new_tokens`, `do_sample`, `eos_logit_bias_by_duration`)
with **non-existent** ones (`fix_ngram_loops`, `mbr_enabled`, `contrastive_decode`).
Blanket-unwrapping would promote the fake keys to top level where `train.py`
silently ignores them — re-introducing the exact wiring-gap bug the F5 nested
validator exists to prevent.

**Why band-aids don't converge.** v4.4.4 added a config-driven normalizer
(`normalize_nested_config`, `flatten_prefix` / `rename:<key>`) that rescues
*known* mis-shapes before the validator. It works (26 rescues observed in one
window) but it is reactive: every new hallucinated shape needs a new rule. The
root cause is that the allowed-key schema lives in **three drifting places** —
the `train.py` argparse, the `nested_config_whitelist` in `orze.yaml`, and the
`RESEARCH_RULES.md` prose. Drift between them *is* the bug.

**Fix (mechanism vs. policy split):**
- **Engine mechanism (general):** let each project *declare* a kwarg schema; the
  engine validates generated configs against it and, where the provider supports
  it, constrains generation (structured output / function-call contract) so
  invalid shapes are unrepresentable rather than rejected after the fact.
- **Project policy (ASR-supplied):** the schema itself, ideally derived from a
  single source of truth (`train.py` argparse) so the whitelist and the role
  prompt are generated from it, not hand-maintained in parallel.

This keeps the engine domain-neutral (it must never hardcode `num_beams`) while
killing the hallucination class at the source for every project.

### G2. Eval objective noise — the shaky scoreboard (ASR-specific magnitude, general principle)

`train.py` itself documents **AMI seed variance of 7–39% WER spread**. When the
objective is that noisy, promotion decisions partly chase luck: an idea can be
crowned champion on a noise-sized margin. This is the source of the "is the
champion 5.04 or 5.08?" confusion — 5.04 was a transient per-sample running
figure; the benchmark champion is 5.085%.

**Fix (principle is general — "denoise the objective"; specifics are ASR):**
- Pin `eval_seed` and a fixed sample set for benchmark eval.
- Use greedy decode (`do_sample=false`) for the scoring run so the metric is a
  function of the model, not the sampler.
- Gate promotion on **mean ± confidence interval over fixed seeds**, not a single
  point estimate. Only crown a new champion when it clears the incumbent by more
  than the noise band.

The engine-general version of this is: the **promotion rule** should accept a
project-supplied noise model / CI threshold rather than comparing raw scalars.

### G3. Log archaeology — debuggability gap

Answering "what happened today?" currently means `grep`+`awk` over
`/tmp/orze_*.log` (this is how the G1 counts above were produced). orze already
ships the right primitive — **`orze.Journal`**, an append-only structured log
(`.md` or `.jsonl`) that survives restart/compaction and is built for
programmatic aggregation.

**Fix:** route idea-lifecycle events
(`idea_generated → normalized → rejected/accepted → launched → eval_done →
promoted`) through `Journal` `.jsonl`, then expose an aggregate view (e.g.
`orze` rejection report) so the G1 table becomes a one-line query instead of a
hand-scrape. This overlaps and reinforces hardening doc **#2** (instrument
soft-failure credit signals) and **#5** (idea-transition audit table).

## Existing primitives we should use (instead of a new framework)

| Need | Already in orze | Status |
|---|---|---|
| Structured, restart-safe event log | `orze.Journal` (`journal/`) | exists, underused for lifecycle events |
| Declarative state machine | `orze.fsm` (`fsm/engine.py`, `fsm/runner.py`) | exists; lifecycle FSM is hardening doc #5 |
| Procedure receipts / wiring checks | SOP system (`orze sop check/status`) | exists |
| Per-project nested-key rescue | `normalize_nested_config` (v4.4.4) | shipped; reactive band-aid for G1 |

Conclusion echoed from `determinism_hardening.md`: a 50-line sqlite/FSM/Journal
approach covers the ground a LangGraph/LangChain migration would, without the
dependency weight, and — critically — without freezing the SOP role loops that
provide adaptive judgment.

## Engine vs. project boundary (cross-cutting discipline)

A recurring theme: keep determinism mechanisms in the **engine** (domain-neutral)
and the specifics in the **project**.

| Concern | Engine (general orze) | ASR project (1.7b) |
|---|---|---|
| Config validity | schema-gate mechanism, normalizer, validators | the kwarg schema (from `train.py`) |
| Objective noise | promotion rule accepts a CI/noise threshold | WER seed variance, greedy decode policy |
| Observability | `Journal`, FSM, receipts | which lifecycle events matter |

v4.4.4's normalizer already follows this split (engine provides the
`flatten_prefix`/`rename` mechanism; `orze.yaml` supplies the map). Holding this
boundary is itself a determinism + debuggability win: it stops project specifics
from drifting into the engine.

## Recommended sequencing (input & objective layer)

1. **G1 — single source-of-truth kwarg schema + generation-time gate.** Highest
   ROI; directly ends the hallucination/rejection churn we are firefighting.
2. **G3 — Journal `.jsonl` lifecycle events + rejection aggregate view.** Low
   risk, immediate debuggability; reinforces hardening #2/#5.
3. **G2 — deterministic eval (seed-pin + greedy + CI-gated promotion).** Stops
   the search from chasing metric noise; ASR-specific implementation.

Sequence against the companion doc: hardening #1/#2/#4 (mechanical) and G1/G3
(input/observability) are independent and can land in parallel; the eval work
(G2) and the lifecycle FSM (hardening #5) are the larger items to schedule after.

## Open questions

- For G1, is the kwarg schema generated by introspecting `train.py` argparse at
  startup, or maintained as a declared schema file the role and validator both
  read? (Introspection removes drift but couples the engine gate to import-time
  execution of project code.)
- For G2, what CI width / seed count is the right promotion gate for ASR WER, and
  does it apply per-dataset or only to the macro metric?
- For G3, do lifecycle events live in the existing `Journal` files or in the
  `idea_lake`/`orze_state` DB proposed by hardening doc #5? (They should share
  one store to avoid a fourth source of truth.)

## References

- Companion: [`determinism_hardening.md`](determinism_hardening.md) (mechanical
  glue: F1–F3, hardening wins #1–#5)
- v4.4.4 normalizer: `engine/launcher.py:normalize_nested_config`,
  `engine/phases.py` (applied to `flat_cfg` pre-write), `CHANGELOG.md` (4.4.4)
- Nested-config guard (F5 validator): `engine/launcher.py:validate_idea_config_no_nested`
- Structured journal: `journal/__init__.py`
- FSM engine: `fsm/engine.py`, `fsm/runner.py`
