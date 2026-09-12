# Orze packaged operations guide

This guide ships inside the Orze wheel. It describes the supported candidate
runtime, not a claim that any particular installation has been deployed or
that research quality or throughput has been proven.

## Identify the runtime before operating it

Use the intended environment's absolute executable path, for example
`/path/to/venv/bin/orze --version` and `/path/to/venv/bin/orze --help`.
A checkout's version does not identify the package used by an existing service.
Record the wheel digest, Python/dependency versions, project configuration,
service working directory and exact runtime before a production change.

Orze Core and Pro are separate packages. Installing Pro does not establish
that its production license or model accounts are valid. Do not bypass the
license gate for deployment checks, and never publish Pro to a public index.

## CPU-only application execution

A project may declare `execution: {version: 1, resource: cpu, slots: 1,
wall_budget_seconds: N}` with a positive finite budget. It uses the existing
CLI and native attempt, artifact, observation and budget protocols; it does
not require a fabricated GPU slot.

For research with no preset overall deadline, explicitly choose
`execution: {version: 2, resource: cpu, slots: 1, wall_budget_seconds: null}`.
This removes only the cumulative wall ceiling: each action still needs a
finite timeout and runtime lease, all reservations remain charged, and slots,
Stop and HOLD still apply. Existing v1 projects are unchanged. A prior resource
scope cannot be switched to another declaration to reset its authorization.
This CPU declaration grants no GPU or paid-provider access.

The registered Policy decides what to explore, analyze, replicate or conclude.
It must handle `remaining_wall_seconds: null` in continuous mode. Queue drain
is not scientific convergence, and a truncated/unavailable evidence view is
not proof that useful research is exhausted.

Custom domains and policies must be registered by the application's entry
module before calling the real Orze CLI. A declaration alone does not install
an adapter. Run that trusted entry module with the intended environment and
an explicit `-c /absolute/project/orze.yaml`. Start with a bounded private
canary, inspect its actual results and ledger, and keep provider/GPU permissions
out of a CPU-only project.

The native CPU runtime lease is bound to the host and boot. Its default limit
is the action timeout and begins at captured INTENT; preparation and publication
consume the same authority window. An explicitly shorter
`cpu_runtime_lease: {version: 1, ttl_seconds: N}` cannot extend an action's
timeout. A worker exiting zero after expiry is not a successful research result.

## Stopping, restart and uncertain state

Use only the control operations supported by the installed runtime and the
specific controller profile. Preserve explicit Stop/HOLD markers, attempt
identities, pending effects, guards and reserved budget. A dead PID, empty
process list, elapsed time or inactive service is not proof that all owned
writers closed. Do not manually remove locks to force a retry.

A registered Policy may return `{kind: Pause, reason: ..., wakeup: null}` only
when no action or reservation is active. The decision is recorded durably and
ends this foreground invocation without setting the permanent Stop latch.
The next invocation re-evaluates current inputs using the same scope and ledger.
Pause does not refund charges, clear an existing Stop/HOLD, claim queued work,
or prohibit another authorized controller from acting later. `Wait` instead
waits within the running invocation; `Stop` persists final scope termination.

Recovery of a fully confirmed CPU terminal effect may settle its original
reservation without executing the worker again. This is not general process
adoption, checkpoint resume, automatic repair or exactly-once external effects.
Unconfirmed publication or ownership remains HOLD.

The current source candidate's generic `orze upgrade` does not perform a
verified package replacement and restart; a stop/refusal exit is not deployment
success. Legacy watchdog/service installation is not an isolated per-project
canary: its global service registration and process cleanup must be reviewed
before enabling it for a new target.

## Release and rollback

Build and inspect real wheels: editable imports can hide missing package data.
The packaged `skills/`, rule files and admin UI assets must match the intended
source. Validate dependencies and a real bounded foreground canary before
switching a service.

A service switch requires an explicit target, backup, old-owner closure,
compatible state handling and a separately verified restart/rollback procedure.
Never assume that an older executable can consume state already written by a
newer runtime. A ready wheel or a passing canary is not a completed production
rollout.

Other bundled skill documents provide historical operational detail; do not
use an older upgrade/watchdog recipe to override these ownership and deployment
boundaries. Consult the checked-out version's contracts for the exact supported
workflow.
