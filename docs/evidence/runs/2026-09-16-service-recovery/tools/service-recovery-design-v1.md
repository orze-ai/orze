# Service recovery work, 2026-09-16

Continue the original service/state compatibility requirement, not a research
gain claim. Current closed services cannot start a successor after their host
exits. Original controller/host registrations and all budget history must stay.

Implement explicit recovery from a verified closed CPU service into a new service
configuration/host claim. The execution scope, project config, DB, package/runtime
identity and budgets stay fixed. Old host and controller must be observed absent
by their exact local host/boot/process identities; never signal or adopt a PID.
History validation remains read-only; a new explicit recovery instruction plus
closure and absence are distinct from the old informational closure record.

Re-use existing RESERVED/SPAWNING/ISSUED/PREPARED/STARTED one-shot grant and actual
new parent's child ownership. New host retains actual successor Popen/pidfd.
Preserve every old registration; no lock deletion, budget reset or auto retry.
An unclosed/unknown/crashed service without complete closure remains HOLD.
Actual systemd, reboot, deployment, legacy-PID migration and backup/rollback need
their own evidence; this cannot close all production requirements.

Implement prepare CLI with pinned old service/snapshot/controller/request; host
startup consumes it. Explicit optional manager installation uses the new target's
existing scoped installer after source inactivity validation. Process fixtures
exercise same host/CLI and real CPU work, never any existing service or manager.

Verify restart after host exit, multi-generation chain, preserved cumulative
budget and artifacts, exactly one successor with racing preparations, changed
source/config, missing ACK, active source, latches and lost startup boundaries.
Existing live handoff tests remain unchanged in assertions. Preserve old failure,
each source snapshot/run, two-repo relevant regression and independent evidence.
