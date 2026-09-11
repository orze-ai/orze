# Policy-generated CPU proposals

This extends the existing opt-in CPU Domain/Policy loop. A registered trusted
Policy can construct a new experiment, check or analysis request; it does not
need an inbox-writing callback, a training-shaped task or a second runner.

## Decision

A selected Domain is required. The exact five-field decision is:

```json
{
  "kind": "Propose",
  "request_id": "check-plan-1",
  "task_id": "idea-check-1",
  "reason": "test the explicit candidate question",
  "domain_request": {
    "version": 1,
    "purpose": "check candidate",
    "inputs": {},
    "timeout_seconds": 2,
    "outputs": {},
    "input_artifact_ids": [],
    "payload": {}
  }
}
```

Domain request fields retain the existing research-interface contract. Payload
is interpreted by the selected Domain; the empty example is not a runnable
command-domain request. Stable IDs are bounded tokens; reason is nonblank and
at most 1024 UTF-8 bytes. Input IDs must uniquely select completed occurrences
in the snapshot captured before Policy ran. Mutating a callback's detached
snapshot cannot add inputs. Empty sources allow proposing the first experiment.

## Admission and the next decision

The existing IdeaLake is the only database. Request identity, source metadata
and the normal admission outcome are composed in one owned transaction. The
ordinary exact-ID and configuration-dedup rules apply. There is no replica
exemption, configuration salt, replacement of existing tasks or inbox ACK.
Sources are captured outside the writer and checked again at its boundaries.
A queued proposal is not an observation, a claim or a successful experiment.

A successful proposal operation yields a durable Wait. A later Execute decision
must still pass source capture, Domain preparation, shared budget, claim and
native supervision. With `--once`, proposal admission ends that invocation
without executing the new task. A subsequent invocation can select that task.

Policy receives `snapshot.recorded_proposals`: a bounded `results` window and
`more_available`. Each result has `request_id`, `task_id`, `status`, `reason`
and `existing_id`. An identical key replays its recorded outcome, not a new
admission; a changed request under that key is a conflict. Normal duplicate
configuration and same-task conflicts are recorded denials, not successful
admissions. Storage or identity uncertainty remains HOLD. Post-commit checks
must allow a peer to have legitimately claimed an admitted task.
Replay still requires the captured inputs to be current and the original
admission evidence to remain consistent; the historical view alone cannot
satisfy those checks. New inserts verify their complete staged lifecycle before
commit and its current consistency afterward. Legacy exact-ID matches do not
acquire invented creation or transition history.

## Limits and authority

Domain identity in the proposal record audits request origin and exact replay.
It is not a new private execution grant: ordinary queued Domain requests still
use the executing invocation's selected Domain and its native run binding.
The history view is metadata, not execution authority or a global snapshot.
Its bounded window is not complete history or a pagination interface.

Trusted Python Policy/Domain callbacks are not an OS sandbox and are not timed
by the supervised worker's wall envelope. Source capture retains the existing
32-input, 16 MiB content and 32 KiB binding limits; metadata records have bounded
JSON encodings. These limits are not a global queue/history or control-loop
budget. No automatic recovery, statistical independence, comparative scientific
gain or completion of the whole V1 follows from successful proposal admission.
