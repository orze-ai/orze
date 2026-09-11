# CPU invocation and confirmed-terminal lifetime

V1-06E removes unconditional process-local roots from the generic CPU path.
It does not change the public Domain, Policy, Propose, Replicate, action or
observation contracts. Collection is an observation, never closure authority.

## Ownership and retirement

An exact InterfaceContext or DomainRun owns its complete captured state.
The identity lookup registries hold weak values, including when a trusted
callback points back to its handle. An external policy, run or live native
owner still strongly retains the context and source bytes it needs. Discarding
one consumer does not invalidate another consumer or explicitly erase shared
PreparedSources. Unused preparation that never entered native execution can
therefore be collected without inventing a release permission.

Native active and uncertain owners remain strongly indexed. A terminal owner
is retired only after the existing supervisor closure proof, effect transaction
and budget settlement have actually succeeded. Completed, failed and confirmed
interrupted outcomes can all qualify; a main-process exit by itself cannot.
Unknown publication, uncertain settlement and HOLD do not qualify, and neither
GC nor elapsed time releases their reservations or authorizes retry/adoption.

The same externally retained terminal handle can still harvest or stop and
read its detached historical result. The cache uses weak keys, detached JSON
and an independent private weak process-identity witness. It neither strongly
retains the key/process nor trusts a replaceable witness on the public handle.
Copied/lookalike handles, changed full references, process objects, scope or
permits remain rejected. This readback is not a new GO grant and does not reopen
or revalidate a terminal Lake merely because an old caller reads it again.

## Invocation hooks

Successful CPU close unregisters its unique exit-callback token, releases its
Domain/Policy references and restores only signal handlers that are still its
own. It must not overwrite a handler installed later by another caller. Nested
Orze invocations must not reinstall an already retired invocation when closed
out of order. Cleanup failures retain the invocation captures and existing
one-shot close/STOP semantics; the close-attempt latch is not success proof.

## Evidence limits

Tests use actual private CPU supervisors, SQLite, artifact publication and the
ordinary CLI loop. The two-action product test proposes an experiment, consumes
its newly published source ID in an analysis and then stops. Callback-registry
unit controls explicitly replace only the interpreter hook registry; real CLI
collection uses the actual registry. No execution-owner registry is cleared to
manufacture collection.

Small-payload weak-reference tests prove these ownership paths can be released;
they are not an RSS benchmark, immediate deallocation promise, global memory cap
or unbounded-lifetime guarantee. Deliberate external references, trusted plugin
registrations, caller-held terminal handles and unresolved owners can still
retain data. Controller membership/ACK proofs and other resource adapters are
not retired by this change. No live campaign, GPU, provider, cross-restart
recovery or research-efficiency percentage is claimed.
