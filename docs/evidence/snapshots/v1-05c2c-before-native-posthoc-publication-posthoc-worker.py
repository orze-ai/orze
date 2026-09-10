"""Fixed private-input driver: candidate files only, never lifecycle/catalog.

The inherited payload descriptor is verified and closed before loading the
adapter runner. Generation is a non-secret command-bound integer; the payload
itself binds task/attempt/configuration and never appears in argv or environment.
This is execution-result metadata, not an observation or scientific verdict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main(argv=None):
    from orze.engine.sealed_payload import read_sealed_payload
    from orze.core.execution_attempts import AttemptRef, _json
    from dataclasses import asdict
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise ValueError("posthoc_driver_arguments_invalid")
    fd, digest, generation = int(args[0]), args[1], int(args[2])
    if fd < 3 or generation < 1:
        raise ValueError("posthoc_driver_identity_invalid")
    try:
        raw = read_sealed_payload(fd, digest)
    finally:
        os.close(fd)
    packet = json.loads(raw)
    if (type(packet) is not dict or set(packet) != {
            "schema", "task_id", "attempt_id", "kind", "work_dir", "configuration"}
            or type(packet["schema"]) is not int or packet["schema"] != 1
            or _json(packet).encode("utf-8") != raw):
        raise ValueError("posthoc_driver_payload_invalid")
    ref = AttemptRef(packet["task_id"], "posthoc", packet["attempt_id"], generation)
    work = Path(packet["work_dir"])
    if (not work.is_absolute() or work.name != "work" or work.parent.name != ref.attempt_id
            or work.parent.parent.name != "_posthoc_attempts"
            or work.parent.parent.parent.name != ref.task_id):
        raise ValueError("posthoc_driver_work_invalid")
    from orze.engine.artifact_publication import _open_directory
    descriptor = _open_directory(work)
    os.close(descriptor)
    # Public dict-returning adapter API is preserved, but the output directory
    # is this attempt's private work and the legacy registrar is never invoked.
    from orze.engine.posthoc_runner import run_posthoc
    metrics = run_posthoc(ref.task_id, packet["configuration"], work,
                          artifact_catalog_db=None)
    outcome = "failed" if metrics.get("status") == "FAILED" else "completed"
    receipt = {"schema": 1, "attempt_ref": asdict(ref),
               "payload_sha256": digest, "outcome": outcome}
    from orze.core.fs import atomic_write
    atomic_write(work / "_posthoc_result.json", _json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
