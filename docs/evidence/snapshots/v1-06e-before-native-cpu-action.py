"""Native CPU actions in the existing scheduler, attempt and effect stores.

No GPU allocation, training phase, implicit observation, process adoption or
failure retry. Inputs are sealed bytes, not a sandbox for the chosen command.
Unknown preparation/publication retains the strong owner and its reservation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import time

import yaml

from orze.core.cpu_action_contract import validate_action, action_fingerprint, artifact_binding
from orze.core.execution_attempts import (
    AttemptRef, create_attempt, finish_attempt, mark_running, require_current,
)
from orze.core.research_artifacts import register_artifacts
from orze.engine import process_supervision as proof
from orze.engine.artifact_publication import prepare_artifacts, verify_prepared_artifacts
from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
from orze.engine.claim_authority import _closed
from orze.engine.execution_authority import (
    canonical_identity_equal as same, execution_transaction, lifecycle_fence,
)
from orze.engine.execution_catalog import bind_catalog, declared_catalog
from orze.engine.sealed_payload import sealed_payload
from orze.engine.supervised_process import prepare_supervised, SupervisedProcess, SupervisionUncertain
from orze.engine.supervisor_worker import canonical as protocol_bytes
from orze.engine.termination_hold import terminate_execution, require_no_unconfirmed_stop
from orze.engine.training_attempts import _read  # structural bounded JSON reader only


class CPUActionHOLD(AttemptEffectInDoubt):
    """No terminal result, budget release or ordinary retry is authorized."""


@dataclass(eq=False)
class CPUActionHandle:
    idea_id: str
    attempt_id: str
    attempt_ref: AttemptRef
    process: object = None


@dataclass
class _Owner:
    handle: CPUActionHandle
    folder: Path
    action: dict
    permit: dict
    admission: object
    binding: dict
    process: object = None
    deadline: float | None = None
    stop_attempted: bool = False
    held: bool = False
    terminal: dict | None = None
    work_identity: tuple | None = None
    started_at: float | None = None
    domain_run: object = None


_OWNERS = {}  # Strong unresolved and terminal owners; no global memory-cap claim.


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _permit_matches(lake, folder, action, permit):
    paths = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
    scope = permit.get("budget_scope") if type(permit) is dict else None
    if (len(paths) != 1 or not paths[0] or type(scope) is not dict
            or not same({"task_id": permit.get("task_id"), "results_dir": scope.get("results_dir"),
                         "database": scope.get("database"), "timeout": permit.get("wall_limit_seconds")},
                        {"task_id": folder.name, "results_dir": str(folder.parent),
                         "database": str(Path(paths[0]).absolute()), "timeout": action["timeout_seconds"]})):
        raise CPUActionHOLD("cpu_action_permit_execution_mismatch")


def _scope(lake, folder, action, domain_run=None, *, cfg):
    if lake is None:
        raise CPUActionHOLD("cpu_action_catalog_required")
    paths = [row[2] for row in lake.conn.execute("PRAGMA database_list") if row[1] == "main"]
    if len(paths) != 1 or not paths[0]:
        raise CPUActionHOLD("cpu_action_persistent_catalog_required")
    database = str(Path(paths[0]).absolute())
    declared = declared_catalog(folder)
    if declared is not None and declared != database:
        raise CPUActionHOLD("cpu_action_catalog_changed")
    claim, claim_sha = _read(folder / "claim.json", 8192)
    if (claim.get("resource") != "cpu" or "gpu" not in claim or claim["gpu"] is not None
            or claim.get("lifecycle_db") != database):
        raise CPUActionHOLD("cpu_action_claim_resource_changed")
    AttemptRef(folder.name, "action", claim.get("attempt_id"), 1)
    row = lake.conn.execute(
        "SELECT kind,config FROM main.ideas WHERE idea_id COLLATE BINARY=?", (folder.name,),
    ).fetchone()
    if row is None or row[0] != "native_cpu_action" or type(row[1]) is not str or len(row[1].encode()) > 65536:
        raise CPUActionHOLD("cpu_action_task_kind_changed")
    configured = yaml.safe_load(row[1])
    if (type(configured) is not dict or configured.get("kind", "native_cpu_action") != "native_cpu_action"):
        raise CPUActionHOLD("cpu_action_task_config_changed")
    if domain_run is None:
        if not same(validate_action(configured.get("action")), action):
            raise CPUActionHOLD("cpu_action_task_config_changed")
    else:
        from orze.core.research_interfaces import require_domain_run, domain_sources
        from orze.engine.cpu_action_sources import require_sources
        require_domain_run(domain_run, raw_config_sha256=hashlib.sha256(row[1].encode()).hexdigest(),
                           action=action)
        require_sources(lake, folder.parent, domain_sources(domain_run), metadata_only=True)
    source = {"claim_attempt_id": claim["attempt_id"], "claim_sha256": claim_sha,
              "config_sha256": hashlib.sha256(row[1].encode()).hexdigest(), "database": database}
    from orze.engine.cpu_replication import authorization
    replication = authorization(lake, folder.name, folder.parent, cfg,
                                action=action, domain_run=domain_run)
    if replication is not None:
        source["replication_request"] = replication
    return source


def _watch_replication(tx, source):
    if "replication_request" in source:
        tx.watch_cpu_replication(source["replication_request"])


def _work_identity(path):
    values = []
    for directory in (*reversed(path.parents), path):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise CPUActionHOLD("cpu_action_work_redirected")
        values.append((str(directory), info.st_dev, info.st_ino, info.st_mode))
    return tuple(values)


def _create_work(work):
    from orze.engine.artifact_publication import _open_directory
    fd = _open_directory(work.parent.parent, create=True)
    try:
        os.mkdir(work.parent.name, 0o700, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = _open_directory(work.parent)
    try:
        os.mkdir("work", 0o700, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    return _work_identity(work)


def _owner(handle, results_dir, permit):
    owner = _OWNERS.get(id(handle))
    if (owner is None or owner.handle is not handle or owner.held
            or type(handle) is not CPUActionHandle
            or handle.idea_id != owner.folder.name or handle.attempt_id != owner.binding["attempt_id"]
            or asdict(handle.attempt_ref) != owner.binding["attempt_ref"]
            or handle.process is not owner.process
            or Path(results_dir).absolute() != owner.folder.parent
            or not same(permit, owner.permit)):
        raise CPUActionHOLD("cpu_action_owner_unavailable")
    return owner


def _owned(owner, lake, cfg, *, states=("RUNNING",)):
    handle, folder = owner.handle, owner.folder
    row = require_current(lake.conn, handle.attempt_ref, states=states)
    if not same(row["binding"], owner.binding):
        raise CPUActionHOLD("cpu_action_binding_changed")
    if not same(_scope(lake, folder, owner.action, owner.domain_run, cfg=cfg), owner.binding["source"]):
        raise CPUActionHOLD("cpu_action_source_changed")
    if owner.domain_run is not None:
        from orze.core.research_interfaces import domain_run_metadata
        from orze.engine.cpu_domain_publication import publication_binding
        if (not same(domain_run_metadata(owner.domain_run), owner.binding["domain_run"])
                or not same({"publication": publication_binding(owner.domain_run, folder.parent)},
                            {"publication": owner.binding.get("observation_publication")})):
            raise CPUActionHOLD("cpu_action_domain_binding_changed")
    if not same(artifact_binding(cfg, folder, owner.action), owner.binding["artifact_publication"]):
        raise CPUActionHOLD("cpu_action_artifact_binding_changed")
    if not same(lifecycle_fence(lake, handle.idea_id, "action"), owner.binding["lifecycle"]):
        raise CPUActionHOLD("cpu_action_lifecycle_changed")
    if owner.work_identity is not None and _work_identity(Path(owner.binding["work_dir"])) != owner.work_identity:
        raise CPUActionHOLD("cpu_action_work_changed")
    if row["state"] != "LAUNCHING":
        ready = proof.bound_binding(handle, row, folder, phase="action")
        if type(row["binding"].get("process_pid")) is not int or row["binding"]["process_pid"] != ready["worker"]["pid"]:
            raise CPUActionHOLD("cpu_action_process_identity_changed")
    return row


def _admit(owner, lake):
    from orze.core.cpu_action_budget import require_permit
    if not callable(owner.admission):
        raise CPUActionHOLD("cpu_action_admission_required")
    owner.admission()
    _permit_matches(lake, owner.folder, owner.action, owner.permit)
    require_permit(lake, owner.permit, owner.handle.attempt_ref)
    require_no_unconfirmed_stop(owner.folder)
    if owner.domain_run is not None:
        from orze.core.research_interfaces import domain_sources
        from orze.engine.cpu_action_sources import require_sources
        require_sources(lake, owner.folder.parent, domain_sources(owner.domain_run))


def _terminate(owner):
    from orze.engine.process import _terminate_and_reap
    if owner.stop_attempted:
        raise CPUActionHOLD("cpu_action_stop_already_attempted")
    owner.stop_attempted = True
    return terminate_execution(owner.handle, owner.folder, phase="action",
                               reaper=_terminate_and_reap, timeout=1)


def _hold(owner, exc):
    if owner is not None:
        owner.held = True
    error = CPUActionHOLD("cpu_action_unconfirmed")
    error.cpu_action_handle = owner.handle if owner is not None else None
    raise error from exc


def launch(idea_id, results_dir, cfg, *, lake, action, permit, admission, domain_run=None):
    """Return an actual READY-bound action owner after GO (or cancellation)."""
    from orze.core.cpu_action_budget import bind, require_permit
    owner = None
    try:
        AttemptRef(idea_id, "action", "validation", 1)
        action = validate_action(action)
        permit = copy.deepcopy(permit)
        folder = Path(results_dir).absolute() / idea_id
        _permit_matches(lake, folder, action, permit)
        if not callable(admission):
            raise CPUActionHOLD("cpu_action_admission_required")
        admission()
        require_permit(lake, permit)
        source = _scope(lake, folder, action, domain_run, cfg=cfg)
        inputs = _canonical(action["inputs"])
        attempt_id = secrets.token_hex(16)
        work = folder / "_action_attempts" / attempt_id / "work"
        binding = {"origin": "native_cpu_action", "kind": "native_cpu_action", "resource": "cpu",
            "attempt_id": attempt_id, "source": source, "scope": str(folder), "work_dir": str(work),
            "action_sha256": action_fingerprint(action), "inputs_sha256": hashlib.sha256(inputs).hexdigest(),
            "command_sha256": hashlib.sha256(protocol_bytes(action["command"])).hexdigest(),
            "timeout_seconds": action["timeout_seconds"], "reservation_id": permit["reservation_id"],
            "process_supervision_protocol": proof.PROTOCOL,
            "artifact_publication": artifact_binding(cfg, folder, action), "lifecycle_phase": "action"}
        if domain_run is not None:
            from orze.core.research_interfaces import domain_run_metadata
            from orze.engine.cpu_domain_publication import publication_binding
            binding["domain_run"] = domain_run_metadata(domain_run)
            observation = publication_binding(domain_run, folder.parent)
            if observation is not None:
                binding["observation_publication"] = observation
        with execution_transaction(lake, folder) as tx:
            if not same(_scope(lake, folder, action, domain_run, cfg=cfg), source):
                raise CPUActionHOLD("cpu_action_source_changed")
            _closed(tx.conn, idea_id)
            fence = lifecycle_fence(lake, idea_id, "action")
            if fence["global_state"] != "CLAIMED" or fence["phase_state"] != "PENDING":
                raise CPUActionHOLD("cpu_action_claimed_lifecycle_required")
            binding["lifecycle"] = fence
            bind_catalog(lake, folder, tx.lease)
            ref = create_attempt(tx.conn, idea_id, "action", attempt_id, binding)
            # Ref is generated by the store, not predicted before admission.
            binding["attempt_ref"] = asdict(ref)
            from orze.core.execution_attempts import _update
            _update(tx.conn, ref, require_current(tx.conn, ref), state="LAUNCHING", binding=binding)
            tx.watch_attempt(ref)
            _watch_replication(tx, source)
        handle = CPUActionHandle(idea_id, attempt_id, ref)
        owner = _Owner(handle, folder, copy.deepcopy(action), permit, admission, copy.deepcopy(binding),
                       domain_run=domain_run)
        _OWNERS[id(handle)] = owner
        bind(lake, permit, ref)
        owner.work_identity = _create_work(work)
        _admit(owner, lake)
        _owned(owner, lake, cfg, states=("LAUNCHING",))
        environment = dict(os.environ)
        for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            environment[key] = ""
        with ExitStack() as streams:
            fd = streams.enter_context(sealed_payload(inputs))
            # This public input descriptor supports ordinary os.read, unlike
            # internal sealed readers which deliberately use offset-free pread.
            if os.lseek(fd, 0, os.SEEK_SET) != 0:
                raise CPUActionHOLD("cpu_action_input_offset_unconfirmed")
            environment["ORZE_ACTION_INPUT_FD"] = str(fd)
            environment["ORZE_ACTION_INPUT_SHA256"] = binding["inputs_sha256"]
            source_fds = ()
            if domain_run is not None:
                from orze.core.research_interfaces import domain_sources
                from orze.engine.cpu_action_sources import sealed_sources
                source_environment, source_fds = streams.enter_context(sealed_sources(domain_sources(domain_run)))
                environment.update(source_environment)
            process = prepare_supervised(list(action["command"]),
                identity=proof.identity(handle, folder, phase="action"), env=environment, cwd=str(work),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, worker_only_fds=(fd, *source_fds))
            owner.process = handle.process = process
            ready = proof.ready_binding(handle, folder, phase="action")
            if ready["command_sha256"] != binding["command_sha256"]:
                raise CPUActionHOLD("cpu_action_command_changed")
            with execution_transaction(lake, folder) as tx:
                _owned(owner, lake, cfg, states=("LAUNCHING",))
                if not lake._record_state_transition_in_tx(idea_id, "CLAIMED", "IN_PROGRESS",
                        "cpu_action_ready", pid=process.pid, sop_type="action"):
                    raise CPUActionHOLD("cpu_action_started_lifecycle_rejected")
                running = {**binding, "supervision": ready, "process_pid": process.pid,
                           "lifecycle": lifecycle_fence(lake, idea_id, "action")}
                mark_running(tx.conn, ref, running)
                tx.watch_attempt(ref)
                _watch_replication(tx, binding["source"])
            owner.binding = copy.deepcopy(running)
            _admit(owner, lake)
            _owned(owner, lake, cfg)
            owner.started_at = time.monotonic()
            owner.deadline = owner.started_at + float(action["timeout_seconds"])
            process.start()
        return handle
    except BaseException as exc:
        if owner is not None and isinstance(exc, SupervisionUncertain) and exc.process is not None:
            owner.process = owner.handle.process = exc.process
            owner.handle._termination_unconfirmed = True
        if owner is not None and owner.process is not None and not owner.stop_attempted:
            try:
                if isinstance(exc, SupervisionUncertain):
                    owner.handle._termination_unconfirmed = True
                elif owner.process.poll() is None:
                    _terminate(owner)
            except BaseException:
                pass  # The retained owner is HOLD, never a clean failure.
        _hold(owner, exc)


def harvest(handle, results_dir, cfg, *, lake, permit):
    """Nonblocking until closure, then publish and settle the captured action."""
    owner = None
    try:
        owner = _owner(handle, results_dir, permit)
        if owner.terminal is not None:
            return copy.deepcopy(owner.terminal)
        row = _owned(owner, lake, cfg)
        ret = owner.process.poll()
        if ret is None and owner.deadline is not None and time.monotonic() >= owner.deadline:
            ret = _terminate(owner)
        if ret is None:
            return None
        closure = proof.require_closed(handle, row, owner.folder, ret, phase="action")
        elapsed_wall_seconds = max(0.0, time.monotonic() - owner.started_at)
        interrupted = closure["stop_requested"] or closure["forced_cleanup"]
        outcome = "interrupted" if interrupted else "completed" if ret == 0 else "failed"
        prepared = (prepare_artifacts(handle.attempt_ref, owner.folder,
                    owner.binding["artifact_publication"], source_dir=Path(owner.binding["work_dir"]))
                    if outcome == "completed" else None)
        domain_publication = None
        if owner.domain_run is not None and prepared is not None:
            from orze.engine.cpu_domain_publication import prepare as prepare_domain_publication
            domain_publication = prepare_domain_publication(owner.domain_run, handle.attempt_ref, prepared)
        with execution_transaction(lake, owner.folder) as tx:
            row = _owned(owner, lake, cfg)
            if not same(closure, proof.require_closed(handle, row, owner.folder, ret, phase="action")):
                raise CPUActionHOLD("cpu_action_tree_changed")
            records = [] if prepared is None else list(verify_prepared_artifacts(prepared,
                handle.attempt_ref, owner.folder, owner.binding["artifact_publication"],
                source_dir=Path(owner.binding["work_dir"])))
            observations = []
            if domain_publication is not None:
                from orze.engine.cpu_domain_publication import verify as verify_domain_publication
                observations = verify_domain_publication(domain_publication, handle.attempt_ref)
            terminal = {"outcome": outcome, "reason_code": "cpu_action_" + outcome,
                "return_code": ret, "process_tree": closure, "artifact_ids": [r["artifact_id"] for r in records],
                "observation_ids": [r["observation_id"] for r in observations],
                "lifecycle_phase": "action", "elapsed_wall_seconds": elapsed_wall_seconds}
            digest = tx.prepare(handle.attempt_ref, {"operation": "cpu_action_terminal", **terminal})
            if prepared is not None:
                verify_prepared_artifacts(prepared, handle.attempt_ref, owner.folder,
                    owner.binding["artifact_publication"], source_dir=Path(owner.binding["work_dir"]))
                register_artifacts(tx.conn, handle.attempt_ref, records)
            if owner.domain_run is not None and "observation_publication" in owner.binding:
                from orze.core.research_observations import register_observations
                register_observations(tx.conn, handle.attempt_ref, observations)
            if not lake._record_state_transition_in_tx(handle.idea_id, "IN_PROGRESS",
                    "COMPLETE" if outcome == "completed" else "FAILED", terminal["reason_code"],
                    pid=owner.process.pid, sop_type="action"):
                raise CPUActionHOLD("cpu_action_terminal_lifecycle_rejected")
            terminal.update(lifecycle=lifecycle_fence(lake, handle.idea_id, "action"),
                            effect_receipt_sha256=digest)
            # Binding's captured launch fence is historical after this exact edge.
            updated = {**owner.binding, "lifecycle": terminal["lifecycle"]}
            from orze.core.execution_attempts import _update
            _update(tx.conn, handle.attempt_ref, require_current(tx.conn, handle.attempt_ref),
                    state="RUNNING", binding=updated)
            if finish_attempt(tx.conn, handle.attempt_ref, terminal) != "committed":
                raise CPUActionHOLD("cpu_action_terminal_not_new")
            if prepared is not None:
                verify_prepared_artifacts(prepared, handle.attempt_ref, owner.folder,
                    owner.binding["artifact_publication"], source_dir=Path(owner.binding["work_dir"]))
            tx.watch_artifacts(handle.attempt_ref, records)
            if owner.domain_run is not None:
                from orze.core.research_interfaces import domain_sources
                if domain_publication is not None:
                    verify_domain_publication(domain_publication, handle.attempt_ref)
                tx.watch_observations(handle.attempt_ref, observations)
                tx.watch_cpu_sources(domain_sources(owner.domain_run))
            tx.watch_attempt(handle.attempt_ref)
            _watch_replication(tx, owner.binding["source"])
        owner.binding = copy.deepcopy(updated)
        from orze.core.cpu_action_budget import settle
        settle(lake, owner.permit, handle.attempt_ref, terminal)
        owner.terminal = copy.deepcopy(terminal)
        return terminal
    except BaseException as exc:
        _hold(owner, exc)


def stop(handle, results_dir, cfg, *, lake, permit):
    """Explicit shutdown uses the same once-STOP and terminal publisher."""
    owner = None
    try:
        owner = _owner(handle, results_dir, permit)
        if owner.terminal is not None:
            return copy.deepcopy(owner.terminal)
        if owner.process.poll() is None:
            _terminate(owner)
        return harvest(handle, results_dir, cfg, lake=lake, permit=permit)
    except BaseException as exc:
        _hold(owner, exc)
