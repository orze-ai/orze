"""Two-phase, owned role execution; not controller-wide stop authority.

The existing role lock contains a durable v2 INTENT before prepare. Unknown
v2 receipts are never adopted by the legacy nonce scanner. Only this process's
strong owner may bind READY, send GO, accept TREE_CLOSED, and release its lock.
No command, environment value, trigger payload, or raw nonce is persisted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import threading

from orze.engine.attempt_effect_receipts import _encoded, _publish, _read, _sync
from orze.engine.gc_tree import identity, plain_directory, snapshot, rename_no_replace, reclaim
from orze.engine.supervised_process import (
    SupervisedProcess, SupervisionUnavailable, SupervisionUncertain, prepare_supervised,
)
from orze.engine.supervisor_worker import canonical


_KEYS = {"role_name", "attempt_id", "scope", "lock_dir", "nonce_sha256",
         "command_sha256", "trigger_delivery", "trigger_delivery_db"}
_REF = {"delivery_id", "scope", "role_name", "generation", "attempt_id",
        "nonce_sha256", "command_sha256"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_GUARD = threading.RLock()
_OWNERS = {}  # Strong until confirmed release; no age-based forgetting.
_BOUND = {}


class RoleSupervisionHOLD(RuntimeError):
    def __init__(self, reason, *, owner=None):
        super().__init__(reason)
        self.owner = owner


def _same(left, right):
    return canonical(left) == canonical(right)


def _path(value):
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("role_path_invalid")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("role_path_invalid")
    return path


def _reference(meta):
    ref, db = meta["trigger_delivery"], meta["trigger_delivery_db"]
    if ref is None:
        if db is not None:
            raise ValueError("role_trigger_database_without_reference")
        return
    if type(ref) is not dict or set(ref) != _REF:
        raise ValueError("role_trigger_reference_invalid")
    if type(ref["generation"]) is not int or ref["generation"] <= 0:
        raise ValueError("role_trigger_generation_invalid")
    for key in _REF - {"generation"}:
        if type(ref[key]) is not str or not ref[key] or len(ref[key].encode("utf-8")) > 4096:
            raise ValueError("role_trigger_reference_invalid")
    for key in ("role_name", "attempt_id", "nonce_sha256", "command_sha256"):
        if ref[key] != meta[key]:
            raise ValueError("role_trigger_reference_mismatch")
    _path(db)


def _trigger_snapshot(meta, token=None):
    if meta["trigger_delivery"] is None:
        return None, None
    from orze.engine.trigger_delivery import _get, _matches
    from orze.engine.trigger_delivery_storage import connect, schema_status
    conn = connect(meta["trigger_delivery_db"])
    try:
        conn.execute("BEGIN")
        if not schema_status(conn):
            raise ValueError("role_trigger_schema_invalid")
        ref = meta["trigger_delivery"]
        row = _get(conn, ref["delivery_id"])
        if row is None:
            raise ValueError("role_trigger_missing")
        if token is None:
            token = {**ref, "owner": row["owner"], "payload_sha256": row["payload_sha256"]}
        if not _matches(row, token, launch=True, conn=conn):
            raise ValueError("role_trigger_binding_changed")
        latest = conn.execute(
            "SELECT from_state,to_state,outcome,exit_code,cleanup_verified,attempt_id "
            "FROM trigger_delivery_transitions WHERE delivery_id=? ORDER BY id DESC LIMIT 1",
            (ref["delivery_id"],),
        ).fetchone()
        return dict(token), (row["state"], tuple(latest) if latest is not None else None,
                             row["process_pid"])
    finally:
        conn.close()


class RoleLaunch:
    is_pending_role = True

    def __init__(self, meta, lock_identity, lock_bytes, trigger_token):
        self._meta = copy.deepcopy(meta)
        self._lock_identity = lock_identity
        self._lock_bytes = lock_bytes
        self._trigger_token = trigger_token
        self._process = None
        self._ready = None
        self._closure = None
        self._raw = None
        self._prepare_entered = False
        self._go_attempted = False
        self._stop_attempted = False
        self._protocol_uncertain = False
        self._held_reason = None
        self._released = False
        self._bound_role = None
        self._holder_snapshot = None
        self.log_fh = None
        self._receipt = {
            "schema_version": 2, "kind": "role_supervision", "stage": "INTENT",
            "role_name": meta["role_name"], "attempt_id": meta["attempt_id"],
            "scope": meta["scope"], "lock_dir": meta["lock_dir"],
            "nonce_sha256": meta["nonce_sha256"], "command_sha256": meta["command_sha256"],
            "trigger_delivery": meta["trigger_delivery"],
            "trigger_binding": trigger_token,
            "owner_id": secrets.token_hex(32), "host": socket.gethostname(),
            "controller_pid": os.getpid(), "lock_identity": list(lock_identity),
            "supervision": None, "process_tree": None,
        }

    @property
    def role_name(self):
        return self._meta["role_name"]

    @property
    def lock_dir(self):
        return Path(self._meta["lock_dir"])

    @property
    def process(self):
        return self._process

    @property
    def supervision_identity(self):
        return {"schema": 1, "kind": "role", "scope": self._meta["scope"],
                "role_name": self.role_name, "attempt_id": self._meta["attempt_id"],
                "role_nonce_sha256": self._meta["nonce_sha256"],
                "owner_id": self._receipt["owner_id"]}

    def close_log(self):
        if self.log_fh is not None and not self.log_fh.closed:
            self.log_fh.close()

    def _hold(self, reason, *, protocol=False):
        self._held_reason = self._held_reason or reason
        self._protocol_uncertain = self._protocol_uncertain or protocol
        return RoleSupervisionHOLD(self._held_reason, owner=self)

    def _current(self):
        try:
            if self._released:
                raise self._hold("role_owner_already_released")
            plain_directory(self.lock_dir)
            info = self.lock_dir.lstat()
            if (info.st_dev, info.st_ino, info.st_mode) != self._lock_identity:
                raise self._hold("role_lock_identity_changed")
            if _read(self.lock_dir / "lock.json") != self._lock_bytes:
                raise self._hold("role_lock_owner_changed")
            if self._raw is None or _read(self.lock_dir / "role-process.json") != self._raw:
                raise self._hold("role_supervision_receipt_changed")
        except RoleSupervisionHOLD:
            raise
        except Exception as exc:
            raise self._hold("role_supervision_receipt_unverifiable") from exc

    def _write(self, stage, **fields):
        if self._raw is not None:
            self._current()
        payload = {**self._receipt, "stage": stage, **fields}
        _encoded(payload)  # Shared bounded JSON validation, not a new format.
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if self._raw is None:
            _publish(self.lock_dir / "role-process.json", raw)
        else:
            from orze.engine.process import _atomic_private_json
            _atomic_private_json(self.lock_dir / "role-process.json", payload)
            if _read(self.lock_dir / "role-process.json") != raw:
                raise ValueError("role_receipt_write_unconfirmed")
        self._receipt, self._raw = payload, raw
        self._current()

    def _handle(self):
        if not isinstance(self._process, SupervisedProcess) or self._ready is None:
            raise self._hold("role_supervised_handle_required")
        if (not _same(self._process.binding, self._ready)
                or not _same(self._ready["identity"], self.supervision_identity)
                or self._ready["command_sha256"] != self._meta["command_sha256"]
                or type(self._process.pid) is not int
                or self._process.pid != self._ready["worker"]["pid"]):
            raise self._hold("role_supervised_binding_changed")
        return self._process

    def prepare(self, cmd, *, env, cwd=None, stdout=None, stderr=None, prepare=None):
        try:
            self._current()
            if self._prepare_entered or self._held_reason:
                raise ValueError("role_prepare_already_attempted")
            if hashlib.sha256(canonical(cmd)).hexdigest() != self._meta["command_sha256"]:
                raise ValueError("role_command_changed")
            self._prepare_entered = True
            try:
                self._process = (prepare_supervised if prepare is None else prepare)(
                    cmd, identity=self.supervision_identity, env=env, cwd=cwd,
                    stdout=stdout, stderr=stderr)
            except SupervisionUnavailable:
                self._prepare_entered = False
                raise
            except SupervisionUncertain as exc:
                self._process = exc.process
                raise self._hold("role_prepare_uncertain", protocol=True) from exc
            if not isinstance(self._process, SupervisedProcess):
                raise ValueError("role_supervised_handle_required")
            self._ready = self._process.binding
            self._handle()
            self._write("READY", supervision=self._ready)
            return self._process
        except SupervisionUnavailable:
            raise
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_prepare_unconfirmed") from exc

    def bind_role(self, rp):
        try:
            self._current()
            self._handle()
            if (getattr(rp, "supervision_owner", None) is not self
                    or rp.process is not self._process or rp.role_name != self.role_name
                    or Path(rp.lock_dir).absolute() != self.lock_dir
                    or self._bound_role not in (None, rp)
                    or hashlib.sha256(rp.process_nonce.encode("ascii")).hexdigest() != self._meta["nonce_sha256"]):
                raise ValueError("role_holder_binding_changed")
            from orze.engine.role_delivery import delivery_reference
            ref = delivery_reference(rp) if rp.trigger_launch is not None else None
            if not _same(ref, self._meta["trigger_delivery"]):
                raise ValueError("role_trigger_holder_changed")
            db = getattr(rp, "trigger_delivery_db", None)
            if db != self._meta["trigger_delivery_db"]:
                raise ValueError("role_trigger_database_changed")
            native = getattr(rp, "native_result_ref", None)
            if native is not None and (native["attempt_id"] != self._meta["attempt_id"]
                    or native["nonce_sha256"] != self._meta["nonce_sha256"]):
                raise ValueError("role_result_holder_changed")
            rp._root_start_ticks = self._ready["worker"]["start_ticks"]
            rp._pgid = self._process.pid  # Observation compatibility only.
            fields = self._holder_fields(rp)
            if self._holder_snapshot is not None and not _same(fields, self._holder_snapshot):
                raise ValueError("role_holder_binding_changed")
            self._holder_snapshot = copy.deepcopy(fields)
            self._bound_role = rp
            with _GUARD:
                _BOUND[id(rp)] = (rp, self)
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_holder_unconfirmed") from exc

    def _holder_fields(self, rp):
        return {"role_name": rp.role_name, "lock_dir": str(Path(rp.lock_dir).absolute()),
                "process_nonce": rp.process_nonce, "trigger_launch": rp.trigger_launch,
                "trigger_delivery_db": rp.trigger_delivery_db,
                "native_result_ref": rp.native_result_ref}

    def _check_holder(self, rp):
        try:
            if (self._bound_role is not rp or self._holder_snapshot is None
                    or not _same(self._holder_fields(rp), self._holder_snapshot)):
                raise ValueError("role_holder_binding_changed")
        except Exception as exc:
            raise self._hold("role_holder_binding_changed") from exc

    def delivery_authority(self):
        """Return captured settlement inputs, never a mutable holder's route."""
        self._current()
        if self._held_reason:
            raise self._hold("role_owner_held")
        if self._bound_role is not None:
            self._check_holder(self._bound_role)
        return self._meta["trigger_delivery_db"], copy.deepcopy(self._trigger_token)

    def start(self):
        try:
            self._current()
            process = self._handle()
            if self._held_reason or self._go_attempted or self._stop_attempted:
                raise ValueError("role_go_not_authorized")
            if self._trigger_token is not None:
                _, state = _trigger_snapshot(self._meta, self._trigger_token)
                if state[0] != "STARTED" or type(state[2]) is not int or state[2] != process.pid:
                    raise ValueError("role_delivery_not_started")
            self._go_attempted = True  # Any uncertain GO attempt forbids PENDING.
            self._write("GO_REQUESTED")
            process.start()
            self._write("STARTED")
        except SupervisionUncertain as exc:
            raise self._hold("role_go_uncertain", protocol=True) from exc
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_go_unconfirmed") from exc

    def require_closed(self, ret=None):
        try:
            if self._held_reason:
                raise self._hold("role_owner_held")
            self._current()
            process = self._handle()
            if self._protocol_uncertain:
                raise ValueError("role_protocol_uncertain")
            actual = process.poll()
            if type(actual) is not int or (ret is not None and (type(ret) is not int or ret != actual)):
                raise ValueError("role_tree_not_closed")
            closure = process.closure_receipt()
            if (type(closure) is not dict or not _same(closure["binding"], self._ready)
                    or type(closure["worker_returncode"]) is not int
                    or closure["worker_returncode"] != actual):
                raise ValueError("role_tree_receipt_invalid")
            if self._closure is not None and not _same(self._closure, closure):
                raise ValueError("role_tree_receipt_changed")
            if self._closure is None:
                self._write("TREE_CLOSED", process_tree=closure)
                self._closure = copy.deepcopy(closure)
            return copy.deepcopy(self._closure)
        except SupervisionUncertain as exc:
            raise self._hold("role_tree_uncertain", protocol=True) from exc
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_closure_unconfirmed") from exc

    def abort(self, timeout=10):
        try:
            if self._protocol_uncertain:
                raise ValueError("role_stop_already_uncertain")
            if not self._prepare_entered and self._process is None:
                self._current()
                return None
            process = self._handle()
            if process.poll() is None:
                if self._stop_attempted:
                    raise ValueError("role_stop_already_attempted")
                self._stop_attempted = True
                if process.stop(timeout=timeout) is not True:
                    raise ValueError("role_stop_unconfirmed")
            return self.require_closed()
        except SupervisionUncertain as exc:
            raise self._hold("role_stop_uncertain", protocol=True) from exc
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_abort_unconfirmed") from exc

    def never_executed_proof(self):
        self._current()
        if self._go_attempted or self._held_reason or self._protocol_uncertain:
            raise self._hold("role_never_executed_unconfirmed")
        if not self._prepare_entered and self._process is None:
            return {"owner_id": self._receipt["owner_id"], "proof": "prepare_not_entered"}
        closure = self.require_closed()
        if (not closure["stop_requested"] or self._process._started is not False):
            raise self._hold("role_never_executed_unconfirmed")
        return {"owner_id": self._receipt["owner_id"], "proof": "no_go_tree_closed",
                "process_tree": closure}

    def release(self, *, outcome=None, exit_code=None):
        if self._released:
            return True
        try:
            self._current()
            if self._held_reason:
                raise ValueError("role_owner_held")
            no_exec = not self._go_attempted
            if no_exec:
                self.never_executed_proof()
            else:
                self.require_closed(exit_code)
            if self._trigger_token is not None:
                _, state = _trigger_snapshot(self._meta, self._trigger_token)
                if no_exec:
                    if (state[0] != "PENDING" or state[1] is None
                            or state[1][:2] != ("LAUNCHING", "PENDING")
                            or state[1][5] != self._meta["attempt_id"]):
                        raise ValueError("role_not_started_settlement_missing")
                elif (type(outcome) is not str or not outcome or state[0] != "TERMINAL"
                        or type(state[2]) is not int or state[2] != self.process.pid
                        or not _same(state[1], ("STARTED", "TERMINAL", outcome, exit_code, 1, self._meta["attempt_id"]))):
                    raise ValueError("role_terminal_settlement_mismatch")
            self._current()
            tree = snapshot(self.lock_dir, identity(self.lock_dir.lstat()))
            self._current()
            quarantine = self.lock_dir.parent / (".role-closed-" + secrets.token_hex(16))
            quarantine.mkdir(mode=0o700)
            _sync(quarantine.parent)
            rename_no_replace(self.lock_dir, quarantine / "content")
            _sync(quarantine.parent)
            _sync(quarantine)
            reclaim(tree, quarantine)
            self._released = True
            with _GUARD:
                _OWNERS.pop(id(self), None)
                if self._bound_role is not None:
                    _BOUND.pop(id(self._bound_role), None)
            return True
        except RoleSupervisionHOLD:
            raise
        except BaseException as exc:
            raise self._hold("role_release_unconfirmed") from exc


def begin_role_launch(metadata):
    owner = None
    try:
        if type(metadata) is not dict or set(metadata) != _KEYS:
            raise ValueError("role_launch_metadata_invalid")
        meta = json.loads(_encoded(metadata))
        for key in ("role_name", "attempt_id"):
            if type(meta[key]) is not str or not _TOKEN.fullmatch(meta[key]) or meta[key] in (".", ".."):
                raise ValueError("role_launch_token_invalid")
        for key in ("nonce_sha256", "command_sha256"):
            if type(meta[key]) is not str or not _SHA.fullmatch(meta[key]):
                raise ValueError("role_launch_hash_invalid")
        plain_directory(_path(meta["scope"]))
        lock = _path(meta["lock_dir"])
        plain_directory(lock)
        _reference(meta)
        lock_bytes = _read(lock / "lock.json")
        lock_data = json.loads(lock_bytes)
        if (type(lock_data) is not dict or lock_data.get("host") != socket.gethostname()
                or type(lock_data.get("pid")) is not int or lock_data["pid"] != os.getpid()):
            raise ValueError("role_lock_not_owned")
        info = lock.lstat()
        lock_identity = (info.st_dev, info.st_ino, info.st_mode)
        trigger_token, state = _trigger_snapshot(meta)
        if state is not None and state[0] != "LAUNCHING":
            raise ValueError("role_trigger_not_launching")
        with _GUARD:
            if len(_OWNERS) >= 1024 or any(item._lock_identity == lock_identity or item.lock_dir == lock for item in _OWNERS.values()):
                raise ValueError("role_owner_already_present")
            owner = RoleLaunch(meta, lock_identity, lock_bytes, trigger_token)
            _OWNERS[id(owner)] = owner
        owner._write("INTENT")
        return owner
    except BaseException as exc:
        if isinstance(exc, RoleSupervisionHOLD):
            raise
        if owner is not None:
            raise owner._hold("role_intent_unconfirmed") from exc
        raise RoleSupervisionHOLD("role_launch_metadata_unconfirmed") from exc


def supervised_role_owner(rp):
    if isinstance(rp, RoleLaunch):
        rp._current()
        return rp
    with _GUARD:
        bound = _BOUND.get(id(rp))
    declared = getattr(rp, "supervision_owner", None)
    owner = bound[1] if bound is not None and bound[0] is rp else declared
    if owner is not None:
        if not isinstance(owner, RoleLaunch) or declared is not owner or rp.process is not owner.process:
            raise RoleSupervisionHOLD("role_supervision_holder_changed", owner=owner if isinstance(owner, RoleLaunch) else None)
        owner._current()
        owner._handle()
        # During __post_init__, bind_role supplies the first immutable snapshot.
        # Every subsequent consumer must use that snapshot, never rebind from
        # public mutable role fields before writing another database/attempt.
        if bound is not None or owner._bound_role is not None:
            owner._check_holder(rp)
        return owner
    if isinstance(getattr(rp, "process", None), SupervisedProcess):
        raise RoleSupervisionHOLD("role_supervision_owner_missing")
    lock = getattr(rp, "lock_dir", None)
    if lock is not None:
        path = Path(lock).absolute() / "role-process.json"
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RoleSupervisionHOLD("role_supervision_owner_unverifiable") from exc
        try:
            receipt = json.loads(_read(path))
            if type(receipt) is not dict or receipt.get("schema_version") != 1:
                raise ValueError("role_supervision_owner_missing")
        except Exception as exc:
            raise RoleSupervisionHOLD("role_supervision_owner_missing") from exc
    return None
