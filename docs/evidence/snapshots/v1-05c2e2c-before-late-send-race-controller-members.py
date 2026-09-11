"""Opt-in controller membership, not a stop ACK or process adoption API.

The SQL action and the OS tree are independent obligations. Native intent is
inserted with its attempt; settlement follows the *whole* effect transaction.
Standalone runners settle explicitly after their own output/receipt cleanup.
Strong private owners survive public-map deletion and unknown preparation.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3

from orze.engine.controller_control import ControllerHOLD, ControllerQuiescing, current_controller
from orze.engine.supervisor_worker import canonical


_SQL = """CREATE TABLE controller_members (
    member_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    controller_id TEXT NOT NULL COLLATE BINARY,
    payload_json TEXT NOT NULL
)"""
_NATIVE = {"training", "evaluation", "posthoc", "post_script", "pre_script", "artifact_preflight"}
_REPORTS = {"launch_failure_report", "pre_script_failure_report", "artifact_preflight_failure_report"}
_OWNERS = {}  # Strong membership; never recovered from a mutable process label.
_HANDLES = {}
_TRANSACTIONS = {}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_MEMBERS = 4096  # Explicit per-instance limit; no age-based unknown eviction.


def _encoded(value):
    data = canonical(value)
    if len(data) > 65536:
        raise ControllerHOLD("controller_member_metadata_limit")
    return data.decode("utf-8")


def _schema(conn, *, create=False):
    row = conn.execute("SELECT type,name,sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                       ("controller_members",)).fetchone()
    if row is None and create:
        conn.execute(_SQL)
        return _schema(conn)
    normalize = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (row is None or row[0] != "table" or row[1] != "controller_members"
            or normalize(row[2]) != normalize(_SQL)):
        raise ControllerHOLD("controller_member_schema_invalid")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' "
                    "AND tbl_name=? COLLATE NOCASE", ("controller_members",)).fetchone():
        raise ControllerHOLD("controller_member_trigger_unsupported")


@dataclass(eq=False)
class _Member:
    ctx: object
    payload: dict
    encoded: str | None = None
    process: object = None
    held: str | None = None

    @property
    def key(self):
        return self.payload["member_id"]


@dataclass
class _Transaction:
    ctx: object
    conn: object
    members: list = field(default_factory=list)
    finished: dict = field(default_factory=dict)


@contextmanager
def _connection(ctx, conn=None):
    owned = conn is None
    if owned:
        conn = sqlite3.connect(ctx.db_path.as_uri() + "?mode=rw", uri=True, timeout=0.25)
    try:
        if owned:
            conn.execute("BEGIN IMMEDIATE")
        elif not conn.in_transaction:
            raise ControllerHOLD("controller_member_writer_required")
        ctx.poll_control(conn=conn)
        _schema(conn, create=True)
        yield conn
        if owned:
            conn.commit()
            if conn.in_transaction:
                raise ControllerHOLD("controller_member_commit_unconfirmed")
            ctx.poll_control(conn=conn)
    except BaseException:
        if owned and conn.in_transaction:
            conn.rollback()
        raise
    finally:
        if owned:
            conn.close()


def _write(member, *, conn=None, **changes):
    expected = member.encoded
    payload = {**member.payload, **changes}
    encoded = _encoded(payload)
    with _connection(member.ctx, conn) as db:
        if expected is None:
            count = db.execute("INSERT INTO main.controller_members VALUES (?,?,?)",
                               (member.key, member.ctx.controller_id, encoded)).rowcount
        else:
            count = db.execute("UPDATE main.controller_members SET payload_json=? "
                "WHERE member_id=? AND controller_id=? AND payload_json=?",
                (encoded, member.key, member.ctx.controller_id, expected)).rowcount
        row = db.execute("SELECT controller_id, CASE WHEN length(CAST(payload_json AS BLOB))<=65536 "
                         "THEN payload_json ELSE NULL END FROM main.controller_members WHERE member_id=?",
                         (member.key,)).fetchone()
        if count != 1 or row is None or tuple(row) != (member.ctx.controller_id, encoded):
            raise ControllerHOLD("controller_member_write_unconfirmed")
    if conn is None:
        _verify_committed(member, encoded)
    member.payload, member.encoded = payload, encoded


def _verify_committed(member, expected):
    # This must also run after a caller-owned member settlement transaction;
    # native _write(conn=...) cannot certify its enclosing commit in advance.
    with closing(sqlite3.connect(member.ctx.db_path.as_uri() + "?mode=ro", uri=True, timeout=0.25)) as check:
        member.ctx.poll_control(conn=check)
        _schema(check)
        actual = check.execute("SELECT controller_id, CASE WHEN length(CAST(payload_json AS BLOB))<=65536 "
                               "THEN payload_json ELSE NULL END FROM main.controller_members WHERE member_id=?",
                               (member.key,)).fetchone()
        if actual is None or tuple(actual) != (member.ctx.controller_id, expected):
            raise ControllerHOLD("controller_member_committed_row_changed")


def _current(member, conn):
    _schema(conn)
    row = conn.execute("SELECT controller_id, CASE WHEN length(CAST(payload_json AS BLOB))<=65536 "
                       "THEN payload_json ELSE NULL END FROM main.controller_members WHERE member_id=?",
                       (member.key,)).fetchone()
    if (member.held or row is None or tuple(row) != (member.ctx.controller_id, member.encoded)
            or _OWNERS.get(member.key) is not member):
        raise ControllerHOLD("controller_member_ownership_changed")


def _new(ctx, kind, identity, *, source=None, command_sha256=None, conn=None, no_os=False):
    ctx.poll_control(conn=conn)
    if ctx.quiescing:
        _qualified_report(ctx, conn, kind, source)
    else:
        ctx.check_admission(conn=conn)
    if sum(m.ctx is ctx for m in _OWNERS.values()) >= MAX_MEMBERS:
        raise ControllerHOLD("controller_membership_limit")
    member = _Member(ctx, {"schema": 1, "member_id": secrets.token_hex(24),
        "controller_id": ctx.controller_id, "kind": kind, "identity": copy.deepcopy(identity),
        "source": source, "command_sha256": command_sha256,
        "os_state": "NOT_REQUIRED" if no_os else "INTENT", "action_state": "PENDING",
        "ready": None, "closure": None, "outcome": None,
        "output_sha256": None, "output_bytes": None, "terminal_sha256": None,
        "hold_reason": None})
    _OWNERS[member.key] = member  # Retain even an ambiguous INSERT response.
    ticket = _TRANSACTIONS.get(id(conn)) if conn is not None else None
    if ticket is not None:
        ticket.members.append(member)
    try:
        _write(member, conn=conn)
    except BaseException:
        hold_member(member, "controller_member_intent_unconfirmed")
        raise
    return member


def _lookup(process):
    pair = _HANDLES.get(id(process))
    if pair is not None and pair[0] is process:
        return pair[1]
    if current_controller() is not None:
        raise ControllerHOLD("controller_process_membership_missing")
    return None


def hold_process(process, reason="controller_process_uncertain"):
    pair = _HANDLES.get(id(process))
    if pair is not None and pair[0] is process:
        hold_member(pair[1], reason)


def hold_member(member, reason="controller_member_uncertain"):
    if member is None:
        return
    member.held = member.held or reason
    # No second writer while a native caller owns this DB's transaction.
    if any(tx.ctx is member.ctx for tx in _TRANSACTIONS.values()):
        return
    try:
        with member.ctx.guard():
            _write(member, action_state="HOLD", hold_reason=member.held)
    except BaseException:
        pass  # Strong owner remains even if durable diagnostics cannot commit.
    member.ctx.hold(member.held)


def begin_transaction(lake, idea_dir):
    ctx = current_controller()
    if ctx is None:
        return None
    with ctx.guard():
        folder = Path(idea_dir).absolute()
        if folder.parent != ctx.scope or folder.name in {"", ".", ".."}:
            raise ControllerHOLD("controller_attempt_scope_mismatch")
        ctx.poll_control(conn=lake.conn)
        if id(lake.conn) in _TRANSACTIONS:
            raise ControllerHOLD("controller_nested_transaction")
        ticket = _Transaction(ctx, lake.conn)
        _TRANSACTIONS[id(lake.conn)] = ticket
        return ticket


def attempt_created(conn, ref, binding):
    ctx = current_controller()
    if ctx is None:
        return
    with ctx.guard():
        no_os = ref.phase in _REPORTS
        if ref.phase not in _NATIVE and not no_os:
            raise ControllerHOLD("controller_attempt_phase_unsupported")
        # Only the three existing source-bound no-OS publishers may finish
        # admitted work after quiesce. This is not a generic cleanup flag.
        source = copy.deepcopy(binding.get("source_attempt")) if no_os else None
        if no_os:
            from orze.core.execution_attempts import AttemptRef, require_current
            if type(source) is not dict or set(source) != {"task_id", "phase", "attempt_id", "generation"}:
                raise ControllerHOLD("controller_report_source_required")
            require_current(conn, AttemptRef(**source), states=("TERMINAL", "NOT_STARTED"))
        identity = {"attempt_ref": asdict(ref), "scope": str(ctx.scope / ref.task_id)}
        member = _new(ctx, ref.phase, identity, source=source, conn=conn, no_os=no_os)


def _for_ref(ctx, ref):
    identity = asdict(ref)
    found = [m for m in _OWNERS.values() if m.ctx is ctx
             and _encoded(m.payload["identity"].get("attempt_ref")) == _encoded(identity)]
    if len(found) != 1:
        raise ControllerHOLD("controller_attempt_membership_missing")
    return found[0]


def _qualified_report(ctx, conn, kind, source):
    from orze.core.execution_attempts import AttemptRef, require_current
    allowed = {"launch_failure_report": {"training", "evaluation", "posthoc"},
               "pre_script_failure_report": {"pre_script"},
               "artifact_preflight_failure_report": {"artifact_preflight"}}
    if kind not in allowed:
        raise ControllerQuiescing("controller_quiescing")
    if (conn is None or kind not in allowed or type(source) is not dict
            or set(source) != {"task_id", "phase", "attempt_id", "generation"}):
        raise ControllerHOLD("controller_quiesce_report_source_required")
    ref = AttemptRef(**source)
    if ref.phase not in allowed[kind]:
        raise ControllerHOLD("controller_quiesce_report_phase_invalid")
    member = _for_ref(ctx, ref)
    _current(member, conn)
    row = require_current(conn, ref, states=("TERMINAL", "NOT_STARTED"))
    if (member.payload["action_state"] != "SETTLED"
            or member.payload["os_state"] not in {"CLOSED", "NO_EXECUTION"}
            or hashlib.sha256(_encoded(row).encode()).hexdigest() != member.payload["terminal_sha256"]):
        raise ControllerHOLD("controller_quiesce_report_source_unsettled")


def record_static_preflight_rejection(ref, binding, captured):
    """Explicit producer proof on its known rejection branch, before prepare.

    A missing handle is never sufficient. The producer supplies its captured
    rejection and admitted inputs; any PREPARING transition invalidates this
    narrowly scoped no-execution authority. Native effect settlement is later.
    """
    ctx = current_controller()
    if ctx is None:
        return
    from orze.core.execution_attempts import require_current
    reasons = {"network_policy_invalid", "script_missing", "config_missing",
               "train_extra_env_not_mapping", "offline_flags_conflict_with_required_network"}
    with ctx.guard(), _connection(ctx) as conn:
        member = _for_ref(ctx, ref)
        _current(member, conn)
        row = require_current(conn, ref, states=("LAUNCHING",))
        rejection = captured.rejection
        if (ref.phase != "artifact_preflight" or type(rejection) is not dict
                or rejection.get("reason") not in reasons
                or member.process is not None or member.payload["os_state"] != "INTENT"
                or _encoded(row["binding"]) != _encoded(binding)
                or _encoded(captured.identity) != _encoded(binding["preflight_identity"])
                or hashlib.sha256(canonical(captured.command)).hexdigest() != binding["command_sha256"]):
            raise ControllerHOLD("controller_static_no_execution_unconfirmed")
        _write(member, conn=conn, os_state="NO_EXECUTION")
    _verify_committed(member, member.encoded)


def member_limit_reached(ctx=None):
    """Finite-profile soft limit; retain half the hard cap for drain reports."""
    ctx = current_controller() if ctx is None else ctx
    if ctx is None:
        return False
    with ctx.guard():
        ctx.poll_control()
        return sum(m.ctx is ctx for m in _OWNERS.values()) >= MAX_MEMBERS // 2


def prove_drained(ctx):
    """Prove a bounded member inventory, never controller exit or resource ACK."""
    from orze.core import execution_attempts as attempts
    from orze.engine.attempt_effect_receipts import _scan
    from orze.engine.supervised_process import SupervisedProcess
    with ctx.guard():
        ctx.poll_control()
        if not ctx.quiescing or any(tx.ctx is ctx for tx in _TRANSACTIONS.values()):
            raise ControllerHOLD("controller_drain_not_quiescent")
        members = sorted((m for m in _OWNERS.values() if m.ctx is ctx), key=lambda m: m.key)
        if len(members) > MAX_MEMBERS:
            raise ControllerHOLD("controller_drain_inventory_limit")
        digest = hashlib.sha256()
        checked_folders = {}
        with _connection(ctx) as conn:
            keys = [r[0] for r in conn.execute(
                "SELECT member_id FROM controller_members WHERE controller_id=? LIMIT ?",
                (ctx.controller_id, MAX_MEMBERS + 1))]
            if set(keys) != {m.key for m in members} or len(keys) != len(members):
                raise ControllerHOLD("controller_membership_inventory_changed")
            for member in members:
                _current(member, conn)
                value = member.payload
                if (value["action_state"] != "SETTLED" or value["hold_reason"] is not None
                        or value["os_state"] not in {"CLOSED", "NO_EXECUTION", "NOT_REQUIRED"}):
                    raise ControllerHOLD("controller_member_not_drained")
                if value["os_state"] == "CLOSED":
                    process = member.process
                    if (type(process) is not SupervisedProcess or _lookup(process) is not member
                            or type(process.returncode) is not int or process._uncertainty
                            or process._supervisor.poll() != 0
                            or _encoded(process.closure_receipt()) != _encoded(value["closure"])
                            or _encoded(value["closure"]["binding"]) != _encoded(value["ready"])):
                        raise ControllerHOLD("controller_member_tree_changed")
                if value["kind"] in _NATIVE | _REPORTS:
                    ref = attempts.AttemptRef(**value["identity"]["attempt_ref"])
                    if not attempts._schema(conn):
                        raise ControllerHOLD("controller_member_attempt_missing")
                    row = attempts._row(conn.execute(attempts._SELECT +
                        " WHERE attempt_id=? COLLATE BINARY", (ref.attempt_id,)).fetchone())
                    if (row is None or {k: row[k] for k in ("task_id", "phase", "attempt_id", "generation")} != asdict(ref)
                            or row["state"] not in {"TERMINAL", "NOT_STARTED"}
                            or hashlib.sha256(_encoded(row).encode()).hexdigest() != value["terminal_sha256"]):
                        raise ControllerHOLD("controller_member_terminal_changed")
                    folder = ctx.scope / ref.task_id
                    if folder not in checked_folders:
                        checked_folders[folder] = _scan(folder)
                    if checked_folders[folder].get(ref.attempt_id) != (
                            row["terminal"].get("effect_receipt_sha256"), True):
                        raise ControllerHOLD("controller_member_effect_changed")
                digest.update(canonical([member.key, hashlib.sha256(member.encoded.encode()).hexdigest()]))
        ctx.poll_control()
        return {"schema": 1, "controller_id": ctx.controller_id,
                "member_count": len(members), "members_sha256": digest.hexdigest()}


def require_settled_process(process):
    """Private membership gate for exact captured shutdown slots, not bool ACK."""
    member = _lookup(process)
    if member is None:
        raise ControllerHOLD("controller_shutdown_member_missing")
    with member.ctx.guard(), _connection(member.ctx) as conn:
        _current(member, conn)
        if member.payload["action_state"] != "SETTLED" or member.payload["os_state"] != "CLOSED":
            raise ControllerHOLD("controller_shutdown_member_unsettled")


def attempt_finished(conn, ref):
    ctx = current_controller()
    if ctx is None:
        return
    with ctx.guard():
        member = _for_ref(ctx, ref)
        ticket = _TRANSACTIONS.get(id(conn))
        if ticket is None or ticket.conn is not conn:
            raise ControllerHOLD("controller_attempt_settlement_requires_effect_transaction")
        from orze.core.execution_attempts import require_current
        row = require_current(conn, ref, states=("TERMINAL", "NOT_STARTED"))
        _current(member, conn)
        ticket.finished[member.key] = (member, ref, _encoded(row))


def end_transaction(ticket, *, success):
    if ticket is None:
        return
    _TRANSACTIONS.pop(id(ticket.conn), None)
    if not success:
        # A failed effect transaction is not action settlement, even if SQL
        # happened to commit or a worker's public holder has disappeared.
        affected = ticket.members + [value[0] for value in ticket.finished.values()]
        for member in affected:
            hold_member(member, "controller_action_transaction_unconfirmed")
        return
    try:
        with ticket.ctx.guard():
            from orze.core.execution_attempts import require_current
            for member, ref, expected in ticket.finished.values():
                with _connection(ticket.ctx) as conn:
                    _current(member, conn)
                    row = require_current(conn, ref, states=("TERMINAL", "NOT_STARTED"))
                    if _encoded(row) != expected:
                        raise ControllerHOLD("controller_action_terminal_changed")
                    if member.payload["os_state"] not in {"CLOSED", "NO_EXECUTION", "NOT_REQUIRED"}:
                        raise ControllerHOLD("controller_action_tree_not_closed")
                    closure = member.payload["closure"] or {}
                    stopped = closure.get("stop_requested") or closure.get("forced_cleanup")
                    outcome = "not_started" if row["state"] == "NOT_STARTED" else "interrupted" if stopped else "completed"
                    _write(member, conn=conn, action_state="SETTLED",
                           terminal_sha256=hashlib.sha256(expected.encode()).hexdigest(),
                           outcome=outcome)
                _verify_committed(member, member.encoded)
    except BaseException:
        for member, _, _ in ticket.finished.values():
            hold_member(member, "controller_action_settlement_unconfirmed")
        raise


def before_prepare(identity, command_sha256):
    ctx = current_controller()
    if ctx is None:
        return None
    with ctx.guard():
        ctx.check_admission()
        if "attempt_ref" in identity:
            from orze.core.execution_attempts import AttemptRef
            member = _for_ref(ctx, AttemptRef(**identity["attempt_ref"]))
        else:
            kind = identity.get("kind")
            if kind not in {"role", "bounded_executor", "controller_probe"}:
                raise ControllerHOLD("controller_process_kind_unsupported")
            if identity.get("scope") != str(ctx.scope):
                raise ControllerHOLD("controller_process_scope_mismatch")
            matches = [m for m in _OWNERS.values() if m.ctx is ctx
                       and _encoded(m.payload["identity"]) == _encoded(identity)]
            member = matches[0] if len(matches) == 1 else _new(ctx, kind, identity, command_sha256=command_sha256)
        if (member.held or member.process is not None or member.payload["os_state"] != "INTENT"
                or _encoded(member.payload["identity"]) != _encoded(identity)
                or member.payload["command_sha256"] not in (None, command_sha256)):
            raise ControllerHOLD("controller_prepare_member_conflict")
        _write(member, command_sha256=command_sha256, os_state="PREPARING")
        return member


def bind_prepared(member, process, *, ready=False):
    if member is None:
        return
    with member.ctx.guard():
        if member.process is not None and member.process is not process:
            # The primitive replaces its pre-Popen fallback with its real
            # constructor exactly once. Both remain mapped to the same owner.
            if member.payload["os_state"] != "PREPARING":
                raise ControllerHOLD("controller_process_rebound")
        member.process = process
        _HANDLES[id(process)] = (process, member)
        if ready:
            binding = process.binding
            if (type(binding) is not dict or _encoded(binding["identity"]) != _encoded(member.payload["identity"])
                    or binding["command_sha256"] != member.payload["command_sha256"]):
                raise ControllerHOLD("controller_ready_binding_changed")
            _write(member, os_state="READY", ready=binding)


def prepare_failed(member, *, no_execution=False):
    if member is None:
        return
    if no_execution:
        with member.ctx.guard():
            _write(member, os_state="NO_EXECUTION")
        return
    hold_member(member, "controller_prepare_uncertain")


def _stop_once(process):
    if process._closed is not None:
        # _accept validated this bound TREE_CLOSED. No writers remain to
        # stop; ordinary poll still requires the supervisor's normal exit.
        return
    if not process._stop_sent and process.returncode is None:
        # One small nonblocking local socket frame. No recursive poll/wait and
        # no background receiver competing for private protocol messages.
        from orze.engine.supervisor_worker import send_frame
        process._stop_sent = True
        try:
            process._channel.setblocking(False)
            send_frame(process._channel, {"command": "STOP", "nonce": process._nonce})
        except BaseException:
            process._fail("controller_stop_send_uncertain")


@contextmanager
def start_guard(process):
    member = _lookup(process)
    if member is None:
        yield True
        return
    with member.ctx.guard():
        member.ctx.poll_control()
        with _connection(member.ctx) as conn:
            _current(member, conn)
        if (member.process is not process or member.payload["os_state"] != "READY"
                or _encoded(process.binding) != _encoded(member.payload["ready"])):
            raise ControllerHOLD("controller_go_binding_changed")
        if member.ctx.quiescing:
            _stop_once(process)
            _write(member, os_state="STOP_REQUESTED")
            yield False
            return
        _write(member, os_state="GO_REQUESTED")
        try:
            yield True
            _write(member, os_state="RUNNING")
        except BaseException:
            hold_member(member, "controller_go_uncertain")
            raise


def poll_process(process):
    member = _lookup(process)
    if member is None:
        return
    # HOLD forbids certification, but does not manufacture a second STOP or
    # prevent the owning caller from draining an already-known handle.
    with member.ctx.guard():
        try:
            member.ctx.poll_control()
        except ControllerHOLD:
            return
        if member.ctx.quiescing:
            _stop_once(process)


def record_closed(process):
    member = _lookup(process)
    if member is None:
        return
    with member.ctx.guard():
        if member.held:
            return  # Physical reap cannot erase action/IO uncertainty.
        try:
            member.ctx.poll_control()
        except ControllerHOLD:
            member.held = "controller_tree_record_unconfirmed"
            return  # Reaping can succeed even while certification is held.
        closure = process._closed
        if (member.process is not process or type(process.returncode) is not int
                or type(closure) is not dict or _encoded(closure["binding"]) != _encoded(member.payload["ready"])):
            hold_member(member, "controller_tree_binding_changed")
            raise ControllerHOLD("controller_tree_binding_changed")
        if member.payload["os_state"] == "CLOSED":
            if _encoded(member.payload["closure"]) != _encoded(closure):
                raise ControllerHOLD("controller_tree_receipt_changed")
            return
        try:
            _write(member, os_state="CLOSED", closure=closure)
        except BaseException:
            hold_member(member, "controller_tree_record_unconfirmed")
            raise


def settle_process(process, *, outcome, output_sha256=None, output_bytes=None):
    member = _lookup(process)
    if member is None:
        return
    if (type(outcome) is not str or outcome not in {"completed", "interrupted", "not_started"}
            or (output_sha256 is not None and (type(output_sha256) is not str or not _SHA.fullmatch(output_sha256)))
            or (output_bytes is not None and (type(output_bytes) is not int or output_bytes < 0))
            or ((output_sha256 is None) != (output_bytes is None))):
        raise ControllerHOLD("controller_process_settlement_metadata_invalid")
    with member.ctx.guard():
        if member.payload["kind"] in _NATIVE or member.payload["kind"] in _REPORTS:
            raise ControllerHOLD("controller_native_settlement_requires_transaction")
        process.poll()
        if member.held or member.payload["os_state"] != "CLOSED":
            raise ControllerHOLD("controller_process_not_closed")
        closure = member.payload["closure"]
        if outcome == "completed" and (closure["stop_requested"] or closure["forced_cleanup"]):
            raise ControllerHOLD("controller_stopped_process_not_success")
        _write(member, action_state="SETTLED", outcome=outcome,
               output_sha256=output_sha256, output_bytes=output_bytes)


def role_intent(owner):
    ctx = current_controller()
    if ctx is None:
        return None
    with ctx.guard():
        if owner.supervision_identity.get("scope") != str(ctx.scope):
            raise ControllerHOLD("controller_role_scope_mismatch")
        return _new(ctx, "role", owner.supervision_identity,
                    command_sha256=owner._meta["command_sha256"])


def role_released(owner):
    member = getattr(owner, "_controller_member", None)
    if member is None:
        return
    if owner.process is not None:
        closure = owner.process.closure_receipt()
        settle_process(owner.process, outcome="interrupted" if closure["stop_requested"] else "completed")
    else:
        # Only the role's verified never-entered-prepare / trigger rollback and
        # exact lock release authorize this no-OS branch, not a missing handle.
        with member.ctx.guard():
            if owner._prepare_entered or owner._go_attempted or owner._held_reason or not owner._released:
                raise ControllerHOLD("controller_role_no_execution_unconfirmed")
            _write(member, os_state="NO_EXECUTION", action_state="SETTLED", outcome="not_started")


def members_snapshot(ctx=None):
    """Diagnostic evidence only; an empty result is NOT a controller ACK."""
    ctx = current_controller() if ctx is None else ctx
    if ctx is None:
        return []
    with ctx.guard(), _connection(ctx) as conn:
        rows = conn.execute("SELECT member_id FROM main.controller_members WHERE controller_id=?",
                            (ctx.controller_id,)).fetchall()
        members = [m for m in _OWNERS.values() if m.ctx is ctx]
        if {r[0] for r in rows} != {m.key for m in members}:
            raise ControllerHOLD("controller_membership_inventory_changed")
        result = []
        for member in members:
            if not member.held:
                _current(member, conn)
            result.append({**copy.deepcopy(member.payload), "local_hold_reason": member.held})
        return result
