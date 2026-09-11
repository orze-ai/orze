"""Internal, opt-in controller registration; NOT a stop/restart capability.

A process may bind one persistent IdeaLake and results directory. Registration
owns a no-age-takeover namespace and remains present after exit or uncertainty.
There is deliberately no public unbind, release, ACK, adoption or restart API
here. Product lifecycle integration must prove member/action/resource drain
and observed controller exit before it can add any such operation.

The strong process-global binding applies to framework threads too. A fork
cannot inherit authority or silently fall back to an unregistered raw launcher.
This is an ownership protocol, not a sandbox against hostile same-UID code.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import stat
import threading

from orze.core.idea_source_lock import _acquire, idea_source_lock_owned
from orze.core.sqlite_policy import inspect_shared_database_policy
from orze.engine.supervisor_worker import canonical, process_identity


class ControllerHOLD(RuntimeError):
    """No authority to admit or certify controller work."""


class ControllerQuiescing(ControllerHOLD):
    """Known local admission cancellation, not a closure certificate."""


_SQL = """CREATE TABLE controller_instances (
    controller_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    scope TEXT NOT NULL COLLATE BINARY UNIQUE,
    identity_json TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('ACTIVE', 'QUIESCING', 'HOLD')),
    request_id TEXT,
    hold_reason TEXT,
    CHECK ((phase = 'HOLD') = (hold_reason IS NOT NULL)),
    CHECK (phase != 'QUIESCING' OR request_id IS NOT NULL)
)"""
_COLUMNS = [("controller_id", "TEXT", 1, 1), ("scope", "TEXT", 1, 0),
            ("identity_json", "TEXT", 1, 0), ("phase", "TEXT", 1, 0),
            ("request_id", "TEXT", 0, 0), ("hold_reason", "TEXT", 0, 0)]
_SQL_V2 = """CREATE TABLE controller_instances (
    controller_id TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    scope TEXT NOT NULL COLLATE BINARY,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    predecessor TEXT COLLATE BINARY UNIQUE,
    identity_json TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('ACTIVE', 'QUIESCING', 'HOLD')),
    request_id TEXT,
    hold_reason TEXT,
    UNIQUE (scope, generation),
    CHECK ((generation = 0) = (predecessor IS NULL)),
    CHECK ((phase = 'HOLD') = (hold_reason IS NOT NULL)),
    CHECK (phase != 'QUIESCING' OR request_id IS NOT NULL)
)"""
_COLUMNS_V2 = _COLUMNS[:2] + [("generation", "INTEGER", 1, 0),
                            ("predecessor", "TEXT", 0, 0)] + _COLUMNS[2:]
_HEAD_SQL = """CREATE TABLE controller_scope_heads (
    scope TEXT NOT NULL COLLATE BINARY PRIMARY KEY,
    current_id TEXT NOT NULL COLLATE BINARY UNIQUE,
    generation INTEGER NOT NULL CHECK (generation >= 0),
    pending_grant TEXT COLLATE BINARY
)"""
_HEAD_COLUMNS = [("scope", "TEXT", 1, 1), ("current_id", "TEXT", 1, 0),
                 ("generation", "INTEGER", 1, 0), ("pending_grant", "TEXT", 0, 0)]
_TOKEN = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_ID = re.compile(r"[0-9a-f]{48}\Z")
_CURRENT = None
_GLOBAL_GUARD = threading.RLock()


def _table_schema(conn, name, sql, expected_columns, expected_keys):
    row = conn.execute("SELECT type, name, sql FROM main.sqlite_master "
                       "WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    normalize = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if (row is None or row[0] != "table" or row[1] != name
            or normalize(row[2]) != normalize(sql)):
        raise ControllerHOLD("controller_schema_incompatible")
    columns = conn.execute("SELECT * FROM pragma_table_xinfo(?, 'main')", (name,)).fetchall()
    if ([(r[1], r[2], r[3], r[5]) for r in columns] != expected_columns
            or any(r[4] is not None or r[6] != 0 for r in columns)):
        raise ControllerHOLD("controller_schema_columns_invalid")
    keys = []
    for index in conn.execute("SELECT * FROM pragma_index_list(?, 'main')", (name,)).fetchall():
        if not index[2]:
            continue
        entries = conn.execute("SELECT name, coll, desc FROM pragma_index_xinfo(?, 'main') "
                               "WHERE key = 1 ORDER BY seqno", (index[1],)).fetchall()
        if index[4] or any(r[1] != "BINARY" or r[2] != 0 for r in entries):
            raise ControllerHOLD("controller_schema_identity_invalid")
        keys.append((index[3], tuple(r[0] for r in entries)))
    if sorted(keys) != sorted(expected_keys):
        raise ControllerHOLD("controller_schema_identity_invalid")
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='trigger' "
                    "AND tbl_name = ? COLLATE NOCASE LIMIT 1",
                    (name,)).fetchone() is not None:
        raise ControllerHOLD("controller_schema_trigger_unsupported")


def _schema(conn, *, create=False, protocol=None):
    row = conn.execute("SELECT sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                       ("controller_instances",)).fetchone()
    head = conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=? COLLATE NOCASE",
                        ("controller_scope_heads",)).fetchone()
    if row is None and create:
        if not conn.in_transaction or head is not None:
            raise ControllerHOLD("controller_schema_transaction_required")
        version = 1 if protocol is None else protocol
        if type(version) is not int or version not in (1, 2):
            raise ControllerHOLD("controller_protocol_invalid")
        conn.execute(_SQL if version == 1 else _SQL_V2)
        if version == 2:
            conn.execute(_HEAD_SQL)
        return _schema(conn, protocol=version)
    normalize = lambda value: " ".join(str(value).strip().rstrip(";").split())
    if row is None:
        raise ControllerHOLD("controller_schema_incompatible")
    version = 1 if normalize(row[0]) == normalize(_SQL) else 2 if normalize(row[0]) == normalize(_SQL_V2) else None
    if version is None or protocol is not None and protocol != version:
        raise ControllerHOLD("controller_schema_incompatible")
    if version == 1:
        if head is not None:
            raise ControllerHOLD("controller_schema_mixed_protocols")
        _table_schema(conn, "controller_instances", _SQL, _COLUMNS,
                      [("pk", ("controller_id",)), ("u", ("scope",))])
    else:
        _table_schema(conn, "controller_instances", _SQL_V2, _COLUMNS_V2,
                      [("pk", ("controller_id",)), ("u", ("scope", "generation")),
                       ("u", ("predecessor",))])
        _table_schema(conn, "controller_scope_heads", _HEAD_SQL, _HEAD_COLUMNS,
                      [("pk", ("scope",)), ("u", ("current_id",))])
    return version


def registration_version(conn):
    """Strict persisted schema version; no migration or best-effort fallback."""
    return _schema(conn)


def _head(conn, scope):
    row = conn.execute("SELECT CASE WHEN length(CAST(current_id AS BLOB))=48 THEN current_id ELSE '' END,"
                       "generation,CASE WHEN pending_grant IS NULL THEN NULL "
                       "WHEN length(CAST(pending_grant AS BLOB))<=128 THEN pending_grant ELSE '' END "
                       "FROM main.controller_scope_heads "
                       "WHERE scope=? COLLATE BINARY", (str(scope),)).fetchone()
    if (row is None or type(row[0]) is not str or not _ID.fullmatch(row[0])
            or type(row[1]) is not int or not 0 <= row[1] <= 2**63 - 1
            or row[2] is not None and (type(row[2]) is not str or not _TOKEN.fullmatch(row[2]))):
        raise ControllerHOLD("controller_scope_head_invalid")
    return tuple(row)


def current_instance(conn, scope):
    """Return the current six-field instance row, never a historical candidate.

    A pending grant remains readable to its stop/handoff observer, but cannot
    authorize a live context's admission. That check is in ControllerContext.
    """
    protocol = _schema(conn)
    scope = str(Path(scope).absolute())
    fields = ("controller_id,scope,CASE WHEN typeof(identity_json)='text' "
              "AND length(CAST(identity_json AS BLOB))<=16384 THEN identity_json ELSE NULL END,"
              "phase,CASE WHEN request_id IS NULL THEN NULL WHEN length(CAST(request_id AS BLOB))<=128 "
              "THEN request_id ELSE '' END,CASE WHEN hold_reason IS NULL THEN NULL "
              "WHEN length(CAST(hold_reason AS BLOB))<=128 THEN hold_reason ELSE '' END")
    if protocol == 1:
        rows = conn.execute("SELECT " + fields + " FROM main.controller_instances "
                            "WHERE scope=? COLLATE BINARY LIMIT 2", (scope,)).fetchall()
        if len(rows) != 1:
            raise ControllerHOLD("controller_current_instance_missing")
        row = tuple(rows[0])
    else:
        head = _head(conn, scope)
        raw = conn.execute("SELECT " + fields + ",generation,predecessor FROM main.controller_instances "
                           "WHERE controller_id=? COLLATE BINARY", (head[0],)).fetchone()
        if (raw is None or raw[1] != scope or raw[6] != head[1]
                or (raw[6] == 0 and raw[7] is not None)
                or (raw[6] > 0 and (type(raw[7]) is not str or not _ID.fullmatch(raw[7])))):
            raise ControllerHOLD("controller_scope_head_instance_mismatch")
        row = tuple(raw[:6])
    if (type(row[2]) is not str or len(row[2].encode("utf-8")) > 16384
            or row[3] not in {"ACTIVE", "QUIESCING", "HOLD"}
            or any(x is not None and (type(x) is not str or not _TOKEN.fullmatch(x)) for x in row[4:])
            or ((row[3] == "HOLD") != (row[5] is not None))
            or (row[3] == "QUIESCING" and row[4] is None)):
        raise ControllerHOLD("controller_registration_changed")
    return row


def _path(value, *, directory):
    path = Path(value).absolute()
    if ".." in path.parts or path == Path(path.anchor):
        raise ControllerHOLD("controller_path_invalid")
    for parent in path.parents:
        if parent.is_symlink() or not parent.is_dir():
            raise ControllerHOLD("controller_path_redirected")
    info = path.lstat()
    if (directory and not stat.S_ISDIR(info.st_mode)
            or not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
        raise ControllerHOLD("controller_path_type_invalid")
    if len(os.fsencode(path)) > 4096:
        raise ControllerHOLD("controller_path_limit")
    return path, (info.st_dev, info.st_ino)


def _route(conn):
    rows = conn.execute("PRAGMA database_list").fetchall()
    paths = [row[2] for row in rows if row[1] == "main"]
    if len(paths) != 1 or not paths[0] or paths[0] == ":memory:":
        raise ControllerHOLD("controller_persistent_database_required")
    if not inspect_shared_database_policy(conn)["compliant"]:
        raise ControllerHOLD("controller_database_policy_invalid")
    return _path(paths[0], directory=False)


def _token(value, reason):
    if type(value) is not str or not _TOKEN.fullmatch(value):
        raise ControllerHOLD(reason)
    return value


def current_controller():
    """Return the actual strong context, including HOLD; never an inferred one."""
    ctx = _CURRENT
    # Check before taking inherited locks. Forked children must exec instead of
    # using a copied controller, even if their copied public maps are empty.
    if ctx is not None and ctx._pid != os.getpid():
        raise ControllerHOLD("controller_fork_authority_refused")
    return ctx


class ControllerContext:
    def __init__(self, lake, scope, scope_witness, db_path, db_witness, *, protocol=1):
        self._lake = lake
        self._lake_conn = lake.conn
        self._scope = scope
        self._scope_witness = scope_witness
        self._db_path = db_path
        self._db_witness = db_witness
        self._controller_id = secrets.token_hex(24)
        self._pid = os.getpid()
        self._guard = threading.RLock()
        self._phase = "PENDING"
        self._request_id = None
        self._hold_reason = None
        self._lease = None
        self._protocol = protocol
        self._generation = 0
        self._predecessor = None
        self._anchor_lease = None
        self._identity = None
        self._identity_json = None
        self._runtime_validator = None

    @property
    def scope(self):
        return self._scope

    @property
    def db_path(self):
        return self._db_path

    @property
    def controller_id(self):
        return self._controller_id

    @property
    def protocol(self):
        return self._protocol

    @property
    def generation(self):
        return self._generation

    @property
    def predecessor(self):
        return self._predecessor

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def quiescing(self):
        return self._phase == "QUIESCING"

    @contextmanager
    def guard(self):
        if self._pid != os.getpid() or _CURRENT is not self:
            raise ControllerHOLD("controller_context_not_current")
        with self._guard:
            yield

    def _paths(self):
        if (self._lake.conn is not self._lake_conn
                or _path(self._scope, directory=True) != (self._scope, self._scope_witness)
                or _path(self._db_path, directory=False) != (self._db_path, self._db_witness)
                or not idea_source_lock_owned(self._lease)):
            raise ControllerHOLD("controller_registration_ownership_changed")
        if self._protocol == 2:
            anchor = self._scope / "_controller_registration.lock"
            owner = anchor / f"instance-{self._generation}-{self._controller_id}.lock"
            if (self._anchor_lease is None or self._anchor_lease.lock_dir != anchor
                    or self._lease.lock_dir != owner
                    or not idea_source_lock_owned(self._anchor_lease)):
                raise ControllerHOLD("controller_generation_ownership_changed")

    @contextmanager
    def _connection(self, conn=None, *, write=False):
        self._paths()
        owned = conn is None
        if owned:
            conn = sqlite3.connect(self._db_path.as_uri() + ("?mode=rw" if write else "?mode=ro"),
                                   uri=True, timeout=0.25)
        try:
            if _route(conn) != (self._db_path, self._db_witness):
                raise ControllerHOLD("controller_database_route_changed")
            if owned and not write:
                conn.execute("PRAGMA query_only=ON")
            yield conn
            self._paths()
            if _route(conn) != (self._db_path, self._db_witness):
                raise ControllerHOLD("controller_database_route_changed")
        finally:
            if owned:
                conn.close()

    def _row(self, conn):
        _schema(conn, protocol=self._protocol)
        row = conn.execute("SELECT controller_id, scope, "
            "CASE WHEN typeof(identity_json)='text' AND length(CAST(identity_json AS BLOB)) "
            "<= 16384 THEN identity_json ELSE NULL END, phase, "
            "CASE WHEN request_id IS NULL OR length(CAST(request_id AS BLOB)) <= 128 "
            "THEN request_id ELSE '' END, "
            "CASE WHEN hold_reason IS NULL OR length(CAST(hold_reason AS BLOB)) <= 128 "
            "THEN hold_reason ELSE '' END FROM main.controller_instances "
            "WHERE controller_id=? COLLATE BINARY", (self._controller_id,)).fetchone()
        if (row is None or tuple(row[:3]) != (self._controller_id, str(self._scope), self._identity_json)
                or type(row[3]) is not str or row[3] not in {"ACTIVE", "QUIESCING", "HOLD"}
                or (row[4] is not None and (type(row[4]) is not str or not _TOKEN.fullmatch(row[4])))
                or (row[5] is not None and (type(row[5]) is not str or not _TOKEN.fullmatch(row[5])))
                or ((row[3] == "HOLD") != (row[5] is not None))
                or (row[3] == "QUIESCING" and row[4] is None)):
            raise ControllerHOLD("controller_registration_changed")
        if self._protocol == 2:
            extra = conn.execute("SELECT generation,predecessor FROM main.controller_instances "
                                 "WHERE controller_id=? COLLATE BINARY", (self._controller_id,)).fetchone()
            if (extra is None or tuple(extra) != (self._generation, self._predecessor)
                    or _head(conn, self._scope) != (self._controller_id, self._generation, None)):
                raise ControllerHOLD("controller_current_generation_changed")
        return tuple(row)

    def poll_control(self, conn=None):
        """Revalidate persistence; no callback, process receive, STOP, or wait."""
        with self.guard():
            if self._hold_reason is not None:
                raise ControllerHOLD(self._hold_reason)
            try:
                if self._runtime_validator is not None:
                    self._runtime_validator()
                with self._connection(conn) as actual:
                    row = self._row(actual)
                if row[3:] != (self._phase, self._request_id, None):
                    raise ControllerHOLD("controller_control_state_changed")
            except Exception as exc:
                self.hold("controller_registration_unconfirmed")
                raise ControllerHOLD(self._hold_reason) from exc
            return self._phase

    def check_admission(self, conn=None):
        with self.guard():
            phase = self.poll_control(conn)
            if phase == "QUIESCING":
                raise ControllerQuiescing("controller_quiescing")
            if phase != "ACTIVE":
                raise ControllerHOLD("controller_not_active")

    def hold(self, reason):
        """Sticky locally; failed persistence never restores permission.

        This best-effort diagnostic is not proof that the database accepted a
        HOLD. The existing member/registration intent still blocks recovery.
        Never commit a caller's open transaction, including an error path.
        """
        if self._pid != os.getpid() or _CURRENT is not self:
            raise ControllerHOLD("controller_context_not_current")
        with self._guard:
            if type(reason) is not str or not _TOKEN.fullmatch(reason):
                reason = "controller_unconfirmed"
            self._hold_reason = self._hold_reason or reason
            self._phase = "HOLD"
            try:
                with self._connection(write=True) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = self._row(conn)
                    if row[4] != self._request_id:
                        raise ControllerHOLD("controller_hold_request_changed")
                    conn.execute("UPDATE main.controller_instances SET phase='HOLD', hold_reason=? "
                        "WHERE controller_id=? COLLATE BINARY AND identity_json=? COLLATE BINARY",
                        (self._hold_reason, self._controller_id, self._identity_json))
                    conn.commit()
                    if self._row(conn)[3:] != ("HOLD", self._request_id, self._hold_reason):
                        raise ControllerHOLD("controller_hold_not_persisted")
            except Exception:
                # Local HOLD is retained even if disk, schema or lock is lost.
                pass

    def quiesce(self, request_id):
        """Freeze this instance's new work; does not acknowledge any closure."""
        request_id = _token(request_id, "controller_request_id_invalid")
        with self.guard():
            self.poll_control()
            if self._request_id is not None:
                if self._request_id != request_id:
                    self.hold("controller_request_replaced")
                    raise ControllerHOLD("controller_request_replaced")
                return
            # The expected request is pinned before any possibly committed
            # write. An unknown outcome is not retried under a different ID.
            self._request_id = request_id
            try:
                with self._connection(write=True) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = self._row(conn)
                    if row[3:] != ("ACTIVE", None, None):
                        raise ControllerHOLD("controller_quiesce_state_changed")
                    changed = conn.execute("UPDATE main.controller_instances "
                        "SET phase='QUIESCING', request_id=? WHERE controller_id=? COLLATE BINARY "
                        "AND phase='ACTIVE' AND request_id IS NULL AND identity_json=? COLLATE BINARY",
                        (request_id, self._controller_id, self._identity_json)).rowcount
                    if changed != 1 or self._row(conn)[3:] != ("QUIESCING", request_id, None):
                        raise ControllerHOLD("controller_quiesce_write_unconfirmed")
                    conn.commit()
                    if self._row(conn)[3:] != ("QUIESCING", request_id, None):
                        raise ControllerHOLD("controller_quiesce_commit_unconfirmed")
                self._phase = "QUIESCING"
                self.poll_control()
            except Exception as exc:
                self.hold("controller_quiesce_unconfirmed")
                raise ControllerHOLD(self._hold_reason) from exc


def register_controller(lake, scope, *, protocol=1, admission=None, session_registrar=None):
    """Bind one real existing Lake; unknown previous registrations are refused.

    V1 retains its stop-only ownership. V2 uses a current scope head and a
    create-only owner per generation; only a validated private handoff may
    advance it. No PID marker, age, or public result label grants registration.
    """
    if type(protocol) is not int or protocol not in (1, 2):
        raise ControllerHOLD("controller_protocol_invalid")
    if protocol == 2:
        return _register_v2(lake, scope, admission, session_registrar)
    if admission is not None or session_registrar is not None:
        raise ControllerHOLD("controller_v1_handoff_unsupported")
    from orze.idea_lake import IdeaLake

    global _CURRENT
    # Do not touch a possibly locked inherited _GLOBAL_GUARD first.
    previous = current_controller()
    if type(lake) is not IdeaLake:
        raise ControllerHOLD("controller_real_lake_required")
    try:
        actual_scope, scope_witness = _path(scope, directory=True)
        if lake.conn.in_transaction:
            raise ControllerHOLD("controller_caller_transaction_active")
        db_path, db_witness = _route(lake.conn)
        if _path(lake.db_path, directory=False) != (db_path, db_witness):
            raise ControllerHOLD("controller_lake_route_mismatch")
    except Exception as exc:
        if previous is not None:
            previous.hold("controller_registration_reentry_invalid")
        raise ControllerHOLD("controller_registration_inputs_invalid") from exc
    with _GLOBAL_GUARD:
        previous = current_controller()
        if previous is not None:
            if (previous.protocol != 1 or previous._lake is not lake or previous._lake_conn is not lake.conn
                    or previous.scope != actual_scope or previous.db_path != db_path):
                raise ControllerHOLD("controller_process_already_registered")
            previous.poll_control()
            return previous
        ctx = ControllerContext(lake, actual_scope, scope_witness, db_path, db_witness)
        _CURRENT = ctx  # retained on every post-reservation failure
        try:
            ctx._lease = _acquire(actual_scope / "_controller_registration.lock")
            if not idea_source_lock_owned(ctx._lease):
                raise ControllerHOLD("controller_scope_already_owned")
            process, _ = process_identity(os.getpid())
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
                raise ControllerHOLD("controller_boot_identity_invalid")
            host = socket.gethostname()
            if not host or len(host.encode("utf-8")) > 1024:
                raise ControllerHOLD("controller_host_identity_invalid")
            ctx._identity = {"schema": 1, "controller_id": ctx.controller_id,
                "scope": str(actual_scope), "scope_device": scope_witness[0],
                "scope_inode": scope_witness[1], "database": str(db_path),
                "database_device": db_witness[0], "database_inode": db_witness[1],
                "host": host, "boot_id": boot_id, "process": process,
                "owner_metadata_sha256": ctx._lease.metadata_sha256}
            ctx._identity_json = canonical(ctx._identity).decode("utf-8")
            if len(ctx._identity_json.encode("utf-8")) > 16384:
                raise ControllerHOLD("controller_identity_limit")
            with ctx._connection(write=True) as conn:
                conn.execute("BEGIN IMMEDIATE")
                _schema(conn, create=True, protocol=1)
                if conn.execute("SELECT 1 FROM main.controller_instances "
                    "WHERE scope=? COLLATE BINARY", (str(actual_scope),)).fetchone() is not None:
                    raise ControllerHOLD("controller_previous_instance_unresolved")
                conn.execute("INSERT INTO main.controller_instances "
                    "(controller_id, scope, identity_json, phase) VALUES (?, ?, ?, 'ACTIVE')",
                    (ctx.controller_id, str(actual_scope), ctx._identity_json))
                if ctx._row(conn)[3:] != ("ACTIVE", None, None):
                    raise ControllerHOLD("controller_registration_write_unconfirmed")
                conn.commit()
                if conn.in_transaction or ctx._row(conn)[3:] != ("ACTIVE", None, None):
                    raise ControllerHOLD("controller_registration_commit_unconfirmed")
            ctx._phase = "ACTIVE"
            ctx.poll_control()
            return ctx
        except BaseException as exc:
            ctx.hold("controller_registration_unconfirmed")
            if not isinstance(exc, Exception):
                raise
            raise ControllerHOLD("controller_registration_unconfirmed") from exc


def _validated_successor(admission, lake, scope):
    try:
        from orze.engine.controller_handoff import _validated_admission
        actual = _validated_admission(admission, lake, scope)
        if (actual is None or type(actual.target_id) is not str or not _ID.fullmatch(actual.target_id)
                or type(actual.generation) is not int or not 0 < actual.generation <= 2**63 - 1
                or type(actual.source_controller_id) is not str or not _ID.fullmatch(actual.source_controller_id)
                or actual.source_controller_id == actual.target_id
                or type(actual.grant_id) is not str or not _TOKEN.fullmatch(actual.grant_id)
                or actual.anchor_lease.lock_dir != scope / "_controller_registration.lock"
                or not idea_source_lock_owned(actual.anchor_lease)
                or canonical(actual.target_process) != canonical(process_identity(os.getpid())[0])):
            raise ControllerHOLD("controller_handoff_admission_invalid")
        return actual
    except ControllerHOLD:
        raise
    except Exception as exc:
        raise ControllerHOLD("controller_handoff_admission_unavailable") from exc


def _register_v2(lake, scope, admission, session_registrar):
    """One writer owns head/instance/session/grant; no previous owner is erased.

    A session callback may only insert its bounded binding using this caller
    transaction. It must not poll the not-yet-committed context or perform OS
    work. The session independently reads its complete committed row afterward.
    """
    from orze.idea_lake import IdeaLake
    global _CURRENT
    previous = current_controller()
    if type(lake) is not IdeaLake or session_registrar is not None and not callable(session_registrar):
        raise ControllerHOLD("controller_registration_options_invalid")
    try:
        actual_scope, scope_witness = _path(scope, directory=True)
        if lake.conn.in_transaction:
            raise ControllerHOLD("controller_caller_transaction_active")
        db_path, db_witness = _route(lake.conn)
        if _path(lake.db_path, directory=False) != (db_path, db_witness):
            raise ControllerHOLD("controller_lake_route_mismatch")
        # V1 is never upgraded online, and partial/mixed V2 metadata cannot be
        # repaired by choosing a more permissive fresh-registration path.
        exists = lake.conn.execute("SELECT 1 FROM main.sqlite_master WHERE name COLLATE NOCASE IN "
                                   "('controller_instances','controller_scope_heads')").fetchone()
        if exists:
            _schema(lake.conn, protocol=2)
    except Exception as exc:
        if previous is not None:
            previous.hold("controller_registration_reentry_invalid")
        raise ControllerHOLD("controller_registration_inputs_invalid") from exc
    with _GLOBAL_GUARD:
        previous = current_controller()
        if previous is not None:
            if (admission is not None or previous.protocol != 2 or previous._lake is not lake
                    or previous._lake_conn is not lake.conn or previous.scope != actual_scope
                    or previous.db_path != db_path):
                raise ControllerHOLD("controller_process_already_registered")
            previous.poll_control()
            return previous
        accepted = None if admission is None else _validated_successor(admission, lake, actual_scope)
        ctx = ControllerContext(lake, actual_scope, scope_witness, db_path, db_witness, protocol=2)
        if accepted is not None:
            ctx._controller_id = accepted.target_id
            ctx._generation = accepted.generation
            ctx._predecessor = accepted.source_controller_id
            ctx._anchor_lease = accepted.anchor_lease
        _CURRENT = ctx  # Strong even if namespace/SQL/response initialization fails.
        try:
            anchor = actual_scope / "_controller_registration.lock"
            if accepted is None:
                ctx._anchor_lease = _acquire(anchor)
            if not idea_source_lock_owned(ctx._anchor_lease):
                raise ControllerHOLD("controller_scope_already_owned")
            owner = anchor / f"instance-{ctx.generation}-{ctx.controller_id}.lock"
            ctx._lease = _acquire(owner)
            if not idea_source_lock_owned(ctx._lease):
                raise ControllerHOLD("controller_generation_already_owned")
            process, _ = process_identity(os.getpid())
            if accepted is not None and canonical(process) != canonical(accepted.target_process):
                raise ControllerHOLD("controller_handoff_process_changed")
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot_id):
                raise ControllerHOLD("controller_boot_identity_invalid")
            host = socket.gethostname()
            if not host or len(host.encode("utf-8")) > 1024:
                raise ControllerHOLD("controller_host_identity_invalid")
            ctx._identity = {"schema": 2, "controller_id": ctx.controller_id,
                "scope": str(actual_scope), "scope_device": scope_witness[0],
                "scope_inode": scope_witness[1], "database": str(db_path),
                "database_device": db_witness[0], "database_inode": db_witness[1],
                "host": host, "boot_id": boot_id, "process": process,
                "owner_metadata_sha256": ctx._lease.metadata_sha256,
                "generation": ctx.generation, "predecessor": ctx.predecessor,
                "owner_directory": str(owner), "anchor_directory": str(anchor),
                "anchor_device": ctx._anchor_lease.device, "anchor_inode": ctx._anchor_lease.inode,
                "anchor_metadata_sha256": ctx._anchor_lease.metadata_sha256}
            ctx._identity_json = canonical(ctx._identity).decode("utf-8")
            if len(ctx._identity_json.encode("utf-8")) > 16384:
                raise ControllerHOLD("controller_identity_limit")
            with ctx._connection(write=True) as conn:
                conn.execute("BEGIN IMMEDIATE")
                _schema(conn, create=True, protocol=2)
                if accepted is None:
                    if conn.execute("SELECT 1 FROM main.controller_instances WHERE scope=? COLLATE BINARY",
                                    (str(actual_scope),)).fetchone() is not None:
                        raise ControllerHOLD("controller_previous_instance_unresolved")
                    conn.execute("INSERT INTO main.controller_scope_heads VALUES (?,?,0,NULL)",
                                 (str(actual_scope), ctx.controller_id))
                else:
                    head = _head(conn, actual_scope)
                    source = current_instance(conn, actual_scope)
                    if (head != (ctx.predecessor, ctx.generation - 1, accepted.grant_id)
                            or source[3] != "QUIESCING" or source[5] is not None):
                        raise ControllerHOLD("controller_handoff_source_changed")
                    count = conn.execute("UPDATE main.controller_scope_heads SET current_id=?,generation=?,"
                        "pending_grant=NULL WHERE scope=? COLLATE BINARY AND current_id=? COLLATE BINARY "
                        "AND generation=? AND pending_grant=? COLLATE BINARY",
                        (ctx.controller_id, ctx.generation, str(actual_scope), ctx.predecessor,
                         ctx.generation - 1, accepted.grant_id)).rowcount
                    if count != 1:
                        raise ControllerHOLD("controller_handoff_head_unconfirmed")
                conn.execute("INSERT INTO main.controller_instances "
                    "(controller_id,scope,generation,predecessor,identity_json,phase) VALUES (?,?,?,?,?,'ACTIVE')",
                    (ctx.controller_id, str(actual_scope), ctx.generation, ctx.predecessor, ctx._identity_json))
                if session_registrar is not None:
                    session_registrar(conn, ctx)
                if accepted is not None:
                    accepted.consume(conn, ctx)
                if ctx._row(conn)[3:] != ("ACTIVE", None, None):
                    raise ControllerHOLD("controller_registration_write_unconfirmed")
                conn.commit()
                if conn.in_transaction:
                    raise ControllerHOLD("controller_registration_commit_unconfirmed")
            with ctx._connection() as check:
                if ctx._row(check)[3:] != ("ACTIVE", None, None):
                    raise ControllerHOLD("controller_registration_commit_unconfirmed")
                if accepted is not None:
                    accepted.verify_consumed(check, ctx)
            ctx._phase = "ACTIVE"
            ctx.poll_control()
            return ctx
        except BaseException as exc:
            ctx.hold("controller_registration_unconfirmed")
            if not isinstance(exc, Exception):
                raise
            raise ControllerHOLD("controller_registration_unconfirmed") from exc
