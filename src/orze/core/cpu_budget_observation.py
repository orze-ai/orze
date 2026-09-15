"""Read-only CPU accounting for research context, never admission authority.

CALLING SPEC:
    observe_cpu_budget(lake, results_dir, declaration) -> detached JSON mapping
        Uses the same canonical scope and complete ledger audit as the engine.
        No schema/scope creation, reservation, decision, recovery or settlement.
        Missing initialization is explicit unknown; partial/corrupt namespaces
        raise CpuBudgetHOLD. Caller must bind this observation to its own scan
        revision if combining it with other queries or artifact reads.

The declaration must already come from the project's CPU execution validator.
Wall time is the cumulative reserved envelope, not measured CPU utilization.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3

from orze.core import cpu_action_budget as budget


def _capture_scope(lake, results_dir, declaration):
    database, db_identity = budget._route(lake)
    scope_path, directory_identity = budget._path(results_dir, directory=True)
    scope = {"schema": 1, "results_dir": scope_path, "database": database,
             "database_identity": db_identity, "directory_identity": directory_identity,
             "declaration": budget._declaration(declaration)}
    scope["policy_sha256"] = hashlib.sha256(budget._json(scope).encode()).hexdigest()
    budget._wall_limit_ns(scope)
    return scope


def local_cpu_budget_hold(lake, results_dir, declaration):
    """Recheck the process-local refusal latch without repeating a ledger audit."""
    return budget._key(_capture_scope(lake, results_dir, declaration)) in budget._HELD


def observe_cpu_budget(lake, results_dir, declaration):
    scope = _capture_scope(lake, results_dir, declaration)
    database, scope_path = scope["database"], scope["results_dir"]
    common = {"schema": 1, "scope": scope_path, "policy_sha256": scope["policy_sha256"],
              "declaration": scope["declaration"], "authority": "accounting_only_no_execution_rights",
              "wall_accounting": "cumulative_reserved_envelopes_not_cpu_utilization"}
    initialized = False
    with closing(sqlite3.connect(Path(database).as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
        conn.execute("BEGIN")
        try:
            names = [*budget._SQL, *budget._INDEX]
            marks = ",".join("?" for _ in names)
            present = conn.execute(
                f"SELECT 1 FROM main.sqlite_master WHERE name COLLATE NOCASE IN ({marks}) LIMIT 1",
                names,
            ).fetchone()
            if present is not None:
                budget._schema(conn)
                initialized = conn.execute(
                    "SELECT 1 FROM main.cpu_action_scopes WHERE scope=?", (scope_path,),
                ).fetchone() is not None
                if not initialized and conn.execute(
                    "SELECT 1 FROM main.cpu_action_reservations WHERE scope=? LIMIT 1", (scope_path,),
                ).fetchone() is not None:
                    raise budget.CpuBudgetHOLD("cpu_budget_observation_orphan_reservations")
            budget._check_route(lake, scope)
        finally:
            conn.rollback()
    if not initialized:
        return {**common, "availability": "unavailable", "reason": "cpu_budget_not_initialized"}
    return {**common, "availability": "verified", "accounting": budget.snapshot(lake, scope)}
