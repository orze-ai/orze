"""Shared recorded-stage agreement for completion observers.

CALLING SPEC:
    validate_lifecycle_schema(conn) -> mapping of table name to columns/None
        Read-only schema and identity checks. Caller owns one read or write
        transaction spanning validation and use; this module never repairs DBs.
    stage_projection(schema) -> (SQL projection, SQL joins)
        Four fields: training presence ID/value, evaluation presence ID/value.
        Uses the fixed ideas alias ``i``. Missing history is represented by NULL.
    completed_stages_agree(training_id, training, evaluation_id, evaluation)
        Missing rows are historical compatibility, not recorded success.
    completed_stage_sql(schema, *, idea_alias='ideas') -> SQL predicate
        Same completed-stage rule for a SELECT or atomic conditional UPDATE.

Only the framework's recorded training/evaluation stages constrain completion.
Other stages do not gain invented semantics. This is lifecycle agreement, not
metric validity, statistical evidence or authority to execute/repair a task.
"""
from __future__ import annotations


_TABLES = {
    "ideas": ({"idea_id", "status"}, ("idea_id",)),
    "idea_state": ({"idea_id", "current_state"}, ("idea_id",)),
    "idea_stage_state": (
        {"idea_id", "stage", "current_state"}, ("idea_id", "stage")),
}


def validate_lifecycle_schema(connection) -> dict:
    """Reject malformed tables/ambiguous identities, including non-candidates.

    Historical tables without a primary key remain readable if their actual
    identities are unique. Current exact identity PKs avoid a duplicate scan;
    nullable SQLite primary keys still need the NULL-identity check. Every SQL
    identifier below is a fixed internal literal, not configuration input.
    """
    schema = {}
    for name, (required, keys) in _TABLES.items():
        kind = connection.execute(
            "SELECT type FROM sqlite_master WHERE name=?", (name,),
        ).fetchone()
        if kind is None and name == "idea_stage_state":
            schema[name] = None
            continue
        if kind is None or kind[0] != "table":
            raise ValueError("lifecycle_schema_invalid")
        info = connection.execute(f"PRAGMA table_info({name})").fetchall()
        columns = {column[1] for column in info}
        if not required.issubset(columns):
            raise ValueError("lifecycle_schema_invalid")
        null_keys = " OR ".join(f"{key} IS NULL" for key in keys)
        if connection.execute(
            f"SELECT 1 FROM {name} WHERE {null_keys} LIMIT 1",
        ).fetchone() is not None:
            raise ValueError("lifecycle_identity_invalid")
        primary = tuple(column[1] for column in sorted(
            (column for column in info if column[5]), key=lambda row: row[5]))
        if set(primary) != set(keys):
            grouped = ", ".join(keys)
            if connection.execute(
                f"SELECT 1 FROM {name} GROUP BY {grouped} "
                "HAVING COUNT(*) > 1 LIMIT 1",
            ).fetchone() is not None:
                raise ValueError("lifecycle_identity_ambiguous")
        schema[name] = columns
    return schema


def stage_projection(schema) -> tuple[str, str]:
    if schema["idea_stage_state"] is None:
        return "NULL, NULL, NULL, NULL", ""
    return (
        "t.idea_id, t.current_state, e.idea_id, e.current_state",
        " LEFT JOIN idea_stage_state t ON t.idea_id=i.idea_id AND t.stage='training'"
        " LEFT JOIN idea_stage_state e ON e.idea_id=i.idea_id AND e.stage='evaluation'",
    )


def completed_stages_agree(training_id, training, evaluation_id, evaluation) -> bool:
    return ((training_id is None or training == "COMPLETE")
            and (evaluation_id is None or evaluation in ("COMPLETE", "SKIPPED")))


def completed_stage_sql(schema, *, idea_alias="ideas") -> str:
    if idea_alias not in ("i", "ideas"):
        raise ValueError("lifecycle_sql_alias_invalid")
    if schema["idea_stage_state"] is None:
        return "1"
    return (
        "NOT EXISTS (SELECT 1 FROM idea_stage_state AS stage_check "
        f"WHERE stage_check.idea_id={idea_alias}.idea_id AND ("
        "(stage_check.stage='training' AND (stage_check.current_state IS NULL "
        "OR stage_check.current_state <> 'COMPLETE')) OR "
        "(stage_check.stage='evaluation' AND (stage_check.current_state IS NULL "
        "OR stage_check.current_state NOT IN ('COMPLETE', 'SKIPPED')))))"
    )
