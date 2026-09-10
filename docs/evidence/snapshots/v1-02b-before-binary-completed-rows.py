def _authoritative_completed_rows(
    db_path: Path,
    *,
    include_family: bool,
) -> tuple[list[tuple], str]:
    """Read agreed lifecycle-complete rows under the shared DB policy."""
    connection, reason = _open_authoritative_lifecycle(db_path)
    if connection is None:
        return [], reason
    try:
        try:
            connection.execute("BEGIN")
            schema = validate_lifecycle_schema(connection)
            select = (
                "i.idea_id, i.approach_family"
                if include_family else "i.idea_id"
            )
            rows = connection.execute(
                f"SELECT {select} FROM ideas AS i "
                "JOIN idea_state AS s ON s.idea_id = i.idea_id "
                "WHERE lower(i.status) = 'completed' "
                "AND s.current_state = 'COMPLETE' AND "
                + completed_stage_sql(schema, idea_alias="i")
            ).fetchall()
        except (sqlite3.Error, ValueError, TypeError):
            return [], "authoritative_lifecycle_database_invalid"
    finally:
        connection.close()
    return rows, "authoritative_lifecycle_loaded"
