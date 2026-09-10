def refresh_metric_snapshot(lake, row: dict) -> None:
    """Preserve legacy archive displays without granting lifecycle authority.

    Schema, identity uniqueness and recorded stages are rechecked inside an
    owned write transaction. The atomic SQL predicate also rejects a requeue
    after qualification. An existing caller transaction is never committed or
    rolled back. This remains a mutable diagnostic mirror, not an immutable
    observation receipt or proof of an atomic filesystem/DB view.
    """
    if lake is None or not row.get("evidence_identity"):
        return
    connection = None
    owns_transaction = False
    try:
        connection = lake.conn
        if connection.in_transaction:
            return
        connection.execute("BEGIN IMMEDIATE")
        owns_transaction = True
        schema = validate_lifecycle_schema(connection)
        stage_predicate = completed_stage_sql(schema, idea_alias="ideas")
        connection.execute(
            "UPDATE ideas SET eval_metrics = ? WHERE idea_id = ? "
            "AND lower(status) = 'completed' "
            "AND EXISTS (SELECT 1 FROM idea_state AS s "
            "WHERE s.idea_id = ideas.idea_id AND s.current_state = 'COMPLETE') "
            f"AND ({stage_predicate})",
            (json.dumps(row["values"]), row["id"]),
        )
        connection.commit()
        owns_transaction = False
    except Exception as exc:
        if owns_transaction:
            try:
                connection.rollback()
            except Exception as rollback_exc:
                logger.warning("Metric snapshot rollback failed: %s",
                               type(rollback_exc).__name__)
        logger.warning("Metric snapshot not refreshed for %s: %s",
                       row.get("id"), type(exc).__name__)
