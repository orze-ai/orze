def admit_proposal(lake, prepared):
    """Atomically check-and-create one queued task without replacing a winner."""
    connection = lake.conn
    if connection.in_transaction:
        return _result(prepared, "rejected", "proposal_caller_transaction")
    if str(prepared["status"]).lower() != "queued":
        return _result(prepared, "rejected", "proposal_requires_queued")
    if (not isinstance(prepared["idea_id"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", prepared["idea_id"]) is None):
        return _result(prepared, "rejected", "proposal_id_invalid")
    if (prepared["config_hash"] is None
            or not isinstance(prepared["config"], str)
            or len(prepared["config"].encode("utf-8")) > _MAX_CONFIG_BYTES):
        return _result(prepared, "rejected", "proposal_config_invalid")
    # The public entry has already prepared the complete metadata; reject
    # non-text identity fields instead of letting SQLite silently coerce them.
    if any(value is not None and not isinstance(value, str)
           for value in (prepared[field] for field in _SOURCE_FIELDS)):
        return _result(prepared, "rejected", "proposal_source_invalid")

    timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    owns_transaction = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        owns_transaction = True
        existing = connection.execute(
            f"SELECT {', '.join(_SOURCE_FIELDS)} FROM ideas WHERE idea_id=?",
            (prepared["idea_id"],),
        ).fetchall()
        if existing:
            exact = (len(existing) == 1 and all(
                existing[0][field] == prepared[field] for field in _SOURCE_FIELDS))
            connection.rollback()
            return _result(prepared, "already_present_exact" if exact else "conflict",
                           "proposal_exact_replay" if exact else "proposal_identity_conflict")
        for table in ("idea_state", "idea_stage_state", "idea_transitions", "idea_stage_transitions"):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE idea_id=? LIMIT 1", (prepared["idea_id"],),
            ).fetchone() is not None:
                raise _Rejected("proposal_orphan_lifecycle")
        owner = _dedup_owner(connection, prepared)
        if owner is not None:
            connection.rollback()
            return _result(prepared, "config_duplicate", "proposal_config_duplicate", existing_id=owner)
        transition_id = _new_rows(lake, prepared)
        connection.commit()
        return _result(prepared, "inserted", "proposal_admitted", transition_id=transition_id)
    except _Rejected as exc:
        if owns_transaction:
            connection.rollback()
        return _result(prepared, "rejected", str(exc))
    except sqlite3.Error:
        if owns_transaction:
            connection.rollback()
        return _result(prepared, "rejected", "proposal_storage_error")
    except BaseException:
        if owns_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute(f"PRAGMA busy_timeout={int(timeout)}")
