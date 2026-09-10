    def _lifecycle_identity_exists_in_tx(self, idea_id: str) -> bool:
        rows = self.conn.execute(
            "SELECT idea_id FROM ideas WHERE idea_id = ?", (idea_id,),
        ).fetchmany(2)
        return len(rows) == 1 and rows[0][0] == idea_id

    def _record_stage_transition_in_tx(
        self,
        idea_id: str,
        stage: str,
        from_state: str,
        to_state: str,
        reason: str,
        host: str,
        pid: int,
        at: str,
    ) -> bool:
        """Compare-and-swap one pipeline stage inside the caller's transaction."""
        if (not self.conn.in_transaction or stage not in PIPELINE_STAGES
                or not self._lifecycle_identity_exists_in_tx(idea_id)):
            return False
        actual = self._stage_state_in_tx(idea_id, stage)
        if actual != from_state:
            logger.warning(
                "Stale stage transition rejected: %s stage=%s expected=%s "
                "actual=%s to=%s", idea_id, stage, from_state, actual,
                to_state,
            )
            return False
        if to_state not in VALID_STAGE_TRANSITIONS.get(from_state, set()):
            logger.warning(
                "Invalid stage transition: %s stage=%s %s -> %s",
                idea_id, stage, from_state, to_state,
            )
            return False

        previous = self.conn.execute(
            "SELECT started_at FROM idea_stage_state "
            "WHERE idea_id COLLATE BINARY = ? AND stage COLLATE BINARY = ?",
            (idea_id, stage),
        ).fetchone()
        started_at = (at if to_state == "IN_PROGRESS" else
                      None if to_state == "PENDING" or previous is None else previous[0])
        terminal_at = at if to_state in STAGE_TERMINALS else None
        if previous is None:
            cursor = self.conn.execute(
                "INSERT INTO idea_stage_state "
                "(idea_id, stage, current_state, updated_at, started_at, "
                "terminal_at) VALUES (?, ?, ?, ?, ?, ?)",
                (idea_id, stage, to_state, at, started_at, terminal_at),
            )
            if cursor.rowcount != 1:
                return False
        else:
            cursor = self.conn.execute(
                "UPDATE idea_stage_state SET current_state = ?, "
                "updated_at = ?, started_at = CASE "
                "WHEN ? = 'PENDING' THEN NULL "
                "WHEN ? = 'IN_PROGRESS' THEN ? ELSE started_at END, "
                "terminal_at = ? WHERE idea_id COLLATE BINARY = ? "
                "AND stage COLLATE BINARY = ? AND current_state COLLATE BINARY = ?",
                (
                    to_state, at, to_state, to_state, at, terminal_at,
                    idea_id, stage, from_state,
                ),
            )
            if cursor.rowcount != 1:
                return False
        cursor = self.conn.execute(
            "INSERT INTO idea_stage_transitions "
            "(idea_id, stage, from_state, to_state, reason, host, pid, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (idea_id, stage, from_state, to_state, reason, host, pid, at),
        )
        if cursor.rowcount != 1:
            return False
        audit = self.conn.execute(
            "SELECT idea_id, stage, from_state, to_state, reason, host, pid, ts "
            "FROM idea_stage_transitions WHERE id = ?", (cursor.lastrowid,),
        ).fetchmany(2)
        state = self.conn.execute(
            "SELECT idea_id, stage, current_state, updated_at, started_at, terminal_at "
            "FROM idea_stage_state WHERE idea_id = ? AND stage = ?",
            (idea_id, stage),
        ).fetchmany(2)
        return (len(audit) == 1 and tuple(audit[0]) ==
                (idea_id, stage, from_state, to_state, reason, host, pid, at)
                and len(state) == 1 and tuple(state[0]) ==
                (idea_id, stage, to_state, at, started_at, terminal_at))

    def _record_state_transition_in_tx(
        self, idea_id: str, from_state: str, to_state: str,
        reason: Optional[str] = None, host: Optional[str] = None,
        pid: Optional[int] = None, sop_type: Optional[str] = None,
        at: Optional[str] = None,
    ) -> bool:
        """Write one exact lifecycle edge in a transaction owned by the caller.

        Never begins, commits or rolls back. False may follow tentative SQL
        writes: the owning caller MUST roll back its transaction on rejection.
        Database exceptions likewise belong to the caller's rollback boundary.
        """
        if not self.conn.in_transaction:
            return False
        if to_state not in VALID_STATE_TRANSITIONS.get(from_state, set()):
            logger.warning("Invalid FSM transition: %s %s → %s", idea_id, from_state, to_state)
            return False
        if not self._lifecycle_identity_exists_in_tx(idea_id):
            return False
        rows = self.conn.execute(
            "SELECT * FROM idea_state WHERE idea_id = ?", (idea_id,),
        ).fetchmany(2)
        if len(rows) > 1 or (rows and rows[0]["idea_id"] != idea_id):
            return False
        before = dict(rows[0]) if rows else {}
        actual_state = before.get("current_state", "QUEUED")
        if actual_state != from_state:
            logger.warning(
                "Stale FSM transition rejected: %s expected=%s actual=%s to=%s",
                idea_id, from_state, actual_state, to_state)
            return False
        import socket as _socket
        host = host or _socket.gethostname()
        pid = pid or os.getpid()
        sop_type = sop_type or "training"
        reason = reason or ""
        at = self._transition_time(self.conn) if at is None else at

        # Derive exact postconditions from the existing current-attempt clock
        # policy, including fields that this edge must leave unchanged.
        clocks = ("first_queued_at", "queued_at", "claimed_at", "started_at",
                  "terminal_at", "completed_at")
        expected = {name: before.get(name) for name in clocks}
        expected.update(idea_id=idea_id, current_state=to_state,
                        updated_by_host=host, updated_by_pid=pid,
                        sop_type=sop_type, updated_at=at)
        if to_state == "QUEUED":
            expected.update(first_queued_at=(before.get("first_queued_at")
                                            if before.get("first_queued_at") is not None else at),
                            queued_at=at, claimed_at=None, started_at=None,
                            terminal_at=None, completed_at=None)
        elif to_state == "CLAIMED":
            expected.update(claimed_at=at, started_at=None, terminal_at=None, completed_at=None)
        elif to_state == "IN_PROGRESS":
            expected.update(started_at=at, terminal_at=None, completed_at=None)
        elif to_state == "COMPLETE":
            expected.update(terminal_at=at, completed_at=at)
        elif to_state in ("FAILED", "SKIPPED"):
            expected.update(terminal_at=at, completed_at=None)

        if rows:
            accepted = self._write_state_row(
                self.conn, idea_id, to_state, host, pid, sop_type, at,
                expected_state=from_state)
        else:
            accepted = self._insert_state_row(
                self.conn, idea_id, to_state, host, pid, sop_type, at)
        pipeline = {}
        if not accepted or not self._sync_pipeline_for_global_transition(
                idea_id, from_state, to_state, reason, host, pid, sop_type, at,
                receipts=pipeline):
            return False
        # Keep already captured per-write receipts, and also ensure subsequent
        # global/legacy writes cannot change an untouched stage.
        for stage in PIPELINE_STAGES:
            if stage not in pipeline:
                pipeline[stage] = self._stage_receipt_in_tx(idea_id, stage)
        cursor = self.conn.execute(
            "INSERT INTO idea_transitions "
            "(idea_id, from_state, to_state, reason, host, pid, sop_type, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (idea_id, from_state, to_state, reason, host, pid, sop_type, at),
        )
        if cursor.rowcount != 1:
            return False
        audit_id = cursor.lastrowid
        legacy_status = STATE_TO_STATUS.get(to_state)
        if legacy_status:
            cursor = self.conn.execute(
                "UPDATE ideas SET status = ? WHERE idea_id COLLATE BINARY = ?",
                (legacy_status, idea_id),
            )
            if cursor.rowcount != 1:
                return False

        # Read back after ALL writes: an AFTER trigger on the compatibility
        # update must not invalidate a state/stage already checked earlier.
        state = self.conn.execute(
            "SELECT * FROM idea_state WHERE idea_id = ?", (idea_id,),
        ).fetchmany(2)
        audit = self.conn.execute(
            "SELECT idea_id, from_state, to_state, reason, host, pid, sop_type, ts "
            "FROM idea_transitions WHERE id = ?", (audit_id,),
        ).fetchmany(2)
        legacy = self.conn.execute(
            "SELECT idea_id, status FROM ideas WHERE idea_id = ?", (idea_id,),
        ).fetchmany(2)
        return (len(state) == 1 and all(state[0][key] == value for key, value in expected.items())
                and len(audit) == 1 and tuple(audit[0]) ==
                (idea_id, from_state, to_state, reason, host, pid, sop_type, at)
                and len(legacy) == 1 and tuple(legacy[0]) == (idea_id, legacy_status)
                and all(self._stage_receipt_in_tx(idea_id, stage) == receipt
                        for stage, receipt in pipeline.items()))
