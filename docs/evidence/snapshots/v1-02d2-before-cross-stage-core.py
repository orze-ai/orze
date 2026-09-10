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
        if not accepted or not self._sync_pipeline_for_global_transition(
                idea_id, from_state, to_state, reason, host, pid, sop_type, at):
            return False

        def pipeline_receipts():
            # Two fixed stages only; never materialize unbounded audit history.
            stages, audits = [], []
            for stage in PIPELINE_STAGES:
                stages.extend(tuple(row) for row in self.conn.execute(
                    "SELECT * FROM idea_stage_state WHERE idea_id COLLATE BINARY = ? "
                    "AND stage COLLATE BINARY = ?", (idea_id, stage),
                ).fetchmany(2))
                audits.extend(tuple(row) for row in self.conn.execute(
                    "SELECT * FROM idea_stage_transitions WHERE idea_id COLLATE BINARY = ? "
                    "AND stage COLLATE BINARY = ? ORDER BY id DESC LIMIT 1", (idea_id, stage),
                ).fetchall())
            return stages, audits

        pipeline = pipeline_receipts()
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
                and pipeline_receipts() == pipeline)
