import sqlite3

from orze.idea_lake import IdeaLake


_CLOCKS = (
    "queued_at", "claimed_at", "started_at", "terminal_at", "completed_at",
)


def _state(lake: IdeaLake, idea_id: str) -> dict:
    row = lake.conn.execute(
        "SELECT updated_at, queued_at, claimed_at, started_at, terminal_at, "
        "completed_at FROM idea_state WHERE idea_id = ?",
        (idea_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


def test_accepted_edges_atomically_stamp_state_and_audit(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea_id = "idea-clock-success"
    lake.insert(idea_id, "clock", "{}", "", status="queued")

    admitted = _state(lake, idea_id)
    assert admitted["queued_at"] is not None
    assert all(admitted[name] is None for name in _CLOCKS[1:])

    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    claimed = _state(lake, idea_id)
    assert claimed["claimed_at"] == claimed["updated_at"]
    assert claimed["claimed_at"] == lake.get_fsm_history(idea_id)[-1]["ts"]
    assert claimed["started_at"] is None
    assert claimed["terminal_at"] is None
    assert claimed["completed_at"] is None

    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    started = _state(lake, idea_id)
    assert started["started_at"] == started["updated_at"]
    assert started["started_at"] == lake.get_fsm_history(idea_id)[-1]["ts"]

    assert lake.record_state_transition(idea_id, "IN_PROGRESS", "COMPLETE")
    complete = _state(lake, idea_id)
    assert complete["terminal_at"] == complete["completed_at"]
    assert complete["completed_at"] == complete["updated_at"]
    assert complete["completed_at"] == lake.get_fsm_history(idea_id)[-1]["ts"]
    lake.close()


def test_rejected_edge_changes_neither_clocks_nor_audit(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea_id = "idea-clock-stale"
    lake.insert(idea_id, "clock", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")

    before = _state(lake, idea_id)
    history_before = lake.get_fsm_history(idea_id)
    assert not lake.record_state_transition(idea_id, "QUEUED", "SKIPPED")
    assert _state(lake, idea_id) == before
    assert lake.get_fsm_history(idea_id) == history_before
    lake.close()


def test_accepted_edges_replace_untrusted_prefilled_clocks(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea_id = "idea-clock-prefill"
    lake.insert(idea_id, "clock", "{}", "", status="queued")
    lake.conn.execute(
        "UPDATE idea_state SET claimed_at = '1900-01-01', "
        "started_at = '1900-01-01', terminal_at = '1900-01-01', "
        "completed_at = '1900-01-01' WHERE idea_id = ?",
        (idea_id,),
    )
    lake.conn.commit()

    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    claimed = _state(lake, idea_id)
    assert claimed["claimed_at"] != "1900-01-01"
    assert claimed["started_at"] is None
    assert claimed["terminal_at"] is None
    assert claimed["completed_at"] is None

    lake.conn.execute(
        "UPDATE idea_state SET started_at = '1900-01-01', "
        "terminal_at = '1900-01-01', completed_at = '1900-01-01' "
        "WHERE idea_id = ?",
        (idea_id,),
    )
    lake.conn.commit()
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    assert _state(lake, idea_id)["started_at"] != "1900-01-01"

    assert lake.record_state_transition(idea_id, "IN_PROGRESS", "COMPLETE")
    complete = _state(lake, idea_id)
    assert complete["terminal_at"] != "1900-01-01"
    assert complete["completed_at"] != "1900-01-01"
    lake.close()


def test_requeue_resets_current_attempt_but_ledger_retains_history(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    idea_id = "idea-clock-retry"
    lake.insert(idea_id, "clock", "{}", "", status="queued")
    assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
    assert lake.record_state_transition(idea_id, "CLAIMED", "IN_PROGRESS")
    assert lake.record_state_transition(idea_id, "IN_PROGRESS", "FAILED")
    first_terminal = _state(lake, idea_id)["terminal_at"]

    assert lake.record_state_transition(idea_id, "FAILED", "QUEUED")
    requeued = _state(lake, idea_id)
    assert requeued["queued_at"] == lake.get_fsm_history(idea_id)[-1]["ts"]
    assert requeued["claimed_at"] is None
    assert requeued["started_at"] is None
    assert requeued["terminal_at"] is None
    assert requeued["completed_at"] is None
    assert first_terminal in {row["ts"] for row in lake.get_fsm_history(idea_id)}
    lake.close()


def test_legacy_and_imported_rows_are_not_given_fabricated_history(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(db_path)
    lake.insert("idea-imported", "old", "{}", "", status="completed")
    assert all(_state(lake, "idea-imported")[name] is None for name in _CLOCKS)

    lake.conn.execute(
        "INSERT INTO idea_state (idea_id, current_state, updated_at) "
        "VALUES ('idea-legacy', 'FAILED', '2020-01-01 00:00:00')"
    )
    lake.conn.commit()
    lake.close()

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        ALTER TABLE idea_state RENAME TO idea_state_with_clocks;
        CREATE TABLE idea_state (
            idea_id TEXT PRIMARY KEY,
            current_state TEXT NOT NULL DEFAULT 'QUEUED',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_by_host TEXT,
            updated_by_pid INTEGER,
            sop_type TEXT DEFAULT 'training'
        );
        INSERT INTO idea_state (
            idea_id, current_state, updated_at, updated_by_host,
            updated_by_pid, sop_type
        ) SELECT idea_id, current_state, updated_at, updated_by_host,
                 updated_by_pid, sop_type
          FROM idea_state_with_clocks;
        DROP TABLE idea_state_with_clocks;
        """
    )
    conn.close()

    lake = IdeaLake(db_path)
    assert all(_state(lake, "idea-imported")[name] is None for name in _CLOCKS)
    assert all(_state(lake, "idea-legacy")[name] is None for name in _CLOCKS)
    lake.close()


def test_original_transition_column_names_are_losslessly_normalized(tmp_path):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(db_path)
    lake.close()

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        DROP INDEX idx_idea_transitions_idea_id;
        DROP INDEX idx_idea_transitions_to_state;
        DROP INDEX idx_idea_transitions_ts;
        DROP INDEX idx_idea_transitions_sop_type;
        DROP TABLE idea_transitions;
        CREATE TABLE idea_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id TEXT NOT NULL,
            from_state TEXT NOT NULL,
            to_state TEXT NOT NULL,
            reason TEXT,
            transitioned_at TEXT NOT NULL,
            transitioned_by_host TEXT,
            transitioned_by_pid INTEGER
        );
        INSERT INTO idea_transitions (
            idea_id, from_state, to_state, reason, transitioned_at,
            transitioned_by_host, transitioned_by_pid
        ) VALUES (
            'idea-legacy-edge', 'FAILED', 'QUEUED', 'old_retry',
            '2020-01-02T03:04:05Z', 'old-host', 123
        );
        INSERT INTO idea_state (
            idea_id, current_state, updated_at, updated_by_host, updated_by_pid,
            sop_type
        ) VALUES (
            'idea-legacy-edge', 'QUEUED', '2020-01-02T03:04:05Z',
            'old-host', 123, 'training'
        );
        """
    )
    conn.close()

    lake = IdeaLake(db_path)
    columns = {
        row[1] for row in lake.conn.execute(
            "PRAGMA table_info(idea_transitions)"
        )
    }
    assert {"ts", "host", "pid", "sop_type"}.issubset(columns)
    assert not {
        "transitioned_at", "transitioned_by_host", "transitioned_by_pid"
    } & columns
    old = lake.get_fsm_history("idea-legacy-edge")[0]
    assert old["ts"] == "2020-01-02T03:04:05Z"
    assert old["host"] == "old-host"
    assert old["pid"] == 123

    assert lake.record_state_transition(
        "idea-legacy-edge", "QUEUED", "CLAIMED", host="new-host", pid=456
    )
    new = lake.get_fsm_history("idea-legacy-edge")[-1]
    assert new["ts"] == _state(lake, "idea-legacy-edge")["claimed_at"]
    assert new["host"] == "new-host"
    assert new["pid"] == 456
    assert lake.conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    lake.close()


def test_legacy_status_and_reconciliation_use_same_clock_contract(tmp_path):
    lake = IdeaLake(tmp_path / "lake.db")
    lake.insert("idea-skip", "skip", "{}", "", status="queued")
    assert lake.set_status("idea-skip", "skipped")
    skipped = _state(lake, "idea-skip")
    assert skipped["terminal_at"] == skipped["updated_at"]
    assert skipped["terminal_at"] == lake.get_fsm_history("idea-skip")[-1]["ts"]
    assert skipped["completed_at"] is None

    lake.insert("idea-reconcile", "repair", "{}", "", status="queued")
    assert lake.reconcile_terminal_state(
        "idea-reconcile", "COMPLETE", "reconcile_verified_metrics"
    )
    reconciled = _state(lake, "idea-reconcile")
    assert reconciled["completed_at"] == reconciled["updated_at"]
    assert reconciled["completed_at"] == lake.get_fsm_history(
        "idea-reconcile"
    )[-1]["ts"]
    lake.close()
