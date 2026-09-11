"""S2 CPU claim-reader compatibility, architecture and real routing.

Compatibility cases exercise the old public native launch and real transaction;
their expected result is already green before consolidation. Architecture
cases are new import-boundary requirements, not historical execution defects.
PublicSnapshotRouting requires the new API and is not a baseline red group.
All process/SQLite/qualification behavior is real. Only private claim files or
transparent observation seams are changed, with existing owned CPU cleanup.
"""
import ast
from contextlib import closing, contextmanager
import hashlib
from pathlib import Path
import sys

import pytest

from cpu_terminal_recovery_helpers import snapshot
from orze.core import cpu_action_budget as budget
from orze.core.execution_attempts import current_attempt
from orze.engine import claim_authority
from orze.engine import native_cpu_action as native
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.idea_lake import IdeaLake
from test_native_cpu_action import context, _finish

pytest_plugins = ["cpu_terminal_recovery_helpers"]


def _observe_public_reader(monkeypatch):
    """Delegate every read and retain only observation metadata, not authority."""
    actual = claim_authority.read_claim_snapshot
    calls = []

    def observe(path, *, limit=65536, required=False):
        caller = sys._getframe(1).f_globals.get("__name__")
        result = actual(path, limit=limit, required=required)
        calls.append({"caller": caller, "path": str(Path(path)),
                      "limit": limit, "required": required,
                      "sha256": None if result is None else result[1]})
        return result

    monkeypatch.setattr(claim_authority, "read_claim_snapshot", observe)
    return calls


def _assert_cpu_reads(calls, module, claim_path):
    selected = [call for call in calls if call["caller"] == module.__name__]
    assert selected
    assert all(call["path"] == str(claim_path) and call["limit"] == 8192
               and call["required"] is True for call in selected)
    digest = hashlib.sha256(claim_path.read_bytes()).hexdigest()
    assert all(call["sha256"] == digest for call in selected)


class TestCPUCompatibility:
    @pytest.mark.parametrize("fault", ["malformed", "missing"])
    def test_second_scope_claim_rejection_rolls_back_without_worker_or_effect_lock(
        self, context, monkeypatch, fault,
    ):
        lake, results, scope, cfg, create, handles = context
        action, permit = create("raise AssertionError('must never start')")
        folder = results / "idea-cpu"
        claim_path = folder / "claim.json"
        before = tuple(lake.conn.iterdump())
        actual_transaction = native.execution_transaction
        entered, statements = [], []

        @contextmanager
        def change_claim_after_real_transaction_enter(*args, **kwargs):
            with actual_transaction(*args, **kwargs) as tx:
                assert tx.conn is lake.conn and tx.conn.in_transaction
                assert not tx.prepare_started
                assert (folder / "_attempt_effect.lock").is_dir()
                entered.append(True)
                if fault == "malformed":
                    claim_path.write_bytes(b'{"unfinished":')
                else:
                    claim_path.unlink()
                yield tx

        monkeypatch.setattr(native, "execution_transaction", change_claim_after_real_transaction_enter)
        lake.conn.set_trace_callback(statements.append)
        try:
            with pytest.raises(native.CPUActionHOLD) as caught:
                native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                              permit=permit, admission=lambda: None)
        finally:
            lake.conn.set_trace_callback(None)

        expected_cause = AttemptEffectBusy if fault == "malformed" else FileNotFoundError
        assert isinstance(caught.value.__cause__, expected_cause)
        assert entered == [True]
        assert "ROLLBACK" in [statement.strip().upper() for statement in statements]
        assert not lake.conn.in_transaction
        assert caught.value.cpu_action_handle is None and handles == []
        assert current_attempt(lake.conn, "idea-cpu", "action") is None
        assert tuple(lake.conn.iterdump()) == before
        assert not (folder / "_attempt_effect.lock").exists()
        assert not (folder / "_action_attempts").exists()
        assert budget.snapshot(lake, scope)["free_slots"] == 0


class TestArchitecture:
    @pytest.mark.parametrize("module", [native, budget], ids=["native-launch", "terminal-recovery"])
    def test_cpu_module_has_no_private_training_reader_dependency(self, module):
        tree = ast.parse(Path(module.__file__).read_bytes())
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "orze.engine.training_attempts"
                or node.module == "orze.engine"
                and any(alias.name == "training_attempts" for alias in node.names)
            ):
                forbidden.append(node.lineno)
            elif isinstance(node, ast.Import) and any(
                alias.name == "orze.engine.training_attempts" for alias in node.names
            ):
                forbidden.append(node.lineno)
        assert forbidden == [], f"CPU consumer imports private training reader at {forbidden}"


class TestPublicSnapshotRouting:
    def test_real_native_launch_and_finish_use_required_8k_public_snapshot(self, context, monkeypatch):
        lake, results, scope, cfg, create, handles = context
        action, permit = create("pass")
        calls = _observe_public_reader(monkeypatch)
        claim_path = results / "idea-cpu" / "claim.json"
        handle = native.launch("idea-cpu", results, cfg, lake=lake, action=action,
                               permit=permit, admission=lambda: None)
        _assert_cpu_reads(calls, native, claim_path)
        launch_end = len(calls)
        terminal = _finish(handle, results, cfg, lake, permit)
        _assert_cpu_reads(calls[launch_end:], native, claim_path)
        assert len(handles) == 1 and handle.process is handles[0]
        assert terminal["outcome"] == "completed" and terminal["return_code"] == 0
        assert terminal["process_tree"]["event"] == "TREE_CLOSED"
        assert budget.snapshot(lake, scope)["active_reservations"] == 0
        assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2

    def test_real_confirmed_terminal_recovery_uses_required_8k_public_snapshot(
        self, make_crashed_project, monkeypatch,
    ):
        # Existing helper executes the actual CLI/native worker and exits 86
        # only after confirmed terminal/effect, before budget settlement.
        project = make_crashed_project()
        captured = project["calls"][0]["metadata"][1]
        assert captured["effect_confirmed"] is True and captured["effect_guard_absent"] is True
        before = snapshot(project, "s2-public-reader-before-parent-reconcile")
        results = project["root"] / "results"
        with closing(IdeaLake(project["root"] / "lake.db")) as lake:
            scope = budget.initialize(lake, results, project["cfg"]["execution"])
            reservation_id = captured["permit"]["reservation_id"]
            assert lake.conn.execute("SELECT state FROM cpu_action_reservations WHERE reservation_id=?",
                                     (reservation_id,)).fetchone()[0] == "BOUND"
            calls = _observe_public_reader(monkeypatch)
            outcome = budget.reconcile_confirmed_terminals(lake, scope)
            _assert_cpu_reads(calls, budget, results / "idea-first" / "claim.json")
            assert outcome["settled"] == [reservation_id] and outcome["retained"] == []
            assert budget.snapshot(lake, scope)["free_slots"] == 1
            assert budget.snapshot(lake, scope)["reserved_wall_seconds"] == 2
            assert lake.conn.execute("SELECT state FROM cpu_action_reservations WHERE reservation_id=?",
                                     (reservation_id,)).fetchone()[0] == "SETTLED"
        after = snapshot(project, "s2-public-reader-after-parent-reconcile")
        for table in ("execution_attempts", "research_artifacts", "research_observations",
                      "ideas", "idea_state", "idea_transitions", "idea_stage_state", "idea_stage_transitions"):
            assert after["database"][table] == before["database"][table]
        assert after["files"] == before["files"]
        assert after["worker_events"] == before["worker_events"]
