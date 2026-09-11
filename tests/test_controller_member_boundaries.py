"""Independent native-action settlement regression for the controller draft.

This uses a real native AttemptRef/effect transaction and a real CPU supervisor,
not an Orze loop, training workload or controller ACK. Only the *later member*
SETTLED transaction's commit is replaced by actual rollback. The preceding
native terminal/effect receipt remains genuinely committed and confirmed.
"""
from test_controller_registration_boundaries import project, _run_owned


def test_native_settlement_rollback_cannot_return_success_with_memory_settled(project):
    result = _run_owned(project, r'''
from dataclasses import asdict
import subprocess
from orze.engine import controller_members as members
from orze.engine import execution_authority as authority
from orze.engine import supervised_process as primitive
from orze.engine.attempt_effect_receipts import require_closed_effects
from orze.core.execution_attempts import create_attempt, mark_running, finish_attempt, require_current

lake = IdeaLake(base / "lake.db")
ctx = cc.register_controller(lake, scope)
folder = scope / "idea-native-member"
folder.mkdir()
process = None
owned_descriptors = []
try:
    with authority.execution_transaction(lake, folder) as tx:
        ref = create_attempt(tx.conn, folder.name, "training", "native-member-attempt", {})
        tx.watch_attempt(ref)
    process = primitive.prepare_supervised(
        [sys.executable, "-c", "pass"],
        identity={"attempt_ref": asdict(ref), "scope": str(folder)},
        cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owned_descriptors.extend([os.pidfd_open(process.supervisor_pid), os.pidfd_open(process.pid)])
    with authority.execution_transaction(lake, folder) as tx:
        mark_running(tx.conn, ref, {"process_supervision": process.binding})
        tx.watch_attempt(ref)
    process.start()
    assert process.wait(timeout=5) == 0
    def payloads():
        return [json.loads(row[0]) for row in lake.conn.execute(
            "SELECT payload_json FROM main.controller_members")]
    assert len(payloads()) == 1
    assert payloads()[0]["os_state"] == "CLOSED"
    assert payloads()[0]["action_state"] == "PENDING"
    assert payloads()[0]["closure"] == process.closure_receipt()

    original_connect = sqlite3.connect
    rollbacks = []
    class MemberSettlementRollback(sqlite3.Connection):
        def commit(self):
            rows = self.execute("SELECT payload_json FROM main.controller_members").fetchall()
            if any(json.loads(row[0])["action_state"] == "SETTLED" for row in rows):
                rollbacks.append("native_member_settlement_rolled_back")
                self.rollback()
                return
            super().commit()
    def connection(*args, **kwargs):
        return original_connect(*args, factory=MemberSettlementRollback, **kwargs)
    members.sqlite3.connect = connection
    rejected = None
    try:
        with authority.execution_transaction(lake, folder) as tx:
            digest = tx.prepare(ref, {"operation": "native_member_settlement_test"})
            assert finish_attempt(tx.conn, ref, {"effect_receipt_sha256": digest}) == "committed"
            assert payloads()[0]["action_state"] == "PENDING"
    except cc.ControllerHOLD as exc:
        rejected = str(exc)
    finally:
        members.sqlite3.connect = original_connect

    assert rollbacks == ["native_member_settlement_rolled_back"]
    assert require_current(lake.conn, ref, states=("TERMINAL",))["state"] == "TERMINAL"
    require_closed_effects(folder)
    durable = payloads()
    assert durable[0]["action_state"] != "SETTLED"
    assert rejected is not None, {
        "native_terminal": "TERMINAL",
        "member_database": durable[0]["action_state"],
        "member_memory": [m.payload["action_state"] for m in members._OWNERS.values()],
        "publication_returned_success": True}
    refused(ctx.check_admission)
    assert cc.current_controller() is ctx
    print(json.dumps({"native_terminal": "TERMINAL", "member_settlement_rejected": True}))
finally:
    # The CPU worker has no descendants; all exact identities were captured at
    # real blocked READY. Cleanup never grants missing controller settlement.
    for descriptor in owned_descriptors:
        if not select.select([descriptor], [], [], 0)[0]:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        assert select.select([descriptor], [], [], 5)[0] == [descriptor]
        os.close(descriptor)
    if process is not None:
        process._supervisor.wait(timeout=5)
        process._close_descriptors()
    lake.close()
''')
    assert result == {"native_terminal": "TERMINAL", "member_settlement_rejected": True}
