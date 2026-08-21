"""Failure semantics for durable research-procedure state."""

import tempfile
import unittest
from pathlib import Path

from orze.fsm.engine import (
    FSM,
    FSMActionError,
    FSMStateError,
    StateNode,
    Transition,
    action,
)


class TestFSMFailClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _fsm(self, action_name=None):
        actions = [action_name] if action_name else []
        return FSM(
            "research",
            {
                "queued": StateNode(
                    "queued",
                    transitions=[Transition(to="running", guards=[], actions=actions)],
                ),
                "running": StateNode("running"),
            },
            "queued",
            self.results,
        )

    def test_corrupt_state_is_not_silently_reset(self):
        fsm = self._fsm()
        fsm.state_file.write_text("{not-json", encoding="utf-8")

        with self.assertRaises(FSMStateError):
            fsm.step()

    def test_failed_action_does_not_advance_state(self):
        name = "test_failure_does_not_advance"

        @action(name)
        def fail(_ctx):
            raise ValueError("experimental action failed")

        fsm = self._fsm(name)
        with self.assertRaises(FSMActionError):
            fsm.step()

        self.assertFalse(fsm.state_file.exists())

    def test_successful_transition_is_durable(self):
        fsm = self._fsm()

        self.assertEqual(fsm.step(), "running")
        self.assertEqual(fsm.load()["current"], "running")


if __name__ == "__main__":
    unittest.main()
