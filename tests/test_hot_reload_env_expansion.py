"""Regression: hot-reload must expand ${VAR} placeholders.

Without this, the on-disk yaml ``${TELEGRAM_BOT_TOKEN}`` survives
``_hot_reload_config`` and overwrites the previously-expanded
in-memory value. The boot canary still passes (it ran against the
initial load) but every subsequent notify() call 404s, and the gap
is silent — the only signal is recurring HTTP 404 lines mentioning
the literal placeholder string. This was observed in production:
``Startup canary delivered to telegram[0]`` at 20:11:12 was followed
by ``Config hot-reloaded: notifications`` at 20:11:14, then hourly
``Notification delivery failed (...bot${TELEGRAM_BOT_TOKEN}/...)``
errors for the rest of the session.

The fix runs the same ``_expand_env_vars`` over the freshly-parsed
yaml that ``load_config`` runs at boot, then rescans for unresolved
placeholders so a partial-rotation gets surfaced (matches the
behavior of the initial-load path).
"""

import os
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class HotReloadEnvExpansionTest(unittest.TestCase):
    def setUp(self):
        # Sandbox env vars to avoid leakage across test runs.
        self._old_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self._old_chat = os.environ.get("TELEGRAM_CHAT_ID")
        os.environ["TELEGRAM_BOT_TOKEN"] = "real-token-12345"
        os.environ["TELEGRAM_CHAT_ID"] = "9999"

    def tearDown(self):
        for k, v in (("TELEGRAM_BOT_TOKEN", self._old_token),
                     ("TELEGRAM_CHAT_ID", self._old_chat)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_hot_reload_expands_env_vars_in_notifications(self):
        from orze.engine.orchestrator import Orze

        cfg_yaml = textwrap.dedent("""
            notifications:
              enabled: true
              channels:
                - type: telegram
                  bot_token: "${TELEGRAM_BOT_TOKEN}"
                  chat_id: "${TELEGRAM_CHAT_ID}"
        """)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "orze.yaml"
            cfg_path.write_text(cfg_yaml)

            self_ = MagicMock(spec=Orze)
            self_.cfg = {
                "_config_path": str(cfg_path),
                # Simulate that the initial load already expanded
                # things to a real value so the reload has something
                # to *change* (otherwise the equality short-circuit
                # skips the reload silently).
                "notifications": {"enabled": True, "channels": [
                    {"type": "telegram", "bot_token": "old-token",
                     "chat_id": "old-chat"},
                ]},
            }
            self_._HOT_RELOAD_KEYS = Orze._HOT_RELOAD_KEYS

            Orze._hot_reload_config(self_)

            ch = self_.cfg["notifications"]["channels"][0]
            self.assertEqual(
                ch["bot_token"], "real-token-12345",
                "hot-reload must expand ${TELEGRAM_BOT_TOKEN}; got literal")
            self.assertEqual(
                ch["chat_id"], "9999",
                "hot-reload must expand ${TELEGRAM_CHAT_ID}; got literal")


if __name__ == "__main__":
    unittest.main()
