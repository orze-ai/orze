"""Regression: validate_channels must catch unresolved ${VAR} placeholders
and missing required fields without doing any network I/O.

Companion to a5eb216 (the input-side fix). validate_channels covers the
output side: even if some future code path re-introduces a placeholder
leak (or an operator rotates a secret to an empty value), this surfaces
within seconds of the next hot-reload, instead of waiting for the next
hourly heartbeat to 404 silently. Boot canary stays as the strict
end-to-end gate; this is the cheap recurring probe.
"""
import unittest

from orze.reporting.notifications import validate_channels


class ValidateChannelsTest(unittest.TestCase):
    def test_resolved_telegram_passes(self):
        cfg = {"notifications": {"enabled": True, "channels": [
            {"type": "telegram", "bot_token": "real-token",
             "chat_id": "12345"},
        ]}}
        report = validate_channels(cfg)
        self.assertEqual(len(report), 1)
        entry = next(iter(report.values()))
        self.assertTrue(entry["delivered"])
        self.assertIsNone(entry["last_error"])

    def test_unresolved_placeholder_telegram_fails(self):
        cfg = {"notifications": {"enabled": True, "channels": [
            {"type": "telegram", "bot_token": "${TELEGRAM_BOT_TOKEN}",
             "chat_id": "12345"},
        ]}}
        report = validate_channels(cfg)
        entry = next(iter(report.values()))
        self.assertFalse(entry["delivered"])
        self.assertIn("unresolved", (entry["last_error"] or "").lower())
        self.assertIn("bot_token", entry["last_error"])

    def test_missing_chat_id_telegram_fails(self):
        cfg = {"notifications": {"enabled": True, "channels": [
            {"type": "telegram", "bot_token": "real-token"},
        ]}}
        report = validate_channels(cfg)
        entry = next(iter(report.values()))
        self.assertFalse(entry["delivered"])
        self.assertIn("missing", (entry["last_error"] or "").lower())
        self.assertIn("chat_id", entry["last_error"])

    def test_unresolved_webhook_url_fails(self):
        cfg = {"notifications": {"enabled": True, "channels": [
            {"type": "webhook", "url": "https://example.com/${HOOK_KEY}"},
        ]}}
        report = validate_channels(cfg)
        entry = next(iter(report.values()))
        self.assertFalse(entry["delivered"])
        self.assertIn("unresolved", (entry["last_error"] or "").lower())

    def test_disabled_notifications_returns_empty(self):
        cfg = {"notifications": {"enabled": False, "channels": [
            {"type": "telegram", "bot_token": "${X}", "chat_id": "${Y}"},
        ]}}
        self.assertEqual(validate_channels(cfg), {})

    def test_no_channels_returns_empty(self):
        cfg = {"notifications": {"enabled": True}}
        self.assertEqual(validate_channels(cfg), {})

    def test_unknown_type_fails(self):
        cfg = {"notifications": {"enabled": True, "channels": [
            {"type": "smoke-signal", "url": "https://x"},
        ]}}
        report = validate_channels(cfg)
        entry = next(iter(report.values()))
        self.assertFalse(entry["delivered"])
        self.assertIn("unknown", (entry["last_error"] or "").lower())

    def test_never_raises_on_garbage_cfg(self):
        # No notifications key at all.
        self.assertEqual(validate_channels({}), {})
        # None channels list.
        self.assertEqual(
            validate_channels({"notifications":
                              {"enabled": True, "channels": None}}), {})


if __name__ == "__main__":
    unittest.main()
