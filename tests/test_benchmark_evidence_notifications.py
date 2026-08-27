from orze.reporting.leaderboard import format_report_text
from orze.reporting.notifications import (
    _format_discord,
    _format_slack,
    _format_telegram,
)


def _data():
    return {
        "idea_id": "idea-1",
        "title": "candidate",
        "metric_name": "WER",
        "metric_value": 4.2,
        "rank": 1,
        "evidence_scope": "local_reproduction",
        "selection_mode": "adaptive",
        "leaderboard": [{"id": "idea-1", "title": "candidate", "value": 4.2}],
    }


def test_adaptive_evidence_context_is_visible_in_text_and_notifications():
    expected = "[local_reproduction/adaptive]"
    data = _data()

    assert expected in format_report_text(data)
    assert expected in _format_slack("completed", data)["text"]
    assert expected in _format_discord("completed", data)["content"]
    _, telegram = _format_telegram(
        "completed", data, {"bot_token": "token", "chat_id": "chat"},
    )
    assert expected in telegram["text"]


def test_new_best_notification_keeps_adaptive_evidence_context():
    expected = "[local_reproduction/adaptive]"
    data = {
        **_data(),
        "prev_best_id": "idea-0",
        "prev_best_val": 4.5,
    }

    assert expected in _format_slack("new_best", data)["text"]
    assert expected in _format_discord("new_best", data)["content"]
    _, telegram = _format_telegram(
        "new_best", data, {"bot_token": "token", "chat_id": "chat"},
    )
    assert expected in telegram["text"]
