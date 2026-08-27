from orze.admin.server import _present_research_efficiency


def _full(score=42.0):
    return {
        "metric": {"name": "avg_wer", "lower_is_better": True},
        "stats": {
            "n_total": 10,
            "n_scored": 2,
            "genuine_evolution_rate": 0.25,
        },
        "research_efficiency": {
            "score": score,
            "grade": "D",
            "evidence_qualification": {
                "mode": "verified_local_artifact",
                "primary_metric": "avg_wer",
                "fallback_metrics_allowed": False,
                "accepted": 2,
                "rejected": {"local_metrics_missing": 1},
            },
            "presentation": {
                "claim_scope": "internal_research_efficiency",
                "qualification_applied": True,
                "evidence_label": "verified local artifacts",
                "leaderboard_rank_comparable": False,
            },
        },
    }


def test_api_preserves_only_qualified_internal_score():
    out = _present_research_efficiency(_full())
    assert out["score"] == 42.0
    assert out["n_scored"] == 2
    assert out["presentation"]["leaderboard_rank_comparable"] is False


def test_api_suppresses_bare_or_malformed_scores():
    unsafe_presentation = _full()
    unsafe_presentation["research_efficiency"]["presentation"][
        "leaderboard_rank_comparable"] = True
    for full in (
        {"research_efficiency": {"score": 99.9, "grade": "A"}},
        _full(True),
        _full(float("nan")),
        unsafe_presentation,
    ):
        out = _present_research_efficiency(full)
        assert out["score"] is None
        assert out["grade"] == "—"
        assert out["error"] == "evidence qualification unavailable"
        assert out["presentation"]["qualification_applied"] is False
