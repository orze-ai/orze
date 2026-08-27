from orze.engine.orchestrator import Orze
from orze.idea_lake import IdeaLake


def test_same_batch_duplicate_is_not_admitted_while_first_is_queued(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        "# Ideas\n\n"
        "## idea-first: First proposal\n"
        "```yaml\nseed: 7\nlr: 0.001\n```\n\n"
        "## idea-replica: Deterministic replica\n"
        "```yaml\nseed: 7\nlr: 0.001\n```\n",
        encoding="utf-8",
    )
    cfg = {
        "ideas_file": str(ideas),
        "results_dir": str(results),
        "idea_lake_db": str(tmp_path / "ideas.db"),
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(results),
        "sweep": {},
    }
    orchestrator = Orze.__new__(Orze)
    orchestrator.cfg = cfg
    orchestrator.results_dir = results
    orchestrator.lake = IdeaLake(cfg["idea_lake_db"])
    orchestrator.failure_counts = {}
    orchestrator.active_roles = {}
    try:
        _, unclaimed, _, _ = orchestrator._sync_ideas(cfg)
        assert orchestrator.lake.get_all_ids() == {"idea-first"}
        assert unclaimed == ["idea-first"]
    finally:
        orchestrator.lake.close()


def test_prior_queued_duplicate_blocks_later_sync(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        "# Ideas\n\n"
        "## idea-later: Later proposal\n"
        "```yaml\nseed: 7\nlr: 0.001\n```\n",
        encoding="utf-8",
    )
    cfg = {
        "ideas_file": str(ideas),
        "results_dir": str(results),
        "idea_lake_db": str(tmp_path / "ideas.db"),
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(results),
        "sweep": {},
    }
    orchestrator = Orze.__new__(Orze)
    orchestrator.cfg = cfg
    orchestrator.results_dir = results
    orchestrator.lake = IdeaLake(cfg["idea_lake_db"])
    orchestrator.lake.insert(
        "idea-existing", "Existing", "lr: 0.001\nseed: 7\n", "",
        status="queued",
    )
    orchestrator.failure_counts = {}
    orchestrator.active_roles = {}
    try:
        orchestrator._sync_ideas(cfg)
        assert orchestrator.lake.get_all_ids() == {"idea-existing"}
    finally:
        orchestrator.lake.close()
