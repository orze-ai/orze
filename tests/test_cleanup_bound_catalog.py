"""A routed authority is protected even when current cfg omits its pathname."""
from orze.engine.scheduler import run_cleanup
from orze.idea_lake import IdeaLake


def test_supplied_catalog_inside_task_is_not_a_disposable_candidate(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-embedded-catalog"
    folder.mkdir(parents=True)
    database = folder / "catalog.db"
    lake = IdeaLake(database)
    try:
        lake.insert(folder.name, "closed fixture", "seed: 1", "", status="completed")
        assert lake.get_fsm_state(folder.name) == "COMPLETE"
        original = database.read_bytes()
        scratch = folder / "scratch.tmp"
        scratch.write_bytes(b"explicit disposable")

        # The supplied Lake is the actual route, not the default cfg DB path.
        # Legacy terminal fixture: no fabricated native attempt or receipt.
        run_cleanup(results, {"cleanup": {"patterns": ["**/*"]}}, lake=lake)

        assert database.is_file(), "cleanup deleted its own actual authority database"
        assert database.read_bytes() == original
        assert not scratch.exists(), "known-closed disposable positive path must still work"
        reopened = IdeaLake(database)
        try:
            assert reopened.get_fsm_state(folder.name) == "COMPLETE"
        finally:
            reopened.close()
    finally:
        lake.close()
