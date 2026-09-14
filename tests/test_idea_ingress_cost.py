"""Real source locks/SQLite for local ingress lookup cost, not worker evidence.

Historical rows are synthetic archived metadata inserted directly for query
scope tests. Every proposed admission/ACK still uses the real production path.
SQL trace and a delegating cache-loader spy observe work; they do not replace it.
"""
import json
from pathlib import Path

import pytest
import yaml

from orze.engine import idea_ingress
from test_idea_ingress_contract import engine, _block


def _history(lake, count):
    lake.conn.executemany(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES(?,?,?,?,?)",
        [(f"idea-history-{n:05d}", "archived metadata", "seed: -1\n", "", "archived")
         for n in range(count)],
    )
    lake.conn.commit()


def _observe(instance, monkeypatch):
    queries, loads = [], []
    instance.lake.conn.set_trace_callback(queries.append)
    original = instance._load_config_hashes

    def load():
        loads.append("actual_cache_loader")
        return original()

    monkeypatch.setattr(instance, "_load_config_hashes", load)
    return queries, loads


def _normalized(query):
    return " ".join(query.lower().split())


@pytest.mark.parametrize("content", ["", "# Ideas\n", "## idea-bad: Invalid\n```yaml\nseed: [\n```\n",
                                     _block("idea-duplicate", 1) + _block("idea-duplicate", 2)])
def test_empty_parsed_batch_reads_no_lake_ids_or_config_cache(engine, monkeypatch, content):
    instance, cfg, source = engine
    _history(instance.lake, 1000)
    source.write_text(content, encoding="utf-8")
    queries, loads = _observe(instance, monkeypatch)

    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)

    print("INGRESS_EMPTY=" + json.dumps({"sql": queries, "cache_loads": len(loads)}))
    assert raw == {} and inserted == []
    assert source.read_text(encoding="utf-8") == content
    assert queries == [], "an already empty parsed batch must not read historical IDs"
    assert loads == [], "an already empty parsed batch must not load the global config cache"
    assert not (instance.results_dir / ".ideas_md.lock").exists()


@pytest.mark.parametrize("history_count", [0, 1000])
def test_nonempty_batch_uses_only_targeted_id_query_and_real_admission(engine, monkeypatch, history_count):
    instance, cfg, source = engine
    _history(instance.lake, history_count)
    source.write_text("# Ideas\n" + _block("idea-fresh", 5), encoding="utf-8")
    queries, loads = _observe(instance, monkeypatch)

    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)

    selects = [_normalized(q) for q in queries if _normalized(q).startswith("select idea_id from")]
    print("INGRESS_BATCH=" + json.dumps({"history_rows": history_count, "id_selects": selects,
                                        "total_sql": len(queries), "cache_loads": len(loads)}))
    assert inserted == ["idea-fresh"] and set(raw) == {"idea-fresh"}
    assert source.read_text(encoding="utf-8") == "# Ideas\n"
    assert not any(q == "select idea_id from ideas" for q in selects)
    assert len(selects) == 1 and "where idea_id in (" in selects[0]
    assert "'idea-fresh'" in selects[0] and "idea-history-" not in selects[0]
    assert loads == ["actual_cache_loader"]


def test_empty_first_page_advances_to_real_later_proposal_without_loading_history(engine, monkeypatch):
    instance, cfg, source = engine
    malformed = "".join(f"## idea-invalid-{n:03d}: Invalid\n```yaml\nseed: [\n```\n" for n in range(128))
    original = "# Ideas\n" + malformed + _block("idea-late", 9)
    source.write_text(original, encoding="utf-8")
    queries, loads = _observe(instance, monkeypatch)
    first = idea_ingress.ingest_ideas_source(instance, cfg)
    first_queries, first_loads = list(queries), list(loads)
    second = idea_ingress.ingest_ideas_source(instance, cfg)
    assert first == ({}, [])
    assert second[1] == ["idea-late"]
    assert source.read_text(encoding="utf-8") == "# Ideas\n" + malformed
    assert first_queries == [] and first_loads == []
    assert loads == ["actual_cache_loader"]


def test_empty_primary_still_reads_and_admits_additive_sidecar(engine, monkeypatch):
    instance, cfg, source = engine
    source.write_bytes(b"")
    sidecar = source.parent / "ideas.d" / "side.md"
    sidecar.parent.mkdir()
    content = _block("idea-side", 11)
    sidecar.write_text(content, encoding="utf-8")
    queries, loads = _observe(instance, monkeypatch)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ["idea-side"] and raw["idea-side"]["_overlay_source"] == "sidecar"
    assert source.read_bytes() == b"" and sidecar.read_text(encoding="utf-8") == content
    assert loads == ["actual_cache_loader"]
    assert any("idea-side" in q for q in queries)


@pytest.mark.parametrize("changed", [False, True])
def test_known_same_id_bypasses_wrong_cache_but_still_requires_real_exact_insert(engine, changed):
    instance, cfg, source = engine
    initial = "# Ideas\n" + _block("idea-known", 17)
    source.write_text(initial, encoding="utf-8")
    assert idea_ingress.ingest_ideas_source(instance, cfg)[1] == ["idea-known"]
    old = instance.lake.get("idea-known")
    instance.lake.insert("idea-other", "Other", "seed: 18\n", "", status="completed")
    seed = 19 if changed else 17
    instance._save_config_hash("idea-other", {"seed": seed})
    replay = "# Ideas\n" + _block("idea-known", seed)
    source.write_text(replay, encoding="utf-8")
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert set(raw) == {"idea-known"} and inserted == []
    assert instance.lake.get("idea-known") == old
    assert source.read_text(encoding="utf-8") == (replay if changed else "# Ideas\n")


@pytest.mark.parametrize("same_domain", [False, True])
def test_cross_id_legacy_cache_still_respects_cpu_kind_domain(engine, same_domain):
    instance, cfg, source = engine
    config = {"action": {"version": 1, "adapter": "command", "purpose": "metadata test",
                         "inputs": {}, "command": ["not-executed"], "timeout_seconds": 1, "outputs": {}}}
    instance.lake.insert("idea-winner", "Winner", yaml.safe_dump(config), "", status="queued",
                         kind="native_cpu_action" if same_domain else "train")
    instance._save_config_hash("idea-winner", config)
    original = ("# Ideas\n## idea-cpu: CPU metadata\n- **Kind**: native_cpu_action\n```yaml\n"
                + yaml.safe_dump(config) + "```\n")
    source.write_text(original, encoding="utf-8")
    _, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert inserted == ([] if same_domain else ["idea-cpu"])
    assert source.read_text(encoding="utf-8") == (original if same_domain else "# Ideas\n")
    if not same_domain:
        assert instance.lake.get("idea-cpu")["kind"] == "native_cpu_action"


def test_redirected_source_is_rejected_before_any_history_read(engine, monkeypatch):
    instance, cfg, source = engine
    foreign = source.with_name("foreign.md")
    foreign.write_text(_block("idea-foreign", 22), encoding="utf-8")
    source.unlink()
    source.symlink_to(foreign.name)
    before = foreign.read_bytes()
    queries, loads = _observe(instance, monkeypatch)
    assert idea_ingress.ingest_ideas_source(instance, cfg) == ({}, [])
    assert queries == [] and loads == []
    assert foreign.read_bytes() == before


def test_actual_source_lock_loss_after_lookup_never_admits_or_acknowledges(engine, monkeypatch):
    instance, cfg, source = engine
    original = _block("idea-lost-lock", 23)
    source.write_text(original, encoding="utf-8")
    real_load = instance._load_config_hashes
    observed = []

    def load_and_move_lock():
        result = real_load()
        lock = instance.results_dir / ".ideas_md.lock"
        lock.rename(instance.results_dir / "displaced-source-lock")
        observed.append("actual_directory_replaced")
        return result

    monkeypatch.setattr(instance, "_load_config_hashes", load_and_move_lock)
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert observed == ["actual_directory_replaced"]
    assert set(raw) == {"idea-lost-lock"} and inserted == []
    assert instance.lake.get("idea-lost-lock") is None
    assert source.read_text(encoding="utf-8") == original


def test_targeted_reader_empty_does_no_sql_and_legacy_reader_is_unchanged(engine):
    instance, _, _ = engine
    _history(instance.lake, 3)
    instance.lake.insert("idea-queued", "Queued", "seed: 41\n", "", status="queued")
    queries = []
    instance.lake.conn.set_trace_callback(queries.append)
    assert instance.lake.find_existing_ids(iter(())) == set()
    assert queries == []
    assert instance.lake.get_all_ids() == {"idea-queued", *(f"idea-history-{n:05d}" for n in range(3))}
    assert instance.lake.get_all_ids(status="queued") == {"idea-queued"}


def test_targeted_reader_128_bound_parameters_keep_duplicate_and_literal_ids_safe(engine):
    instance, _, _ = engine
    _history(instance.lake, 1000)
    literal_id = "idea-quote') OR 1=1 --"
    instance.lake.conn.execute(
        "INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES(?,?,?,?,?)",
        (literal_id, "literal metadata ID", "seed: -1\n", "", "archived"),
    )
    instance.lake.conn.commit()
    requested = [f"idea-history-{n:05d}" for n in range(125)]
    requested += [requested[0], "idea-missing", literal_id]
    queries = []
    instance.lake.conn.set_trace_callback(queries.append)
    found = instance.lake.find_existing_ids(iter(requested))
    assert len(requested) == 128
    assert found == set(requested) - {"idea-missing"}
    assert len(queries) == 1 and "from main.ideas where idea_id in (" in _normalized(queries[0])
    assert "idea-history-00999" not in found


@pytest.mark.parametrize("invalid", [None, "idea-scalar", [""], [True], ["idea-x"] * 129])
def test_targeted_reader_invalid_batch_fails_before_sql(engine, invalid):
    instance, _, _ = engine
    queries = []
    instance.lake.conn.set_trace_callback(queries.append)
    with pytest.raises(ValueError, match="idea_id_batch_invalid"):
        instance.lake.find_existing_ids(invalid)
    assert queries == []


def test_lightweight_lake_adapter_uses_bounded_get_and_real_insert(engine):
    instance, cfg, source = engine
    source.write_text(_block("idea-known", 51), encoding="utf-8")
    assert idea_ingress.ingest_ideas_source(instance, cfg)[1] == ["idea-known"]
    actual_lake = instance.lake
    reads, global_reads = [], []

    class LegacyAdapter:
        # Only the new optional method is absent. All persistence delegates to
        # the same actual SQLite lake, including atomic admission and readback.
        def __getattr__(self, name):
            if name == "find_existing_ids":
                raise AttributeError(name)
            return getattr(actual_lake, name)

        def get(self, idea_id):
            reads.append(idea_id)
            return actual_lake.get(idea_id)

        def get_all_ids(self, *args, **kwargs):
            global_reads.append("global_read")
            return actual_lake.get_all_ids(*args, **kwargs)

    instance.lake = LegacyAdapter()
    source.write_text(_block("idea-known", 51) + _block("idea-new", 52), encoding="utf-8")
    raw, inserted = idea_ingress.ingest_ideas_source(instance, cfg)
    assert set(raw) == {"idea-known", "idea-new"} and inserted == ["idea-new"]
    assert reads[:2] == ["idea-known", "idea-new"] and global_reads == []
    assert actual_lake.get("idea-new") is not None and source.read_bytes() == b""
