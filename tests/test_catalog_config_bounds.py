"""Opt-in observer configuration is bounded data, never lifecycle authority.

These are acceptance tests for a new catalog option. An absent keyword/API is
mechanism absence, not a behavioral regression in a previous release.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from orze.idea_lake import IdeaLake
from orze.reporting import catalog


@pytest.fixture
def lake(tmp_path):
    instance = IdeaLake(tmp_path / "ideas.db")
    instance.insert("idea-config", "Existing task", "seed: 13", "", status="completed")
    try:
        yield instance
    finally:
        instance.close()


def _replace_config(lake, payload):
    # Avoid any unrelated IdeaLake YAML summary processing: only the catalog
    # parser under test receives these intentionally malformed stored values.
    lake.conn.execute("UPDATE ideas SET config=? WHERE idea_id='idea-config'", (payload,))
    lake.conn.commit()


def _read(lake):
    before = Path(lake.db_path).read_bytes()
    snapshot = catalog.load_catalog_snapshot(lake.db_path, include_configs=True)
    assert snapshot.available
    record = snapshot.records["idea-config"]
    assert record["lifecycle_state"] == "COMPLETE"
    assert record["lifecycle_reason"] == "lifecycle_agreed"
    assert snapshot.get_lifecycle_counts() == {"COMPLETED": 1}
    assert snapshot.unknown_count == 0
    assert Path(lake.db_path).read_bytes() == before
    return record


def _assert_unavailable(lake):
    record = _read(lake)
    assert record["config_available"] is False
    assert record["config"] == {}


def test_explicit_config_opt_in_loads_normal_json_safe_seed_thirteen(lake):
    expected = {"seed": 13, "model": {"layers": [2, 3], "enabled": True},
                "optional": None, "name": "示例"}
    _replace_config(lake, json.dumps(expected, ensure_ascii=False))

    record = _read(lake)

    assert record["config_available"] is True
    assert record["config"] == expected
    assert json.loads(json.dumps(record["config"], allow_nan=False)) == expected


def test_oversize_ascii_or_utf8_config_is_rejected_before_yaml_parsing(lake, monkeypatch):
    parse = Mock(side_effect=AssertionError("oversize config must not reach YAML parser"))
    monkeypatch.setattr("yaml.safe_load", parse)
    for payload in ("value: " + "x" * 65536, "value: " + "研" * 22000):
        assert len(payload.encode("utf-8")) > 65536
        _replace_config(lake, payload)

        _assert_unavailable(lake)

    parse.assert_not_called()


def test_malformed_or_nonmapping_yaml_only_marks_configuration_unavailable(lake):
    for payload in ("seed: [", "- first\n- second\n", "13\n", "null\n"):
        _replace_config(lake, payload)

        _assert_unavailable(lake)


def test_nonfinite_or_non_json_yaml_types_cannot_enter_admin_configuration(lake):
    for payload in (
        "value: .nan\n", "value: .inf\n", "date: 2026-09-10\n",
        "binary: !!binary aGk=\n", "value: !Custom {}\n",
    ):
        _replace_config(lake, payload)

        _assert_unavailable(lake)


def test_recursive_yaml_alias_is_rejected_without_changing_lifecycle(lake):
    _replace_config(lake, "recursive: &loop [*loop]\n")

    _assert_unavailable(lake)


def test_deep_or_wide_configuration_exceeding_structure_budgets_is_rejected(lake):
    for payload in (
        "nested: " + "[" * 40 + "0" + "]" * 40,
        "nodes: [" + ",".join("0" for _ in range(2100)) + "]",
    ):
        assert len(payload.encode("utf-8")) < 65536
        _replace_config(lake, payload)

        _assert_unavailable(lake)


def test_compact_alias_graph_cannot_expand_past_node_or_string_budgets(lake):
    levels = ["level0: &level0 [0, 0]"]
    for level in range(1, 12):
        levels.append(f"level{level}: &level{level} [*level{level - 1}, *level{level - 1}]")
    expanded_strings = "text: &text " + "x" * 4000 + "\nvalues: [" + ", ".join(
        "*text" for _ in range(20)
    ) + "]\n"
    for payload in ("\n".join(levels), expanded_strings):
        # The encoded YAML is small; accepting the expanded object would break
        # the separate JSON-shaped node or aggregate UTF-8 string budget.
        assert len(payload.encode("utf-8")) < 65536
        _replace_config(lake, payload)

        _assert_unavailable(lake)


def test_default_report_snapshot_never_selects_sizes_or_parses_huge_configuration(
    lake, monkeypatch,
):
    _replace_config(lake, "huge: " + "x" * (2 * 1024 * 1024))
    original = catalog._open_authoritative_lifecycle
    statements = []

    def trace(path):
        connection, reason = original(path)
        if connection is not None:
            connection.set_trace_callback(statements.append)
        return connection, reason

    monkeypatch.setattr(catalog, "_open_authoritative_lifecycle", trace)
    parse = Mock(side_effect=AssertionError("report snapshot must not parse configs"))
    monkeypatch.setattr("yaml.safe_load", parse)

    snapshot = catalog.load_catalog_snapshot(lake.db_path)

    assert snapshot.available
    assert snapshot.records["idea-config"]["lifecycle_state"] == "COMPLETE"
    assert "config" not in snapshot.records["idea-config"]
    assert "config_available" not in snapshot.records["idea-config"]
    queries = [statement.lower() for statement in statements
               if statement.lstrip().upper().startswith("SELECT")]
    assert queries
    assert all("i.config" not in query and "length(" not in query
               and "select *" not in query and "i.*" not in query for query in queries)
    parse.assert_not_called()
