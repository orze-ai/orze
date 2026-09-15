"""Lazy heading inspection retains exact sidecar parsing and first-valid precedence."""
from itertools import islice

import pytest

from orze.core import ideas


def block(key, config, title="Title"):
    return f"## {key}: {title}\n```yaml\n{config}\n```\n"


def test_first_128_valid_records_do_not_materialize_all_10000_headings(monkeypatch):
    text = "".join(block(f"idea-{i:05d}", f"seed: {i}") for i in range(10000))
    compile_re = ideas.re.compile
    matched = []

    class ObservedPattern:
        def __init__(self, pattern):
            self.pattern = pattern

        def finditer(self, source):
            for match in self.pattern.finditer(source):
                matched.append(match.start())
                yield match

    def compile(pattern, *args, **kwargs):
        result = compile_re(pattern, *args, **kwargs)
        return ObservedPattern(result) if isinstance(pattern, str) and pattern.startswith("^## (") else result

    monkeypatch.setattr(ideas.re, "compile", compile)
    seen = set()
    stream = ideas._iter_sidecar_text(text, seen)
    try:
        found = list(islice(stream, 128))
    finally:
        stream.close()
    assert [key for key, _ in found] == [f"idea-{i:05d}" for i in range(128)]
    assert [value["config"] for _, value in found] == [{"seed": i} for i in range(128)]
    assert seen == {key for key, _ in found}
    assert len(matched) <= 129, "first page eagerly materialized every later heading"


def test_adjacent_heading_boundaries_keep_unknown_heading_and_last_raw_bytes():
    text = ("Preamble\n" + block("idea-a", "seed: 1")
            + "## Unknown heading\nkeep these bytes\n"
            + block("idea-b", "seed: 2") + "final note without newline")
    found = list(ideas._iter_sidecar_text(text, set()))
    assert [key for key, _ in found] == ["idea-a", "idea-b"]
    assert found[0][1]["raw"] == "```yaml\nseed: 1\n```\n## Unknown heading\nkeep these bytes"
    assert found[1][1]["raw"] == "```yaml\nseed: 2\n```\nfinal note without newline"


def test_invalid_or_excluded_definition_keeps_first_valid_duplicate_precedence():
    text = (block("idea-a", "seed: [") + block("idea-a", "seed: 1")
            + block("idea-a", "seed: 9") + "## idea-b: No YAML\nnotes\n"
            + block("idea-b", "seed: 2") + block("idea-excluded", "seed: 3"))
    seen = {"idea-excluded"}
    found = list(ideas._iter_sidecar_text(text, seen))
    assert [(key, value["config"]) for key, value in found] == [
        ("idea-a", {"seed": 1}), ("idea-b", {"seed": 2})]
    assert seen == {"idea-a", "idea-b", "idea-excluded"}


@pytest.mark.parametrize("text", ["", "# Notes\n", "## Unknown\n", "## idea-missing: Heading only"])
def test_empty_or_unparsed_source_keeps_precedence_set_unchanged(text):
    seen = {"idea-existing"}
    assert list(ideas._iter_sidecar_text(text, seen)) == []
    assert seen == {"idea-existing"}
