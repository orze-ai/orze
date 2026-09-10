"""L2 new structured-composition acceptance, not missing-old-API red evidence.

Only temporary skill sources are read; no Pro service, provider or process runs.
Legacy string/gate behavior remains covered alongside strict native behavior.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

from orze.skills import loader
from orze.skills.loader import compose_skills, compose_skills_with_manifest


FIELDS = {"source_key", "requested_ref", "resolved_source", "trigger", "source_sha256"}


def _skill(root, name, *, trigger=None, body=None, order=100, identifier="same-display-id"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = f"trigger: {trigger}\n" if trigger is not None else ""
    path.write_text(
        f"---\nid: {identifier}\nname: same-display-name\norder: {order}\n{gate}---\n"
        + (name if body is None else body), encoding="utf-8")
    return path


def test_manifest_matches_actual_sorted_nonempty_included_fragments(tmp_path):
    early = _skill(tmp_path, "early.md", body="EARLY {cycle}", order=1)
    _skill(tmp_path, "late.md", body="LATE", order=90)
    _skill(tmp_path, "blocked.md", trigger="periodic_research_cycles(20)")
    _skill(tmp_path, "blank.md", body=" ")
    cfg = {"skills": ["late.md", "blocked.md", "missing.md", "blank.md", "early.md"]}
    before = deepcopy(cfg)

    result = compose_skills_with_manifest(cfg, tmp_path, {"cycle": "7"}, strict=True)

    assert result["text"] == "EARLY 7\n\n---\n\nLATE"
    assert [row["requested_ref"] for row in result["included"]] == ["early.md", "late.md"]
    assert all(set(row) == FIELDS for row in result["included"])
    assert result["included"][0]["source_sha256"] == hashlib.sha256(early.read_bytes()).hexdigest()
    assert all("body" not in row for row in result["included"])
    assert cfg == before
    assert {row["reason"] for row in result["excluded"]} == {
        "skill_trigger_not_met", "skill_content_empty"}


def test_local_relative_absolute_and_symlink_aliases_share_source_key(tmp_path):
    path = _skill(tmp_path, "skills/candidate.md")
    alias = tmp_path / "alias.md"
    alias.symlink_to(path)
    refs = ["./skills/candidate.md", str(path), "alias.md"]

    result = compose_skills_with_manifest({"skills": refs}, tmp_path, strict=True)

    assert len(result["included"]) == 3  # Preserve actual repeated fragments.
    assert {row["source_key"] for row in result["included"]} == {"file:" + str(path.resolve())}
    assert {row["resolved_source"] for row in result["included"]} == {str(path.resolve())}
    assert [row["requested_ref"] for row in result["included"]] == refs


def test_same_frontmatter_ids_do_not_share_periodic_clocks(tmp_path):
    _skill(tmp_path, "five.md", trigger="periodic_research_cycles(5)")
    _skill(tmp_path, "twenty.md", trigger="periodic_research_cycles(20)")
    config = {"skills": ["five.md", "twenty.md"]}
    clocks = {}
    seen = []
    for cycle in (5, 10, 15, 20):
        def context(descriptor):
            seen.append(deepcopy(descriptor))
            return {"research_cycles": cycle,
                    "last_activation_cycle": clocks.get(descriptor["source_key"], 0)}
        before = deepcopy(clocks)
        result = compose_skills_with_manifest(config, tmp_path,
                                              context_for_skill=context, strict=True)
        assert clocks == before  # Composition itself never acknowledges launch.
        assert [row["requested_ref"] for row in result["included"]] == (
            ["five.md", "twenty.md"] if cycle == 20 else ["five.md"])
        for row in result["included"]:  # Simulated successful-launch caller ACK.
            clocks[row["source_key"]] = cycle
    assert len(seen) == 8 and all(set(row) == FIELDS for row in seen)


def test_hash_binds_same_read_and_callback_cannot_mutate_manifest(tmp_path):
    source = _skill(tmp_path, "frozen.md", body="OLD {cycle}")
    original = source.read_bytes()

    def mutate_source_and_descriptor(descriptor):
        source.write_text("NEW CONTENT", encoding="utf-8")
        descriptor.clear()
        return {}

    result = compose_skills_with_manifest(
        {"skills": ["frozen.md"]}, tmp_path, {"cycle": "17"}, strict=True,
        context_for_skill=mutate_source_and_descriptor)

    assert result["text"] == "OLD 17"
    assert result["included"][0]["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["included"][0]["source_key"] == "file:" + str(source.resolve())
    later = compose_skills_with_manifest({"skills": ["frozen.md"]}, tmp_path, strict=True)
    assert later["included"][0]["source_key"] == result["included"][0]["source_key"]
    assert later["included"][0]["source_sha256"] != result["included"][0]["source_sha256"]


def test_builtin_source_key_survives_install_path_change(tmp_path, monkeypatch):
    roots = [tmp_path / "installation-a", tmp_path / "installation-b"]
    descriptors = []
    for root in roots:
        _skill(root, "core.skill.md", body="CORE BODY")
        monkeypatch.setattr(loader, "_SKILLS_DIR", root)
        result = compose_skills_with_manifest({"skills": ["@core"]}, tmp_path, strict=True)
        descriptors.append(result["included"][0])
    assert {row["source_key"] for row in descriptors} == {"orze:skills/core.skill.md"}
    assert descriptors[0]["resolved_source"] != descriptors[1]["resolved_source"]
    assert descriptors[0]["source_sha256"] == descriptors[1]["source_sha256"]


def test_bundled_manifest_uses_actual_resolver_not_registry_id_override(tmp_path, monkeypatch):
    bundled_path = _skill(tmp_path, "package/sops/example.skill.md", body="BUNDLED BODY")
    _skill(tmp_path, "skills/override.skill.md", body="DYNAMIC SAME ID")
    package = ModuleType("orze_pro")
    package.__path__ = []
    skills = ModuleType("orze_pro.skills")
    skills.__path__ = []
    bundled = ModuleType("orze_pro.skills.bundled")
    calls = []

    def load(name):
        calls.append(name)
        return bundled_path.read_text(encoding="utf-8"), bundled_path

    bundled.load_bundled_skill = load
    # Fixture-local optional-package substitute, automatically restored in full.
    for name, module in (("orze_pro", package), ("orze_pro.skills", skills),
                         ("orze_pro.skills.bundled", bundled)):
        monkeypatch.setitem(sys.modules, name, module)
    result = compose_skills_with_manifest({"skills": ["@sop:example"]}, tmp_path, strict=True)
    assert calls == ["example"]
    assert result["text"] == "BUNDLED BODY"
    assert result["included"][0]["source_key"] == "orze_pro:sops/example.skill.md"
    assert result["included"][0]["resolved_source"] == str(bundled_path.resolve())


@pytest.mark.parametrize("gate", ["bogus_expression(99)", "on_plateu(2)", "false", "[always]"])
def test_strict_invalid_gate_excludes_before_callback_while_legacy_includes(tmp_path, gate):
    _skill(tmp_path, "bad.md", trigger=gate, body="NOT STRICTLY AUTHORIZED")
    cfg = {"skills": ["bad.md"]}
    legacy = compose_skills(cfg, tmp_path)

    result = compose_skills_with_manifest(
        cfg, tmp_path, strict=True,
        context_for_skill=lambda descriptor: pytest.fail("Invalid gate must not request context"))

    assert legacy == "NOT STRICTLY AUTHORIZED"
    assert result["text"] == "" and result["included"] == []
    assert result["excluded"][0]["reason"] == "skill_trigger_invalid"


@pytest.mark.parametrize("context", [None, [], "unknown", {"research_cycles": "unknown"}])
def test_strict_bad_callback_context_excludes_without_falling_back(tmp_path, context):
    _skill(tmp_path, "periodic.md", trigger="periodic_research_cycles(5)")
    cfg = {"skills": ["periodic.md"], "_trigger_context": {"research_cycles": 99}}
    result = compose_skills_with_manifest(cfg, tmp_path, strict=True,
                                          context_for_skill=lambda descriptor: context)
    assert result["text"] == "" and result["included"] == []
    assert result["excluded"][0]["reason"].startswith("skill_")


def test_strict_callback_exception_does_not_include_failed_skill(tmp_path):
    _skill(tmp_path, "only.md")

    def unavailable(descriptor):
        raise RuntimeError("synthetic context unavailable")

    result = compose_skills_with_manifest({"skills": ["only.md"]}, tmp_path,
                                          strict=True, context_for_skill=unavailable)
    assert result["text"] == "" and result["included"] == []
    assert result["excluded"][0]["reason"] == "skill_context_invalid"


@pytest.mark.parametrize("value", [True, -1, 2.5, "20"])
def test_strict_context_counts_cannot_be_coerced_into_periodic_eligibility(tmp_path, value):
    _skill(tmp_path, "periodic.md", trigger="periodic_research_cycles(1)")
    result = compose_skills_with_manifest(
        {"skills": ["periodic.md"]}, tmp_path, strict=True,
        context_for_skill=lambda descriptor: {"research_cycles": value})
    assert result["text"] == "" and result["included"] == []
    assert result["excluded"][0]["reason"] == "skill_context_invalid"


@pytest.mark.parametrize("gate", ["manual", "plateau_or_new", "on_plateau(0)",
                                "periodic_research_cycles(0)"])
def test_legacy_aliases_remain_explicitly_always_not_evidence_gates(tmp_path, caplog, gate):
    _skill(tmp_path, "legacy.md", trigger=gate, body="LEGACY BODY")
    result = compose_skills_with_manifest({"skills": ["legacy.md"]}, tmp_path, strict=True)
    assert result["text"] == "LEGACY BODY"
    assert result["included"][0]["trigger"] == gate
    assert "legacy always-included compatibility" in caplog.text


@pytest.mark.parametrize("frontmatter", ["trigger: [", "- trigger: always", "trigger: always\n"])
def test_strict_malformed_header_cannot_turn_into_implicit_always(tmp_path, frontmatter):
    path = tmp_path / "malformed.md"
    ending = "" if frontmatter.endswith("\n") else "\n---\nBODY"
    path.write_text("---\n" + frontmatter + ending, encoding="utf-8")
    result = compose_skills_with_manifest({"skills": [str(path)]}, tmp_path, strict=True)
    assert result["text"] == "" and result["included"] == []
    assert result["excluded"][0]["reason"] == "skill_frontmatter_invalid"


def test_legacy_empty_separator_output_never_invents_manifest_inclusion(tmp_path):
    _skill(tmp_path, "empty-a.md", body="")
    _skill(tmp_path, "empty-b.md", body="")
    config = {"skills": ["empty-a.md", "empty-b.md"]}
    assert compose_skills(config, tmp_path) == "\n\n---\n\n"
    legacy = compose_skills_with_manifest(config, tmp_path)
    assert legacy["text"] == "\n\n---\n\n" and legacy["included"] == []
    native = compose_skills_with_manifest(config, tmp_path, strict=True)
    assert native["text"] == "" and native["included"] == []


def test_template_emptied_fragment_is_not_included(tmp_path):
    _skill(tmp_path, "conditional.md", body="{optional}")
    result = compose_skills_with_manifest({"skills": ["conditional.md"]}, tmp_path,
                                          {"optional": ""}, strict=True)
    assert result["text"] == "" and result["included"] == []


def test_on_file_keeps_existing_path_evaluation_and_does_not_consume(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    _skill(root, "file.md", trigger="on_file(signal)", body="FILE BODY")
    signal = tmp_path / "signal"
    signal.write_text("operator payload", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = compose_skills_with_manifest({"skills": ["file.md"]}, root, strict=True)
    assert result["text"] == "FILE BODY"  # Existing cwd-relative semantics.
    assert signal.read_text(encoding="utf-8") == "operator payload"
