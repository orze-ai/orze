"""Tests for orze.skills.registry and loader frontmatter handling."""
import textwrap

from orze.skills.loader import parse_frontmatter
from orze.skills.registry import (
    SkillMetadata,
    discover_skills,
    validate_wiring,
)


def _write_skill(dirpath, filename, body_yaml):
    path = dirpath / filename
    path.write_text(textwrap.dedent(body_yaml).lstrip("\n"), encoding="utf-8")
    return path


def test_skill_frontmatter_exposes_sop_metadata(tmp_path):
    p = tmp_path / "sop_example.skill.md"
    p.write_text(textwrap.dedent("""\
        ---
        name: anomaly_hypotheses
        id: sop-001
        role: data_analyst
        order: 20
        produces: [results/_failure_hypotheses.md]
        consumed_by: [research, thinker]
        requires: []
        trigger: periodic_research_cycles(5)
        ---

        ## Job: Failure-Driven Hypotheses
        Body here.
    """))
    meta, body = parse_frontmatter(p.read_text())
    assert meta["id"] == "sop-001"
    assert meta["role"] == "data_analyst"
    assert meta["produces"] == ["results/_failure_hypotheses.md"]
    assert meta["trigger"] == "periodic_research_cycles(5)"
    assert "## Job: Failure-Driven Hypotheses" in body


def test_discover_skills_scans_project_skills_dir(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "a.skill.md", """
        ---
        id: sop-a
        role: data_analyst
        produces: [out_a.md]
        ---
        body a
    """)
    _write_skill(skills_dir, "b.skill.md", """
        ---
        id: sop-b
        role: professor
        consumed_by: [research]
        ---
        body b
    """)
    skills = discover_skills(tmp_path)
    ids = {s.id: s for s in skills}
    assert "sop-a" in ids
    assert ids["sop-a"].role == "data_analyst"
    assert ids["sop-a"].produces == ["out_a.md"]
    assert "sop-b" in ids
    assert ids["sop-b"].consumed_by == ["research"]


def test_discover_ignores_skill_without_id(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "no_id.skill.md", """
        ---
        name: anonymous
        ---
        body
    """)
    skills = discover_skills(tmp_path)
    assert skills == []


def test_discover_handles_no_skills_dir(tmp_path):
    assert discover_skills(tmp_path) == []


def test_dangling_requires_is_reported():
    skills = [
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-nonexistent"]),
    ]
    issues = validate_wiring(skills)
    errs = [i for i in issues if i.severity == "error"]
    assert any("sop-nonexistent" in i.message and "requires" in i.message
               for i in errs)


def test_dangling_produces_is_reported_as_warning():
    skills = [
        SkillMetadata(id="sop-orphan", name="orphan", role="data_analyst",
                      produces=["out_orphan.md"]),
    ]
    issues = validate_wiring(skills)
    warns = [i for i in issues if i.severity == "warning"]
    assert any("out_orphan.md" in i.message and "no consumer" in i.message
               for i in warns)


def test_requires_counts_as_consumption():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"]),
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-a"]),
    ]
    issues = validate_wiring(skills)
    # sop-a is required by sop-b -> consumed, no warning expected
    assert all("sop-a" not in i.message or i.severity != "warning"
               for i in issues)


def test_consumed_by_role_name_counts_as_consumption():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"], consumed_by=["thinker"]),
        SkillMetadata(id="sop-b", name="b", role="thinker"),
    ]
    issues = validate_wiring(skills)
    # role 'thinker' exists in registry, satisfies consumed_by
    warns_for_a = [i for i in issues if i.skill_id == "sop-a"
                   and i.severity == "warning"]
    assert warns_for_a == []


def test_overrides_target_must_exist():
    skills = [
        SkillMetadata(id="sop-override", name="o", role="thinker",
                      overrides="sop-missing"),
    ]
    issues = validate_wiring(skills)
    errs = [i for i in issues if i.severity == "error"]
    assert any("sop-missing" in i.message and "overrides" in i.message
               for i in errs)


def test_compose_skills_honors_order(tmp_path):
    from orze.skills.loader import compose_skills
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "late.skill.md", """
        ---
        id: late
        order: 50
        ---
        LATE_BODY
    """)
    _write_skill(skills_dir, "early.skill.md", """
        ---
        id: early
        order: 10
        ---
        EARLY_BODY
    """)
    role_cfg = {"skills": ["./skills/early.skill.md",
                           "./skills/late.skill.md"]}
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    assert composed.index("EARLY_BODY") < composed.index("LATE_BODY")


def test_compose_skills_respects_reverse_list_order_via_frontmatter(tmp_path):
    from orze.skills.loader import compose_skills
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    # Passed in reversed list order, but frontmatter says early should win.
    _write_skill(skills_dir, "a.skill.md", """
        ---
        id: zz
        order: 90
        ---
        Z_BODY
    """)
    _write_skill(skills_dir, "b.skill.md", """
        ---
        id: aa
        order: 5
        ---
        A_BODY
    """)
    role_cfg = {"skills": ["./skills/a.skill.md", "./skills/b.skill.md"]}
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    assert composed.index("A_BODY") < composed.index("Z_BODY")


def test_compose_skills_skips_gated_skills(tmp_path):
    from orze.skills.loader import compose_skills
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "gated.skill.md", """
        ---
        id: gated
        trigger: periodic_research_cycles(10)
        ---
        GATED_BODY
    """)
    _write_skill(skills_dir, "always.skill.md", """
        ---
        id: always
        ---
        ALWAYS_BODY
    """)
    role_cfg = {
        "skills": ["./skills/gated.skill.md", "./skills/always.skill.md"],
        "_trigger_context": {"research_cycles": 2,
                             "last_activation_cycle": 0},
    }
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    assert "ALWAYS_BODY" in composed
    assert "GATED_BODY" not in composed


def test_compose_skills_unknown_trigger_is_included_and_warns(tmp_path, caplog):
    from orze.skills.loader import compose_skills
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "weird.skill.md", """
        ---
        id: weird
        trigger: bogus_expression(99)
        ---
        WEIRD_BODY
    """)
    role_cfg = {"skills": ["./skills/weird.skill.md"]}
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    # fail-open: unknown triggers don't silently drop the skill
    assert "WEIRD_BODY" in composed


def test_clean_wiring_produces_no_issues():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"], consumed_by=["thinker"]),
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-a"]),
    ]
    issues = validate_wiring(skills)
    assert issues == []
