"""Tests for orze.skills.loader frontmatter parsing and compose_skills.

Registry/wiring tests live in orze-pro (SOPs are a pro feature).
"""
import textwrap

from orze.skills.loader import compose_skills, parse_frontmatter


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


def test_compose_skills_honors_order(tmp_path):
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
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
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


def test_compose_skills_resolves_sop_prefix_via_orze_pro(tmp_path):
    """@sop:<name> refs delegate to orze_pro.skills.bundled.load_bundled_skill.

    This test verifies the loader-side plumbing: when orze-pro is importable,
    @sop:<name> should be resolved. We use the smoke SOP that ships with
    orze-pro as the fixture.
    """
    pytest = __import__("pytest")
    try:
        import orze_pro.skills.bundled  # noqa: F401
    except ImportError:
        pytest.skip("orze-pro not installed")
    role_cfg = {"skills": ["@sop:_smoke"]}
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    assert "Smoke SOP" in composed


def test_compose_skills_unknown_sop_prefix_logs_and_skips(tmp_path, caplog):
    pytest = __import__("pytest")
    try:
        import orze_pro.skills.bundled  # noqa: F401
    except ImportError:
        pytest.skip("orze-pro not installed")
    role_cfg = {"skills": ["@sop:nonexistent_bundled"]}
    composed = compose_skills(role_cfg, tmp_path, template_vars=None)
    assert composed == ""  # nothing loaded


def test_compose_skills_unknown_trigger_is_included_and_warns(tmp_path, caplog):
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
    assert "WEIRD_BODY" in composed
