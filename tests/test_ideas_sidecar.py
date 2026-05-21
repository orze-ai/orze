"""Sidecar overlay: ideas.d/*.md entries are picked up even when absent from ideas.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_sidecar_overlay_fills_missing_entry(tmp_path):
    from orze.core.ideas import parse_ideas

    ideas_md = tmp_path / "ideas.md"
    ideas_md.write_text("# Ideas\n\n", encoding="utf-8")

    sidecar_dir = tmp_path / "ideas.d"
    sidecar_dir.mkdir()
    (sidecar_dir / "idea-foo-bar.md").write_text(
        "## idea-foo-bar: Hyphenated slug test\n"
        "- **Priority**: critical\n"
        "- **Approach Family**: test\n"
        "- **Config overrides**:\n"
        "  ```yaml\n"
        "  rank: 8\n"
        "  ```\n",
        encoding="utf-8",
    )

    ideas = parse_ideas(str(ideas_md))
    assert "idea-foo-bar" in ideas, "sidecar entry not found in parse_ideas result"
    assert ideas["idea-foo-bar"]["config"]["rank"] == 8


def test_sidecar_does_not_override_ideas_md_entry(tmp_path):
    from orze.core.ideas import _parse_ideas_cache, parse_ideas

    # Reset cache so previous test doesn't pollute
    _parse_ideas_cache["mtime"] = 0.0
    _parse_ideas_cache["result"] = {}

    ideas_md = tmp_path / "ideas.md"
    ideas_md.write_text(
        "## idea-foo-bar: Primary entry\n"
        "- **Priority**: medium\n"
        "- **Approach Family**: baseline\n"
        "  ```yaml\n"
        "  rank: 64\n"
        "  ```\n",
        encoding="utf-8",
    )

    sidecar_dir = tmp_path / "ideas.d"
    sidecar_dir.mkdir()
    (sidecar_dir / "idea-foo-bar.md").write_text(
        "## idea-foo-bar: Sidecar entry\n"
        "- **Priority**: critical\n"
        "  ```yaml\n"
        "  rank: 8\n"
        "  ```\n",
        encoding="utf-8",
    )

    ideas = parse_ideas(str(ideas_md))
    assert ideas["idea-foo-bar"]["config"].get("rank") == 64, "ideas.md must take precedence over sidecar"
