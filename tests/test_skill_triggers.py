"""Tests for orze.skills.triggers."""
import pytest

from orze.skills.triggers import evaluate_trigger


def test_none_trigger_is_always_active():
    assert evaluate_trigger(None, {}) is True


def test_always_literal_is_active():
    assert evaluate_trigger("always", {}) is True
    assert evaluate_trigger("  Always  ", {}) is True


def test_periodic_research_cycles_fires_on_interval():
    assert evaluate_trigger(
        "periodic_research_cycles(5)",
        {"research_cycles": 5, "last_activation_cycle": 0},
    ) is True


def test_periodic_research_cycles_blocks_under_interval():
    assert evaluate_trigger(
        "periodic_research_cycles(5)",
        {"research_cycles": 3, "last_activation_cycle": 0},
    ) is False


def test_periodic_research_cycles_respects_last_activation():
    # 10 cycles total, last fired at cycle 8 → 2 cycles elapsed, under interval
    assert evaluate_trigger(
        "periodic_research_cycles(5)",
        {"research_cycles": 10, "last_activation_cycle": 8},
    ) is False


def test_on_file_exists(tmp_path):
    target = tmp_path / "exists.txt"
    target.write_text("present")
    assert evaluate_trigger(f"on_file({target})", {}) is True


def test_on_file_missing(tmp_path):
    target = tmp_path / "ghost.txt"
    assert evaluate_trigger(f"on_file({target})", {}) is False


def test_on_plateau_fires_above_threshold():
    assert evaluate_trigger("on_plateau(20)", {"plateau_patience": 25}) is True


def test_on_plateau_blocks_under_threshold():
    assert evaluate_trigger("on_plateau(20)", {"plateau_patience": 10}) is False


def test_unknown_trigger_raises():
    with pytest.raises(ValueError):
        evaluate_trigger("bogus(42)", {})
