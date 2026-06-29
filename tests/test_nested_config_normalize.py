"""Normalizer that rescues known nested-config mistakes pre-validation."""
from orze.engine.launcher import (
    normalize_nested_config,
    validate_idea_config_no_nested,
)


def test_no_map_is_noop():
    cfg = {"lora": {"rank": 8}, "lr": 0.001}
    out, changes = normalize_nested_config(cfg, None)
    assert out == cfg and changes == []


def test_flatten_prefix():
    cfg = {"lora": {"rank": 8, "alpha": 16, "dropout": 0.05}, "lr": 0.001}
    out, changes = normalize_nested_config(cfg, {"lora": "flatten_prefix"})
    assert "lora" not in out
    assert out["lora_rank"] == 8
    assert out["lora_alpha"] == 16
    assert out["lora_dropout"] == 0.05
    assert out["lr"] == 0.001
    assert len(changes) == 3


def test_flatten_prefix_does_not_overwrite_explicit_flat_key():
    cfg = {"lora": {"rank": 8}, "lora_rank": 99}
    out, _ = normalize_nested_config(cfg, {"lora": "flatten_prefix"})
    assert out["lora_rank"] == 99  # explicit flat key wins


def test_rename_to_whitelisted_key():
    cfg = {"datasets": {"ami": 0.5, "gigaspeech": 0.5}, "lr": 0.001}
    out, changes = normalize_nested_config(cfg, {"datasets": "rename:data_mix"})
    assert "datasets" not in out
    assert out["data_mix"] == {"ami": 0.5, "gigaspeech": 0.5}
    assert "datasets -> data_mix" in changes


def test_rename_merges_existing_dict_explicit_wins():
    cfg = {"dataset_mix": {"ami": 0.5, "ted": 0.1}, "data_mix": {"ami": 0.9}}
    out, _ = normalize_nested_config(cfg, {"dataset_mix": "rename:data_mix"})
    # existing explicit data_mix value wins on key collision
    assert out["data_mix"]["ami"] == 0.9
    assert out["data_mix"]["ted"] == 0.1


def test_unconfigured_nested_dict_still_rejected_after_normalize():
    cfg = {"lora": {"rank": 8}, "backbone": {"name": "x"}}
    out, _ = normalize_nested_config(cfg, {"lora": "flatten_prefix"})
    # lora rescued, but backbone (unconfigured) still fails the F5 guard
    err = validate_idea_config_no_nested(out)
    assert err is not None and "backbone" in err
    assert "lora" not in err


def test_normalized_config_passes_validator():
    cfg = {"lora": {"rank": 8, "alpha": 16}, "eval": {"max_samples": 500}}
    nmap = {"lora": "flatten_prefix", "eval": "flatten_prefix"}
    out, _ = normalize_nested_config(cfg, nmap)
    assert validate_idea_config_no_nested(out) is None


def test_non_dict_value_ignored():
    cfg = {"lora": "not_a_dict", "lr": 0.001}
    out, changes = normalize_nested_config(cfg, {"lora": "flatten_prefix"})
    assert out == cfg and changes == []
