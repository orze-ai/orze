import sqlite3

import pytest
import yaml

import orze.engine.reproducibility as reproduction_module
from orze.engine.reproducibility import (
    audit_campaign_reproducibility,
    config_identity_sha256,
    validate_reproducibility_contract,
)
from orze.idea_lake import IdeaLake


IDEA_IDS = ["idea-replica-a", "idea-replica-b"]
DEFAULT_CONFIGS = [
    "model: base\ntraining:\n  seed: 1\n  epochs: 2\n",
    "model: base\ntraining:\n  seed: 2\n  epochs: 2\n",
]


def _identities(configs=DEFAULT_CONFIGS, idea_ids=IDEA_IDS):
    return {
        idea_id: config_identity_sha256(yaml.safe_load(config))
        for idea_id, config in zip(idea_ids, configs)
    }


def _not_applicable(configs=DEFAULT_CONFIGS):
    return {
        "mode": "not_applicable",
        "rationale": "This campaign does not ask a replication question.",
        "expected_config_identity_sha256": _identities(configs),
    }


def _group_contract(*, tolerance=0.02, path="training.seed",
                    configs=DEFAULT_CONFIGS):
    return {
        "mode": "groups",
        "expected_config_identity_sha256": _identities(configs),
        "groups": [{
            "question": (
                "Whether independent seeds reproduce the observed metric."
            ),
            "idea_ids": list(IDEA_IDS),
            "varying_config_paths": [path],
            "max_absolute_metric_delta": tolerance,
        }],
    }


def _lake(tmp_path, configs, *, terminal_states=None, idea_ids=IDEA_IDS):
    db_path = tmp_path / "lake.db"
    lake = IdeaLake(str(db_path))
    terminal_states = terminal_states or ["COMPLETE"] * len(configs)
    for idea_id, config, terminal in zip(idea_ids, configs, terminal_states):
        lake.insert(
            idea_id,
            "replica",
            config,
            "",
            status="queued",
        )
        assert lake.record_state_transition(idea_id, "QUEUED", "CLAIMED")
        assert lake.record_state_transition(
            idea_id, "CLAIMED", "IN_PROGRESS"
        )
        assert lake.record_state_transition(
            idea_id, "IN_PROGRESS", terminal, reason="synthetic_test"
        )
    lake.close()
    return db_path


def _audit(tmp_path, monkeypatch, configs, *, contract=None, values=None,
           terminal_states=None, idea_ids=IDEA_IDS):
    db_path = _lake(
        tmp_path, configs, terminal_states=terminal_states,
        idea_ids=idea_ids,
    )
    values = values or {
        idea_id: 0.80 + index / 100
        for index, idea_id in enumerate(idea_ids)
    }
    monkeypatch.setattr(
        reproduction_module,
        "qualify_authoritative_report_evidence",
        lambda idea_id, *_args: (
            {}, {}, values.get(idea_id),
            "authoritative_local_evidence_verified",
        ),
    )
    return audit_campaign_reproducibility(
        db_path,
        tmp_path / "results",
        {"report": {"primary_metric": "score"}},
        expected_idea_ids=list(idea_ids),
        contract=contract or _group_contract(configs=configs),
    )


def test_not_applicable_verifies_distinct_configs(tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        ["training:\n  seed: 1\n", "training:\n  seed: 2\n"],
        contract=_not_applicable(
            ["training:\n  seed: 1\n", "training:\n  seed: 2\n"]
        ),
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["checks"]["no_exact_duplicate_configs"]["passed"] is True
    assert receipt["rank_claim_proven"] is False


def test_not_applicable_rejects_exact_duplicate_configs(tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        ["training:\n  seed: 1\n", "training:\n  seed: 1\n"],
        contract=_not_applicable(
            ["training:\n  seed: 1\n", "training:\n  seed: 1\n"]
        ),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["no_exact_duplicate_configs"]["passed"] is False


def test_not_applicable_rejects_explicit_seed_replicas_without_question(
        tmp_path, monkeypatch):
    configs = [
        "training:\n  seed: 1\n"
        "replication_role: seed_reproduction\n"
        "replication_index: 1\n"
        "_replicate_of: idea-root\n",
        "training:\n  seed: 2\n"
        "replication_role: seed_reproduction\n"
        "replication_index: 2\n"
        "_replicate_of: idea-root\n",
    ]

    receipt = _audit(
        tmp_path,
        monkeypatch,
        configs,
        contract=_not_applicable(configs),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "declared_replicas_without_question"
    assert receipt["checks"]["declared_replicas_preregistered"][
        "passed"
    ] is False
    assert receipt["checks"]["declared_replicas_preregistered"][
        "unpreregistered_idea_ids"
    ] == IDEA_IDS


def test_not_applicable_rejects_metadata_relabelled_duplicate_configs(
        tmp_path, monkeypatch):
    configs = [
        "training:\n  seed: 7\ntitle: first label\n",
        "training:\n  seed: 7\ntitle: changed label\n"
        "hypothesis: changed prose only\n",
    ]

    receipt = _audit(
        tmp_path,
        monkeypatch,
        configs,
        contract=_not_applicable(configs),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "exact_duplicate_configs_detected"
    assert receipt["checks"]["no_exact_duplicate_configs"] == {
        "passed": False,
        "idea_id_groups": [IDEA_IDS],
    }


def test_group_verifies_only_declared_seed_and_metric_tolerance(
        tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        [
            "model: base\ntraining:\n  seed: 1\n  epochs: 2\n",
            "model: base\ntraining:\n  seed: 2\n  epochs: 2\n",
        ],
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["groups"][0]["only_declared_variables_changed"] is True
    assert receipt["groups"][0]["metric_delta"] == pytest.approx(0.01)


def test_group_preregisters_explicit_replica_metadata(tmp_path, monkeypatch):
    configs = [
        "model: base\ntraining:\n  seed: 1\n  epochs: 2\n"
        "replication_role: seed_reproduction\nreplication_index: 1\n"
        "_replicate_of: idea-root\n",
        "model: base\ntraining:\n  seed: 2\n  epochs: 2\n"
        "replication_role: seed_reproduction\nreplication_index: 2\n"
        "_replicate_of: idea-root\n",
    ]

    receipt = _audit(tmp_path, monkeypatch, configs)

    assert receipt["status"] == "VERIFIED"
    assert receipt["checks"]["declared_replicas_preregistered"] == {
        "passed": True,
        "declared_idea_ids": IDEA_IDS,
        "unpreregistered_idea_ids": [],
    }


def test_group_rejects_explicit_replica_omitted_from_questions(
        tmp_path, monkeypatch):
    idea_ids = ["idea-replica-a", "idea-replica-b", "idea-control-c"]
    configs = [
        "training:\n  seed: 1\nreplication_role: seed_reproduction\n"
        "replication_index: 1\n_replicate_of: idea-root\n",
        "training:\n  seed: 2\nreplication_role: seed_reproduction\n"
        "replication_index: 2\n_replicate_of: idea-root\n",
        "training:\n  seed: 3\n",
    ]
    contract = {
        "mode": "groups",
        "expected_config_identity_sha256": _identities(
            configs, idea_ids
        ),
        "groups": [{
            "question": (
                "Whether the declared seed change reproduces the metric."
            ),
            "idea_ids": [idea_ids[0], idea_ids[2]],
            "varying_config_paths": ["training.seed"],
            "max_absolute_metric_delta": 0.02,
        }],
    }

    receipt = _audit(
        tmp_path,
        monkeypatch,
        configs,
        contract=contract,
        idea_ids=idea_ids,
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["declared_replicas_preregistered"] == {
        "passed": False,
        "declared_idea_ids": idea_ids[:2],
        "unpreregistered_idea_ids": [idea_ids[1]],
    }


@pytest.mark.parametrize(
    "configs",
    [
        [
            "model: base\ntraining:\n  seed: 1\n  epochs: 2\n",
            "model: changed\ntraining:\n  seed: 2\n  epochs: 2\n",
        ],
        [
            "model: base\ntraining:\n  seed: 1\n  epochs: 2\n",
            "model: changed\ntraining:\n  seed: 1\n  epochs: 2\n",
        ],
    ],
)
def test_group_rejects_undeclared_drift_or_nonvarying_declared_path(
        tmp_path, monkeypatch, configs):
    receipt = _audit(tmp_path, monkeypatch, configs)

    assert receipt["status"] == "FAILED"
    assert receipt["groups"][0]["only_declared_variables_changed"] is False


def test_group_rejects_metric_delta_above_preregistered_tolerance(
        tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        ["training:\n  seed: 1\n", "training:\n  seed: 2\n"],
        values=dict(zip(IDEA_IDS, [0.80, 0.83])),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["groups"][0]["passed"] is False


def test_terminal_failure_is_complete_but_fails_target(tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        ["training:\n  seed: 1\n", "training:\n  seed: 2\n"],
        terminal_states=["COMPLETE", "FAILED"],
    )

    assert receipt["status"] == "FAILED"
    assert receipt["checks"]["group_evidence_complete"]["passed"] is True
    assert receipt["groups"][0]["all_complete"] is False


def test_missing_complete_metric_is_unverified(tmp_path, monkeypatch):
    receipt = _audit(
        tmp_path,
        monkeypatch,
        ["training:\n  seed: 1\n", "training:\n  seed: 2\n"],
        values={IDEA_IDS[0]: 0.80},
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["checks"]["group_evidence_complete"]["passed"] is False


def test_direct_config_tampering_is_unverified(tmp_path, monkeypatch):
    db_path = _lake(
        tmp_path,
        ["training:\n  seed: 1\n", "training:\n  seed: 2\n"],
    )
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE ideas SET config = ? WHERE idea_id = ?",
        ("training:\n  seed: 999\n", IDEA_IDS[0]),
    )
    connection.commit()
    connection.close()

    receipt = audit_campaign_reproducibility(
        db_path,
        tmp_path / "results",
        {"report": {"primary_metric": "score"}},
        expected_idea_ids=list(IDEA_IDS),
        contract=_group_contract(configs=[
            "training:\n  seed: 1\n", "training:\n  seed: 2\n",
        ]),
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "reproducibility_config_integrity_invalid"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda contract: contract["groups"][0].update(
                {"max_absolute_metric_delta": float("nan")}
            ),
            "reproducibility_group_tolerance_invalid",
        ),
        (
            lambda contract: contract["groups"][0].update(
                {"varying_config_paths": ["training..seed"]}
            ),
            "reproducibility_group_paths_invalid",
        ),
        (
            lambda contract: contract["groups"][0].update(
                {"idea_ids": [IDEA_IDS[0], "idea-outside"]}
            ),
            "reproducibility_group_idea_ids_invalid",
        ),
    ],
)
def test_contract_validation_rejects_malformed_groups(mutation, reason):
    contract = _group_contract()
    mutation(contract)

    assert validate_reproducibility_contract(contract, IDEA_IDS) == reason


def test_recomputed_database_hashes_cannot_change_preregistered_config(
        tmp_path, monkeypatch):
    configs = ["training:\n  seed: 1\n", "training:\n  seed: 2\n"]
    contract = _group_contract(configs=configs)
    db_path = _lake(tmp_path, configs)
    changed = "training:\n  seed: 999\n"
    changed_config = yaml.safe_load(changed)
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE ideas SET config = ? WHERE idea_id = ?",
        (changed, IDEA_IDS[0]),
    )
    # Simulate a stronger database editor that also repairs both derived
    # integrity columns after the config-mutation trigger cleared them.
    connection.execute(
        "UPDATE ideas SET config_hash = ?, config_source_sha256 = ? "
        "WHERE idea_id = ?",
        (
            reproduction_module.hash_config(changed_config),
            reproduction_module.hashlib.sha256(changed.encode()).hexdigest(),
            IDEA_IDS[0],
        ),
    )
    connection.commit()
    connection.close()

    receipt = audit_campaign_reproducibility(
        db_path,
        tmp_path / "results",
        {"report": {"primary_metric": "score"}},
        expected_idea_ids=list(IDEA_IDS),
        contract=contract,
    )

    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "reproducibility_config_identity_mismatch"
