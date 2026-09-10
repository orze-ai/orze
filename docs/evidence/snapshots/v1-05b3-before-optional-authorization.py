def begin(lake, tp, idea_dir, cfg=None):
    """Persist a native intent before Popen, without inventing a started stage."""
    from orze.core.artifact_contract import artifact_publication_binding
    artifact_binding = artifact_publication_binding(
        cfg or {}, idea_dir, getattr(tp, "execution_identity", None))
    with execution_transaction(lake, idea_dir) as tx:
        from orze.engine.execution_catalog import bind_catalog
        bind_catalog(lake, idea_dir, tx.lease)
        _, claim_sha = _claim(tp, idea_dir, lake)
        launch_state = _launch_state(lake, tp.idea_id)
        from orze.engine.replication import replication_authorization
        replication = replication_authorization(
            lake, tp.idea_id, idea_dir, cfg or {}, getattr(tp, "execution_identity", None),
            claim_id=tp.attempt_id)
        if not canonical_identity_equal(replication, getattr(tp, "replication_authorization", None)):
            raise AttemptEffectBusy("training_replication_authorization_changed")
        binding = {
            "origin": "native_training", "claim_sha256": claim_sha,
            "launch_lifecycle": launch_state,
        }
        if artifact_binding is not None:
            binding["artifact_publication"] = artifact_binding
        if replication is not None:
            binding["replication"] = replication
        ref = create_attempt(tx.conn, tp.idea_id, "training", tp.attempt_id, binding)
        if not canonical_identity_equal(_launch_state(lake, tp.idea_id), launch_state):
            raise AttemptAuthorityError("training_launch_lifecycle_changed")
        if replication is not None and not canonical_identity_equal(
                replication_authorization(lake, tp.idea_id, idea_dir, cfg or {},
                                          tp.execution_identity, claim_id=tp.attempt_id), replication):
            raise AttemptEffectBusy("training_replication_authorization_changed")
        tx.watch_attempt(ref)
    return ref
