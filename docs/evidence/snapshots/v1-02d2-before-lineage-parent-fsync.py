def publish_model_lineage_finalization(
    prepared: PreparedModelLineageFinalization, tp, idea_dir: Path, cfg: Mapping,
) -> dict:
    """Publish a compact envelope after current-attempt checks by the caller.

    Does not rehash model/manifest contents. Any uncertain post-write exception
    must make the caller HOLD its effect intent; no rollback/replay is claimed.
    """
    if (type(prepared) is not PreparedModelLineageFinalization
            or not isinstance(cfg, Mapping) or validate_model_lineage_config(cfg)
            or type(prepared.payload_json) is not str
            or len(prepared.payload_json.encode("utf-8")) > _MAX_FINALIZATION_PAYLOAD_BYTES):
        raise ModelLineageError("model_lineage_preparation_invalid")
    idea_dir = Path(idea_dir).absolute()
    if str(idea_dir) != prepared.idea_dir or _finalization_policy(cfg) != prepared.policy_sha256:
        raise ModelLineageError("model_lineage_publication_scope_changed")
    try:
        payload = json.loads(prepared.payload_json)
    except (ValueError, UnicodeError) as exc:
        raise ModelLineageError("model_lineage_preparation_invalid") from exc
    if not isinstance(payload, dict) or _canonical_hash(payload) != prepared.payload_sha256:
        raise ModelLineageError("model_lineage_preparation_invalid")
    if not cfg.get("model_lineage", {}).get("enabled", False):
        if payload != {"status": "disabled"}:
            raise ModelLineageError("model_lineage_preparation_invalid")
        return payload
    if set(payload) != _LINEAGE_KEYS or payload.get("artifact_kind") != "file":
        raise ModelLineageError("model_lineage_preparation_invalid")
    paths = _finalization_paths(tp, idea_dir, cfg)

    def verify():
        if (prepared.subject != _finalization_subject(tp, idea_dir)
                or prepared.policy_sha256 != _finalization_policy(cfg)
                or (prepared.files, prepared.parents) != _publication_bindings(paths)):
            raise ModelLineageError("model_lineage_publication_input_changed")

    verify()
    output = _idea_path(idea_dir, LINEAGE_FILE)
    _write_envelope_once(output, payload)
    actual, digest = _read_envelope(output, _LINEAGE_KEYS)
    if actual != payload or digest != prepared.payload_sha256:
        raise ModelLineageError("model_lineage_publication_readback_failed")
    verify()
    return payload
