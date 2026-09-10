def _dependencies(conn, ref, binding, records):
    artifacts = {}
    for artifact_id in binding["input_artifact_ids"]:
        artifact = get_artifact(conn, artifact_id)
        if (artifact is None or artifact["scope"] != binding["scope"]
                or artifact["spec_fingerprint"] != binding["spec_fingerprint"]):
            raise ResearchObservationError("observation_input_artifact_mismatch")
        artifacts[artifact_id] = artifact
    for record in records:
        for artifact_id in record["result_artifact_ids"]:
            artifact = get_artifact(conn, artifact_id)
            if (artifact is None or artifact["scope"] != binding["scope"]
                    or artifact["producer"] != asdict(ref)):
                raise ResearchObservationError("observation_result_artifact_mismatch")
            artifacts[artifact_id] = artifact
    return artifacts
