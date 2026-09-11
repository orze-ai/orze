"""Fixed native resolver failure policy, including confirmed non-execution."""
from orze.engine.cpu_admission_failure_report import FailurePolicy, report_failure

PHASE = "artifact_preflight_failure_report"


def report_artifact_preflight_failure(lake, idea_dir, ref, failure_counts, cfg):
    from orze.engine.native_artifact_preflight import ArtifactPreflightHOLD, _cached, _scope
    policy = FailurePolicy(
        "artifact_preflight", ArtifactPreflightHOLD, _scope, _cached,
        "Artifact preflight failed; training was not launched",
        ("TERMINAL", "NOT_STARTED"))
    return report_failure(lake, idea_dir, ref, failure_counts, cfg, policy=policy)
