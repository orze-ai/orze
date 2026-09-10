def request_evaluation_retry(idea_id: str, results_dir: Path,
                             cfg: dict, lake) -> dict:
    """Serialize retry file moves with current-attempt terminal publication."""
    from orze.core.execution_attempts import current_attempt
    from orze.core.ideas import IDEA_ID_PATTERN
    from orze.engine.attempt_effect_lock import (
        AttemptEffectBusy, AttemptEffectInDoubt, attempt_effect_lock,
    )
    from orze.engine.termination_hold import TerminationUnconfirmed
    import re

    if (not isinstance(idea_id, str) or len(idea_id) > 128
            or re.fullmatch(IDEA_ID_PATTERN, idea_id) is None):
        raise EvaluationRetryError("evaluation_retry_idea_id_invalid")
    # Static path/allowlist rejection must not create even a guard namespace.
    # The same policy is checked again under the guard before actual moves.
    safe_file(Path(results_dir) / idea_id, "metrics.json")
    retry_file_policy(Path(results_dir) / idea_id, cfg)
    from orze.engine.evaluation_retry_files import _safe_directory
    _safe_directory(Path(results_dir) / idea_id / "_evaluation_retries")
    try:
        with attempt_effect_lock(Path(results_dir) / idea_id):
            current = current_attempt(lake.conn, idea_id, "evaluation")
            if current is not None and current["state"] not in ("TERMINAL", "NOT_STARTED"):
                raise EvaluationRetryError("evaluation_retry_attempt_not_closed")
            return _request_evaluation_retry(idea_id, results_dir, cfg, lake)
    except (AttemptEffectBusy, AttemptEffectInDoubt):
        raise
    except TerminationUnconfirmed as exc:
        raise EvaluationRetryError("evaluation_retry_termination_unconfirmed") from exc
