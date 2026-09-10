    def _sync_pipeline_for_global_transition(
        self,
        idea_id: str,
        from_state: str,
        to_state: str,
        reason: str,
        host: str,
        pid: int,
        sop_type: str,
        at: str,
    ) -> bool:
        """Keep stage truth atomic with lifecycle launch/terminal/retry edges."""
        if sop_type != "training":
            return True

        def move(stage: str, target: str, stage_reason: str) -> bool:
            current = self._stage_state_in_tx(idea_id, stage)
            if current == target:
                return True
            return self._record_stage_transition_in_tx(
                idea_id, stage, current, target, stage_reason, host, pid, at)

        if from_state == "CLAIMED" and to_state == "IN_PROGRESS":
            return (
                move("training", "IN_PROGRESS", reason)
                and move("evaluation", "PENDING", "pipeline_initialized")
            )
        if to_state == "QUEUED":
            return all(
                move(stage, "PENDING", "pipeline_reset_for_retry")
                for stage in PIPELINE_STAGES
                if self._stage_state_in_tx(idea_id, stage) != "NOT_STARTED"
            )
        if to_state == "FAILED":
            evaluation = self._stage_state_in_tx(idea_id, "evaluation")
            if evaluation == "IN_PROGRESS":
                return move("evaluation", "FAILED", reason)
            training = self._stage_state_in_tx(idea_id, "training")
            ok = True
            if training not in ("COMPLETE", "FAILED"):
                ok = move("training", "FAILED", reason)
            evaluation = self._stage_state_in_tx(idea_id, "evaluation")
            if evaluation not in ("COMPLETE", "FAILED", "SKIPPED"):
                ok = move(
                    "evaluation", "SKIPPED", "training_did_not_complete") and ok
            return ok
        if to_state == "COMPLETE":
            evaluation = self._stage_state_in_tx(idea_id, "evaluation")
            if evaluation == "IN_PROGRESS":
                return move("evaluation", "COMPLETE", reason)
            training = self._stage_state_in_tx(idea_id, "training")
            ok = True
            if training == "IN_PROGRESS":
                ok = move("training", "COMPLETE", reason)
            evaluation = self._stage_state_in_tx(idea_id, "evaluation")
            if evaluation in ("NOT_STARTED", "PENDING"):
                ok = move(
                    "evaluation", "SKIPPED", "no_evaluation_configured") and ok
            return ok
        return True
