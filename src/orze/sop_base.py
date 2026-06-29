"""Abstract base class for SOP (Standard Operating Procedure) implementations.

All concrete SOP types (training, analysis, etc.) inherit from this class
and implement the abstract methods. This ensures consistent interface and
error handling across all workflow types.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("orze")


class SOPProcess:
    """Represents an active process running a SOP step."""

    def __init__(
        self,
        idea_id: str,
        gpu: int,
        process,
        start_time: float,
        log_path: Path,
        timeout: int,
    ):
        """
        Args:
            idea_id: Research idea identifier
            gpu: GPU device index
            process: subprocess.Popen object or equivalent
            start_time: When the process started (time.time())
            log_path: Path to log file
            timeout: Timeout in seconds
        """
        self.idea_id = idea_id
        self.gpu = gpu
        self.process = process
        self.start_time = start_time
        self.log_path = log_path
        self.timeout = timeout

    def poll(self) -> Optional[int]:
        """Check if process is still running. Returns exit code or None."""
        return self.process.poll()

    def terminate(self, timeout: int = 10) -> None:
        """Terminate the process."""
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except:
            self.process.kill()


class BaseSOP(ABC):
    """Abstract base class for all SOP implementations.

    Concrete SOPs override these methods to implement their specific workflows.
    The base class handles common patterns like error logging and FSM transitions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """SOP type name (e.g., 'training', 'analysis')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this SOP."""
        pass

    @abstractmethod
    def launch(
        self,
        idea_id: str,
        gpu: int,
        results_dir: Path,
        cfg: Dict[str, Any],
        lake=None,
    ) -> Optional[SOPProcess]:
        """Launch a work process for this SOP.

        Args:
            idea_id: Research idea ID
            gpu: GPU device index
            results_dir: Directory for results
            cfg: Configuration dict
            lake: IdeaLake instance for FSM transitions (optional)

        Returns:
            SOPProcess if successful, None if skipped (already done, etc.)

        Should record FSM transition: CLAIMED → IN_PROGRESS
        """
        pass

    @abstractmethod
    def check_active(
        self,
        active_processes: Dict[int, SOPProcess],
        results_dir: Path,
        cfg: Dict[str, Any],
        lake=None,
    ) -> List[Tuple[str, int]]:
        """Check running processes and return finished ones.

        Args:
            active_processes: Dict mapping gpu -> SOPProcess
            results_dir: Directory for results
            cfg: Configuration dict
            lake: IdeaLake instance for FSM transitions (optional)

        Returns:
            List of (idea_id, gpu) for processes that finished this cycle
            Mutates active_processes dict (removes finished entries)

        Should record FSM transition: IN_PROGRESS → COMPLETE or FAILED
        """
        pass

    @property
    def substates(self) -> List[str]:
        """Optional detailed state tracking for this SOP.

        E.g., training SOP might have substates=['training', 'evaluating']
        for finer-grained tracking beyond just IN_PROGRESS.

        Return empty list if no substates needed.
        """
        return []

    # Helper methods for common patterns

    def _write_failure_marker(
        self,
        results_dir: Path,
        idea_id: str,
        failure_file: str,
        reason: str,
    ) -> None:
        """Write a failure marker file.

        This is a common pattern: write a file indicating why work failed.
        Subclasses can call this or override with SOP-specific logic.
        """
        import json

        marker_path = results_dir / idea_id / failure_file
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "reason": reason[:500],
                },
                indent=2,
            )
        )

    def _record_fsm_transition(
        self,
        lake,
        idea_id: str,
        from_state: str,
        to_state: str,
        reason: str,
        host: str,
        pid: int,
    ) -> None:
        """Record an FSM transition (wrapper for common pattern).

        Non-blocking: logs warnings but never crashes.
        """
        if not lake:
            return

        try:
            lake.record_state_transition(
                idea_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                host=host,
                pid=pid,
            )
        except Exception as e:
            logger.warning("FSM transition failed (non-blocking) for %s: %s", idea_id, e)

    @staticmethod
    def close_process_log(process: SOPProcess) -> None:
        """Close the log file handle for a process if it has one."""
        if hasattr(process, "_log_fh"):
            try:
                process._log_fh.close()
            except:
                pass
