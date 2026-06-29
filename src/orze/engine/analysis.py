"""Analysis SOP (Standard Operating Procedure) - Data/model analysis workflow.

Simple analysis: load data, compute statistics, save results.
No training, no GPU required, pure compute-only.

Demonstrates non-training SOP type support (v4.5+).
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("orze.analysis")


class AnalysisProcess:
    """Represents an active analysis process."""

    def __init__(
        self,
        idea_id: str,
        gpu: int,  # unused for analysis, but kept for interface compatibility
        process,
        start_time: float,
        log_path: Path,
        timeout: int,
    ):
        self.idea_id = idea_id
        self.gpu = gpu
        self.process = process
        self.start_time = start_time
        self.log_path = log_path
        self.timeout = timeout

    def poll(self) -> Optional[int]:
        """Check if process is still running."""
        return self.process.poll()

    def terminate(self, timeout: int = 10) -> None:
        """Terminate the process."""
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except:
            self.process.kill()


def launch_analysis(
    idea_id: str,
    gpu: int,
    results_dir: Path,
    cfg: Dict[str, Any],
    lake=None,
) -> Optional[AnalysisProcess]:
    """Launch an analysis process.

    Args:
        idea_id: Analysis idea ID
        gpu: GPU index (unused for analysis, for interface compatibility)
        results_dir: Results directory
        cfg: Config dict (requires 'analysis_script')
        lake: IdeaLake instance for FSM transitions (optional)

    Returns:
        AnalysisProcess if launched, None if skipped

    Records FSM transition: CLAIMED → IN_PROGRESS
    """
    analysis_script = cfg.get("analysis_script")
    if not analysis_script:
        logger.info("No analysis_script in config for %s, skipping", idea_id)
        return None

    # Check if already done
    results_path = results_dir / idea_id / "analysis_results.json"
    if results_path.exists():
        logger.info("[ANALYSIS_SKIP] %s already completed", idea_id)
        return None

    python = cfg.get("python", sys.executable)
    analysis_args = cfg.get("analysis_args") or []
    analysis_timeout = cfg.get("analysis_timeout", 3600)

    cmd = [python, analysis_script]
    cmd.extend(analysis_args)

    log_path = results_dir / idea_id / "analysis_output.log"
    logger.info("Launching analysis for %s", idea_id)

    try:
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            log_fh.close()
            raise

        # Record FSM transition: CLAIMED → IN_PROGRESS
        if lake:
            try:
                lake.record_state_transition(
                    idea_id,
                    from_state="CLAIMED",
                    to_state="IN_PROGRESS",
                    reason="analysis_launched",
                    host=socket.gethostname(),
                    pid=os.getpid(),
                    sop_type=cfg.get("sop", "analysis"),
                )
            except Exception as e:
                logger.warning("FSM transition failed (non-blocking): %s", e)

        return AnalysisProcess(
            idea_id=idea_id,
            gpu=gpu,
            process=proc,
            start_time=time.time(),
            log_path=log_path,
            timeout=analysis_timeout,
        )
    except Exception as e:
        logger.warning("Failed to launch analysis for %s: %s", idea_id, e)
        return None


def check_active_analysis(
    active_analysis: Dict[int, AnalysisProcess],
    results_dir: Path,
    cfg: Dict[str, Any],
    lake=None,
) -> List[Tuple[str, int]]:
    """Check running analysis processes.

    Args:
        active_analysis: Dict mapping gpu -> AnalysisProcess (mutated in-place)
        results_dir: Results directory
        cfg: Config dict
        lake: IdeaLake instance for FSM transitions (optional)

    Returns:
        List of (idea_id, gpu) for processes that finished

    Records FSM transition: IN_PROGRESS → COMPLETE or IN_PROGRESS → FAILED
    """
    finished = []
    analysis_results_file = cfg.get("analysis_results_file", "analysis_results.json")

    for gpu in list(active_analysis.keys()):
        ap = active_analysis[gpu]
        ret = ap.process.poll()
        elapsed = time.time() - ap.start_time

        if ret is None:
            # Still running - check timeout
            if elapsed > ap.timeout:
                logger.warning("[ANALYSIS TIMEOUT] %s after %.0fm — killing",
                              ap.idea_id, elapsed / 60)
                ap.process.terminate()
                ap.process.wait(timeout=10)
                _write_analysis_failure_marker(
                    results_dir, ap.idea_id, analysis_results_file,
                    f"Timed out after {elapsed/60:.0f}m", lake=lake
                )
                del active_analysis[gpu]
                finished.append((ap.idea_id, gpu))
            continue

        # Process exited
        if ret == 0:
            logger.info("[ANALYSIS OK] %s in %.1fm", ap.idea_id, elapsed / 60)
            # Check if results file was created
            results_path = results_dir / ap.idea_id / analysis_results_file
            if results_path.exists():
                # Record FSM transition: IN_PROGRESS → COMPLETE
                if lake:
                    try:
                        lake.record_state_transition(
                            ap.idea_id,
                            from_state="IN_PROGRESS",
                            to_state="COMPLETE",
                            reason="analysis completed",
                            host=socket.gethostname(),
                            pid=os.getpid(),
                            sop_type=cfg.get("sop", "analysis"),
                        )
                    except Exception as e:
                        logger.warning("FSM transition failed (non-blocking): %s", e)
            else:
                # Process succeeded but no results file
                _write_analysis_failure_marker(
                    results_dir, ap.idea_id, analysis_results_file,
                    "Exit code 0 but no results file", lake=lake
                )
        else:
            # Process failed
            logger.warning("[ANALYSIS FAILED] %s exit code %d", ap.idea_id, ret)
            _write_analysis_failure_marker(
                results_dir, ap.idea_id, analysis_results_file,
                f"Exit code {ret}", lake=lake
            )

        del active_analysis[gpu]
        finished.append((ap.idea_id, gpu))

    return finished


def _write_analysis_failure_marker(
    results_dir: Path,
    idea_id: str,
    results_file: str,
    reason: str,
    lake=None,
) -> None:
    """Write analysis failure marker and record FSM transition."""
    results_path = results_dir / idea_id / results_file
    if results_path.exists():
        return

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "status": "FAILED",
                "reason": reason[:500],
            },
            indent=2,
        )
    )
    logger.info("Wrote analysis failure marker for %s", idea_id)

    # Record FSM transition: IN_PROGRESS → FAILED
    if lake:
        try:
            lake.record_state_transition(
                idea_id,
                from_state="IN_PROGRESS",
                to_state="FAILED",
                reason=reason[:100],
                host=socket.gethostname(),
                pid=os.getpid(),
                sop_type=cfg.get("sop", "analysis"),
            )
        except Exception as e:
            logger.warning("FSM transition failed (non-blocking): %s", e)
