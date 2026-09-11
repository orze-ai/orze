from .process import TrainingProcess, EvalProcess, RoleProcess
from .scheduler import claim, get_unclaimed, cleanup_orphans, run_cleanup
from .launcher import launch, check_active
from .evaluator import launch_eval, check_active_evals, run_eval, run_post_scripts
from .health import check_stalled, detect_fatal_in_log, check_disk_space
from .roles import check_active_roles
from .failure import get_skipped_ideas

# Lazy import to avoid circular dependency with orze-pro
# Use: from orze.engine.orchestrator import Orze
def __getattr__(name):
    if name == "Orze":
        from .orchestrator import Orze
        return Orze
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
