"""Compatibility entry for the existing once-only pre-script failure action.

The shared transaction preserves the action ID, receipt/metrics fields, fixed
messages, zero-allocation reason and HOLD/stale semantics. Native pre-script
still accepts only its confirmed TERMINAL source, never NOT_STARTED.
"""
from orze.engine.cpu_admission_failure_report import FailurePolicy, report_failure
from orze.engine.native_pre_script import PreScriptHOLD, _cached, _scope

PHASE = "pre_script_failure_report"
_POLICY = FailurePolicy("pre_script", PreScriptHOLD, _scope, _cached, "Pre-script failed")


def report_pre_script_failure(lake, idea_dir, ref, failure_counts, cfg):
    return report_failure(lake, idea_dir, ref, failure_counts, cfg, policy=_POLICY)
