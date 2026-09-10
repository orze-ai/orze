"""One native result conclusion, before trigger settlement and parent usage."""
from orze.core.research_result import make_result, read_native_result


def native_role_outcome(rp, exit_code, *, cleanup_verified=True,
                        rate_limited=False, forced_reason=None):
    """Return the legacy operational bucket and retain the exact details.

    Caller invokes this only for an explicitly native dispatch. A malformed
    reference is still native and never falls back to shared-file heuristics.
    Process failure, quota and timeout cannot be upgraded by a good sidecar.
    """
    if not cleanup_verified:
        result = make_result("error", "process_cleanup_unconfirmed")
        bucket = "error"
    elif forced_reason is not None:
        result = make_result("error", forced_reason)
        bucket = "timeout" if forced_reason == "process_timeout" else "error"
    elif rate_limited:
        result = make_result("error", "provider_rate_limited")
        bucket = "rate_limited"
    else:
        try:
            ref = rp.native_result_ref
            if (not isinstance(ref, dict) or ref.get("role_name") != rp.role_name
                    or (getattr(rp, "trigger_launch", None) is not None
                        and (ref.get("attempt_id") != rp.trigger_launch.get("attempt_id")
                             or ref.get("nonce_sha256") != rp.trigger_launch.get("nonce_sha256")))):
                raise ValueError("native_result_process_binding_invalid")
            result = read_native_result(ref, process_nonce=rp.process_nonce)
        except (OSError, ValueError, TypeError, RecursionError):
            result = make_result("error", "native_result_unavailable")
        if exit_code != 0:
            if result["status"] != "error":
                result = make_result("error", "process_exit_nonzero")
            bucket = "error"
        else:
            bucket = (
                "ok" if result["status"] in {"accepted", "partial"}
                else "error" if result["status"] == "error"
                else "soft_failure"
            )
    rp.native_result_details = result
    return bucket
