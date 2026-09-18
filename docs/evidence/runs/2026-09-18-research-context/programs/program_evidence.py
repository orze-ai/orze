"""Keep a recorded program beside its execution result in bounded context.

The application qualifies the history before calling this helper and chooses
its leader. Findings are program-authored claims, not scientific certification.
No prose summary, inferred model label or mandatory research stage is added.
"""
import copy
import json

from .source_edits import historical_program, parent_program


def context(history, leader_id, max_bytes=48000):
    """Return complete records for latest action, latest valid method and leader.

    Overlapping roles share a record. Capacity omissions are explicit; source
    and facts are never independently cut or summarized. The caller must also
    enforce its full prompt limit, including instructions and other context.
    """
    if type(max_bytes) is not int or max_bytes < 128:
        raise ValueError("provide an integer context limit of at least 128 bytes")
    parent_program(history, leader_id)
    programs = [row for row in history if row["action"]["kind"] in {"method", "analyze"}]
    recent = programs[-1]
    valid = next(row for row in reversed(programs)
                 if row["valid"] and row["action"]["kind"] == "method")
    leader = next(row for row in programs if row["action_id"] == leader_id)
    choices = {}
    for role, row in [("latest_action", recent), ("latest_valid_method", valid),
                      ("development_leader", leader)]:
        task = row["task_id"]
        if task not in choices:
            choices[task] = {"roles": [], "task_id": task,
                "action_id": row["action_id"], "valid": row["valid"],
                "action": historical_program(history, row["action_id"]),
                "facts": copy.deepcopy(row["facts"])}
        choices[task]["roles"].append(role)
    value = {"records": [], "omitted_task_ids": list(choices),
             "meaning": "recorded_source_and_execution; findings_require_interpretation"}
    encode = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
    if len(encode(value).encode()) > max_bytes:
        raise ValueError("context limit cannot hold omission references")
    for task, record in choices.items():
        candidate = copy.deepcopy(value)
        candidate["records"].append(record)
        candidate["omitted_task_ids"].remove(task)
        if len(encode(candidate).encode()) <= max_bytes:
            value = candidate
    return encode(value)
