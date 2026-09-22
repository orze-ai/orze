"""Validate complete Python proposals without running generated code."""
import ast


def parse_python_proposal(text, *, complete, max_bytes=16000):
    """Accept raw Python or one Python fence, for a method or a diagnostic.

    ``complete`` must come from the provider's completion state, not from the
    apparent syntax of its text. This is a format check, not a security sandbox:
    the returned source must still execute in the caller's isolated worker.
    """
    if complete is not True:
        raise ValueError("provider response is incomplete")
    if not isinstance(text, str):
        raise ValueError("proposal must be Python text")
    source = text.strip()
    if source.startswith("```"):
        lines = source.splitlines()
        if lines[0] not in ("```python", "```py", "```") or lines[-1] != "```":
            raise ValueError("expected one complete Python fence")
        source = "\n".join(lines[1:-1]).strip()
    if not source or len(source.encode("utf-8")) > max_bytes:
        raise ValueError("proposal source is empty or oversized")
    tree = ast.parse(source, filename="candidate.py")
    compile(tree, "candidate.py", "exec")
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    entrypoints = {"build", "train", "predict", "analyze"}
    if any(functions.count(name) > 1 for name in entrypoints):
        raise ValueError("duplicate research entrypoint")
    names = set(functions) & entrypoints
    if names == {"build", "train", "predict"}:
        kind = "method"
    elif names == {"analyze"}:
        kind = "analysis"
    else:
        raise ValueError("define build/train/predict or analyze")
    return {"kind": kind, "source": source}
