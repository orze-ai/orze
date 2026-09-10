"""External draft: credentials are capability, not permission to enable roles."""
import pytest

from orze.core.config import load_project_config


@pytest.mark.parametrize("case", [
    "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "no-key-control", "explicit-legacy-control",
])
def test_project_loader_never_invents_work_from_an_ambient_credential(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "orze.yaml"
    if case == "explicit-legacy-control":
        monkeypatch.setenv("OPENAI_API_KEY", "TEST_ONLY_NOT_A_CREDENTIAL")
        path.write_text("research:\n  mode: script\n  script: explicit-worker.py\n", encoding="utf-8")
    else:
        if case != "no-key-control":
            monkeypatch.setenv(case, "TEST_ONLY_NOT_A_CREDENTIAL")
        path.write_text("roles: {}\n", encoding="utf-8")
    cfg = load_project_config(str(path))
    if case == "explicit-legacy-control":
        assert cfg["roles"] == {"research": {"mode": "script", "script": "explicit-worker.py"}}
    else:
        assert cfg["roles"] == {}, "finding a credential cannot opt the project into paid research work"
