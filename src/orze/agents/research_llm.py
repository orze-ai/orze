"""LLM backend implementations for the orze research agent.

CALLING SPEC:
    call_llm(prompt, backend, api_key="", model="", endpoint="") -> str
        Route to the appropriate LLM backend and return response text.
        Supported backends: gemini, openai, anthropic, ollama, custom.

    call_gemini(prompt, api_key, model="gemini-2.5-flash", ...) -> str
    call_openai(prompt, api_key, model="gpt-4o", ...) -> str
    call_anthropic(prompt, api_key, model="claude-sonnet-4-5-20250929", ...) -> str
    call_ollama(prompt, model="llama3", ...) -> str

    DEFAULT_SYSTEM_PROMPT: str
        Base system prompt for the research agent LLM.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger("orze.research_agent")


# ---------------------------------------------------------------------------
#  LLM backends
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict,
               timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_gemini(prompt: str, api_key: str,
                model: str = "gemini-2.5-flash",
                max_tokens: int = 8192,
                web_search: bool = False) -> str:
    """Call Gemini API. Tries multiple models on failure.

    Args:
        web_search: Enable Google Search grounding so Gemini can fetch
            live web results alongside its own knowledge.
    """
    models = [model, "gemini-3-flash-preview", "gemini-2.5-flash"]
    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    for m in unique_models:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{m}:generateContent?key={api_key}")
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            if web_search:
                payload["tools"] = [{"google_search": {}}]
            result = _post_json(url, payload, {"Content-Type": "application/json"})
            parts = (result.get("candidates", [{}])[0]
                     .get("content", {}).get("parts", []))
            text = "\n".join(p.get("text", "") for p in (parts or []) if p.get("text"))
            if text:
                logger.info("Gemini (%s) returned %d chars", m, len(text))
                return text
            logger.warning("Gemini (%s) returned empty response", m)
        except Exception as e:
            logger.warning("Gemini (%s) failed: %s", m, e)
    return ""


def call_openai(prompt: str, api_key: str,
                model: str = "gpt-4o",
                max_tokens: int = 8192,
                endpoint: str = "https://api.openai.com/v1") -> str:
    """Call OpenAI-compatible API (OpenAI, Azure, vLLM, etc.)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        result = _post_json(url, payload, headers)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("OpenAI (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("OpenAI (%s) failed: %s", model, e)
        return ""


def call_anthropic(prompt: str, api_key: str,
                   model: str = "claude-sonnet-4-5-20250929",
                   max_tokens: int = 8192) -> str:
    """Call Anthropic API directly."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        result = _post_json(url, payload, headers)
        text = "".join(
            b.get("text", "") for b in result.get("content", [])
            if b.get("type") == "text"
        )
        logger.info("Anthropic (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Anthropic (%s) failed: %s", model, e)
        return ""


def call_ollama(prompt: str, model: str = "llama3",
                endpoint: str = "http://localhost:11434") -> str:
    """Call local Ollama instance."""
    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        result = _post_json(url, payload, {"Content-Type": "application/json"},
                            timeout=300)
        text = result.get("response", "")
        logger.info("Ollama (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Ollama (%s) failed: %s", model, e)
        return ""


def call_llm(prompt: str, backend: str, api_key: str = "",
             model: str = "", endpoint: str = "") -> str:
    """Route to the appropriate LLM backend."""
    if backend == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            logger.error("GEMINI_API_KEY not set")
            return ""
        return call_gemini(prompt, key, model=model or "gemini-2.5-flash",
                          web_search=True)

    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.error("OPENAI_API_KEY not set")
            return ""
        return call_openai(prompt, key, model=model or "gpt-4o",
                           endpoint=endpoint or "https://api.openai.com/v1")

    elif backend == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logger.error("ANTHROPIC_API_KEY not set")
            return ""
        return call_anthropic(prompt, key,
                              model=model or "claude-sonnet-4-5-20250929")

    elif backend == "ollama":
        return call_ollama(prompt, model=model or "llama3",
                           endpoint=endpoint or "http://localhost:11434")

    elif backend == "custom":
        # OpenAI-compatible custom endpoint
        key = api_key or os.environ.get("LLM_API_KEY", "")
        return call_openai(prompt, key, model=model or "default",
                           endpoint=endpoint)

    else:
        logger.error("Unknown backend: %s", backend)
        return ""


# ---------------------------------------------------------------------------
#  Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a research agent for an automated ML experiment system called orze.
Your job is to analyze past results and generate new experiment ideas.

## How It Works
- You receive the full leaderboard, failure analysis, and performance patterns
- You propose new experiments as structured ideas
- Each idea needs: a title, hypothesis, and YAML config
- The system will automatically train and evaluate your ideas

## Strategy Guidelines
- **Study the leaderboard**: understand what makes top performers successful
- **Study the failures**: avoid config patterns that consistently fail
- **Use performance patterns**: config dimensions show which values correlate with
  better results — build on high-performing values, avoid low-performing ones
- **Set parent IDs**: when iterating on a successful experiment, set "parent" to
  its idea ID so the lineage is tracked
- **Balance exploitation and exploration**: ~60% ideas should refine what works,
  ~40% should try genuinely new approaches

## Output Format
Return a JSON array of ideas. Each idea is an object with:
- "title": short descriptive name (string)
- "hypothesis": why this might work, referencing evidence from the context (string)
- "config": YAML-compatible dict with experiment config (object)
- "priority": "critical" | "high" | "medium" | "low" (string, optional)
- "category": free-form label like "architecture", "hyperparameter", "loss" (string, optional)
- "approach_family": one of: "architecture", "training_config", "data", "infrastructure", "optimization", "regularization", "ensemble", "other" (string, required)
- "parent": "none" or an existing idea ID if building on a previous idea (string, optional)

Note: idea IDs are auto-generated as 6-char content hashes (e.g. "idea-a7f3b2").
You do NOT need to assign IDs — just provide the config and orze will hash it.

Example:
```json
[
  {
    "title": "Larger learning rate with cosine schedule",
    "hypothesis": "Current best uses lr=1e-4. A 3x larger lr with cosine decay may converge faster and find a better minimum.",
    "config": {
      "model": {"type": "resnet50", "pretrained": true},
      "training": {"lr": 3e-4, "scheduler": "cosine", "epochs": 20}
    },
    "priority": "high",
    "category": "hyperparameter",
    "approach_family": "training_config",
    "parent": "none"
  }
]
```

Output ONLY the JSON array, no markdown fences or extra text.
"""
