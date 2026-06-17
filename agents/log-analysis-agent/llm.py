# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""OpenAI-compatible LLM helpers (vLLM on MI300, OpenAI cloud, Ollama)."""

from __future__ import annotations

import os
import re

_THINKING_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)
_REDACTED_THINKING_RE = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "Qwen3-30B-A3B"


def sanitize_llm_text(text: str) -> str:
    """Remove chain-of-thought / thinking blocks from model output."""
    cleaned = text
    for pattern in (_THINKING_BLOCK_RE, _REDACTED_THINKING_RE):
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def is_vllm_configured() -> bool:
    if os.getenv("VLLM_BASE_URL") or os.getenv("LLM_BASE_URL"):
        return True
    return os.getenv("USE_VLLM", "").lower() in ("1", "true", "yes")


def llm_env_config() -> dict[str, str]:
    """Resolve vLLM / OpenAI-compatible endpoint settings from environment."""
    base_url = (
        os.getenv("VLLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or DEFAULT_VLLM_BASE_URL
    )
    model = (
        os.getenv("VLLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_VLLM_MODEL
    )
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "abc-123"
    return {"base_url": base_url, "model": model, "api_key": api_key}


def llm_credentials_available() -> bool:
    """True when an LLM backend can be reached (cloud, NVIDIA, or local vLLM)."""
    if os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return True
    return is_vllm_configured()


def call_openai_compatible(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    system: str | None = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai package: pip install openai") from exc

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra_body: dict | None = None
    if os.environ.get("VLLM_DISABLE_THINKING", "").lower() in ("1", "true", "yes"):
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "not-needed")
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def call_ollama(
    prompt: str,
    *,
    model: str,
    base_url: str,
    system: str | None = None,
) -> str:
    import requests

    user_prompt = prompt
    if system:
        user_prompt = f"System: {system}\n\nUser: {prompt}"

    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": user_prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def invoke_llm_backend(
    backend: str,
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    system: str | None = None,
) -> str:
    if backend == "template":
        raise ValueError("template backend does not call an LLM")
    if backend == "ollama":
        raw = call_ollama(prompt, model=model, base_url=base_url, system=system)
        return sanitize_llm_text(raw)
    if backend == "openai":
        raw = call_openai_compatible(
            prompt,
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            system=system,
        )
        return sanitize_llm_text(raw)
    if backend == "vllm":
        cfg = llm_env_config()
        raw = call_openai_compatible(
            prompt,
            base_url=base_url or cfg["base_url"],
            model=model or cfg["model"],
            api_key=api_key or cfg["api_key"],
            system=system,
        )
        return sanitize_llm_text(raw)
    raise ValueError(f"Unknown backend: {backend}")
