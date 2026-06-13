import json
import re
from typing import Any

import httpx

from app.config import settings


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style reasoning blocks before JSON parsing."""
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    pattern = re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


class LLMClient:
    async def complete(self, system: str, user: str, max_tokens: int = 1200) -> str | None:
        if not settings.use_llm:
            return None

        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        try:
            timeout = httpx.Timeout(settings.llm_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return strip_thinking(content) if content else None
        except (httpx.HTTPError, KeyError, IndexError):
            return None

    async def ping(self) -> tuple[bool, str]:
        """Quick connectivity check against the configured vLLM server."""
        if not settings.use_llm:
            return False, "USE_LLM is false"

        try:
            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{settings.llm_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                )
                response.raise_for_status()
                data = response.json()
                model_ids = [m.get("id", "") for m in data.get("data", [])]
                if settings.llm_model in model_ids:
                    return True, f"Connected; model `{settings.llm_model}` is available"
                if model_ids:
                    return True, (
                        f"Connected; configured model `{settings.llm_model}` not listed. "
                        f"Available: {', '.join(model_ids[:5])}"
                    )
                return True, "Connected; no models returned in /models response"
        except httpx.HTTPError as exc:
            return False, f"HTTP error: {exc}"

    @staticmethod
    def parse_json_block(text: str) -> Any | None:
        if not text:
            return None

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = fenced.group(1).strip() if fenced else text.strip()

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    return None
            start = candidate.find("[")
            end = candidate.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None
