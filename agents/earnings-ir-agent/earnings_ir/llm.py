import json
import re
from typing import Any

import httpx

from earnings_ir.config import settings


def strip_thinking(text: str) -> str:
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    pattern = re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag)
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


class LLMClient:
    async def complete(self, system: str, user: str, max_tokens: int = 2000) -> str | None:
        if not settings.use_llm:
            return None

        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
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
                content = response.json()["choices"][0]["message"]["content"]
                return strip_thinking(content) if content else None
        except (httpx.HTTPError, KeyError, IndexError):
            return None

    @staticmethod
    def parse_json_block(text: str) -> Any | None:
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = fenced.group(1).strip() if fenced else text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
            start, end = candidate.find("["), candidate.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    async def ping(self) -> tuple[bool, str]:
        if not settings.use_llm:
            return False, "USE_LLM is false"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{settings.llm_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                )
                response.raise_for_status()
                ids = [m.get("id", "") for m in response.json().get("data", [])]
                if settings.llm_model in ids:
                    return True, f"Model {settings.llm_model} available"
                return True, f"Connected; available: {', '.join(ids[:3])}"
        except httpx.HTTPError as exc:
            return False, str(exc)
