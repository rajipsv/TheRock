# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Optional LangGraph ReAct agent for log qualification (requires requirements-agent.txt)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from failure_kb import FailureKnowledgeBase, get_default_kb
from llm import is_vllm_configured, llm_credentials_available, llm_env_config, sanitize_llm_text
from log_tools import LogSession, create_langchain_tools, run_tool_only_analysis
from rag_tools import create_rag_tools

SYSTEM_PROMPT = """You are the Log Analysis Agent, an expert log qualification agent for software test and CI pipelines.

You MUST use tools to analyze logs. Never guess file contents.

Workflow:
1. Call get_log_stats first.
2. For large logs, call chunk_overview then read_log_window on suspicious regions.
3. Use grep_error_keyword and grep_log to find failures (ERROR, CRITICAL, FATAL, EXCEPTION).
4. Call extract_stack_traces for root-cause context.
5. For each distinct error line, call lookup_known_failure with the exact error signature.
6. Use search_failure_knowledge for ROCm/GPU/install failures if category is unclear.
7. Enrich recommendations using knowledge base solutions (do not invent fixes).
8. Produce a final answer with:
   - Executive summary (2-4 sentences)
   - JSON block with schema:
     {"errors": [{"type": str, "line_number": int, "message": str, "severity": str, "category": str, "recommendation": str, "kb_pattern_id": str|null}], "total_errors": int, "summary": str}
   - Top 3 recommended actions for validation engineers

Severity: CRITICAL > HIGH > MEDIUM > LOW.
Categories: Database, Network, Authentication, GPU/Driver, Configuration, Runtime, Memory, Security, Other.
"""


def _extract_json_from_text(text: str) -> dict | None:
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _messages_to_trace(messages: list) -> list[dict]:
    trace = []
    for m in messages:
        role = getattr(m, "type", type(m).__name__)
        content = getattr(m, "content", str(m))
        tool_calls = getattr(m, "tool_calls", None)
        entry: dict[str, Any] = {"role": role}
        if content:
            entry["content"] = content if isinstance(content, str) else str(content)[:2000]
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name", getattr(tc, "name", "")), "args": tc.get("args", getattr(tc, "args", {}))}
                for tc in tool_calls
            ]
        trace.append(entry)
    return trace


def count_errors_from_tool_only(tool_data: dict) -> int:
    text = (
        tool_data.get("error_samples", "")
        + tool_data.get("critical_samples", "")
        + tool_data.get("fatal_samples", "")
        + tool_data.get("github_error_samples", "")
        + "\n".join((tool_data.get("rocm_ci_samples") or {}).values())
    )
    return len(re.findall(r"^>>\s+\d+:", text, re.MULTILINE))


def errors_from_tool_only(tool_data: dict, kb: FailureKnowledgeBase | None = None) -> list[dict]:
    rag_by_sig = {}
    for item in tool_data.get("rag_lookups", []):
        rag_by_sig[item.get("error_signature", "")] = item.get("matches", [])

    errors = []
    for field, default_type, default_severity in (
        ("error_samples", "ERROR", "HIGH"),
        ("critical_samples", "CRITICAL", "CRITICAL"),
        ("fatal_samples", "FATAL", "CRITICAL"),
        ("github_error_samples", "GITHUB_ACTIONS", "HIGH"),
    ):
        block = tool_data.get(field, "")
        for match in re.finditer(r">>\s+(\d+):\s*(.+)", block):
            msg = match.group(2).strip()
            rec = "Inspect surrounding context and stack traces; check recent CI changes."
            category = "Runtime"
            kb_id = None
            matches = rag_by_sig.get(msg) or []
            if not matches and kb is not None:
                top = kb.lookup_known_failure(msg, top_k=1)
                matches = [m.to_dict() for m in top]
            if matches:
                m0 = matches[0]
                rec = m0.get("solutions", rec)
                category = m0.get("category", category)
                kb_id = m0.get("pattern_id")
            errors.append(
                {
                    "type": default_type,
                    "line_number": int(match.group(1)),
                    "message": msg,
                    "severity": default_severity,
                    "category": category,
                    "recommendation": rec,
                    "kb_pattern_id": kb_id,
                }
            )

    for label, block in (tool_data.get("rocm_ci_samples") or {}).items():
        if not block or block.startswith("No matches"):
            continue
        for match in re.finditer(r">>\s+(\d+):\s*(.+)", block):
            msg = match.group(2).strip()
            rec = "Inspect ROCm GPU memory, driver stack, and test parallelism on the runner."
            category = "GPU/Driver"
            kb_id = None
            if kb is not None:
                top = kb.lookup_known_failure(msg, top_k=1)
                if top:
                    m0 = top[0].to_dict()
                    rec = m0.get("solutions", rec)
                    category = m0.get("category", category)
                    kb_id = m0.get("pattern_id")
            severity = "CRITICAL" if label in ("hipErrorOutOfMemory", "gtest_failed") else "HIGH"
            errors.append(
                {
                    "type": label.upper(),
                    "line_number": int(match.group(1)),
                    "message": msg,
                    "severity": severity,
                    "category": category,
                    "recommendation": rec,
                    "kb_pattern_id": kb_id,
                }
            )
    return errors[:50]


def summary_from_tool_only(tool_data: dict) -> str:
    stats = tool_data.get("stats", "")
    n = count_errors_from_tool_only(tool_data)
    return f"Tool-only qualification pass found {n} highlighted error lines. Stats: {stats.replace(chr(10), '; ')}"


class LogAnalysisAgent:
    """Tool-using ReAct agent for log qualification."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 16,
        kb: FailureKnowledgeBase | None = None,
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.kb = kb or get_default_kb()

    def _all_tools(self, session: LogSession) -> list:
        return create_langchain_tools(session) + create_rag_tools(self.kb)

    def _resolve_model_name(self) -> str:
        if is_vllm_configured():
            cfg = llm_env_config()
            return cfg["model"]
        return self.model_name

    def _build_graph(self, tools: list):
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not llm_credentials_available():
            return None

        kwargs: dict = {"temperature": 0}

        if nvidia_key:
            kwargs["api_key"] = nvidia_key
            kwargs["base_url"] = os.getenv(
                "NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"
            )
            kwargs["model"] = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
        elif is_vllm_configured() or (
            openai_key and os.getenv("VLLM_BASE_URL")
        ):
            cfg = llm_env_config()
            kwargs["api_key"] = cfg["api_key"]
            kwargs["base_url"] = cfg["base_url"]
            kwargs["model"] = (
                os.getenv("VLLM_MODEL")
                or os.getenv("LLM_MODEL")
                or self.model_name
            )
        elif openai_key:
            kwargs["api_key"] = openai_key
            kwargs["model"] = self.model_name
        else:
            cfg = llm_env_config()
            kwargs["api_key"] = cfg["api_key"]
            kwargs["base_url"] = cfg["base_url"]
            kwargs["model"] = cfg["model"]

        llm = ChatOpenAI(**kwargs)
        return create_react_agent(
            llm,
            tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def analyze(
        self,
        log_path: str | Path,
        *,
        use_llm: bool = True,
        extra_patterns: list[str] | None = None,
    ) -> dict:
        path = Path(log_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Log file not found: {path}")

        session = LogSession(path=path)
        timestamp = datetime.now().isoformat()

        has_llm = llm_credentials_available()
        if not use_llm or not has_llm:
            tool_data = run_tool_only_analysis(session, kb=self.kb, extra_patterns=extra_patterns)
            return {
                "file": str(path),
                "timestamp": timestamp,
                "mode": "tool_only",
                "tool_analysis": tool_data,
                "errors": errors_from_tool_only(tool_data, self.kb),
                "errors_count": count_errors_from_tool_only(tool_data),
                "summary": summary_from_tool_only(tool_data),
                "rag_lookups": tool_data.get("rag_lookups", []),
                "agent_trace": [],
            }

        tools = self._all_tools(session)
        graph = self._build_graph(tools)
        if graph is None:
            raise RuntimeError(
                "Failed to build agent graph — set OPENAI_API_KEY, NVIDIA_API_KEY, "
                "or vLLM env (VLLM_BASE_URL / USE_VLLM=1)"
            )

        user_msg = (
            f"Analyze this log file for qualification triage: {path}\n"
            "Use tools systematically. Return structured JSON errors plus summary."
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config={"recursion_limit": self.max_iterations},
        )

        messages = result.get("messages", [])
        final_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                final_text = m.content if isinstance(m.content, str) else str(m.content)
                break
        final_text = sanitize_llm_text(final_text)

        parsed = _extract_json_from_text(final_text) or {}
        errors = parsed.get("errors", [])
        if not isinstance(errors, list):
            errors = []

        return {
            "file": str(path),
            "timestamp": timestamp,
            "mode": "agent",
            "model": self._resolve_model_name(),
            "errors": errors,
            "errors_count": parsed.get("total_errors", len(errors)),
            "summary": parsed.get("summary", final_text[:1500]),
            "raw_agent_response": final_text,
            "agent_trace": _messages_to_trace(messages),
        }
