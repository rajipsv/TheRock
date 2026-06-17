# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Deterministic log analysis tools (ported from ARVIL agentic log_tools)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ERROR_KEYWORDS = ("ERROR", "EXCEPTION", "CRITICAL", "FATAL", "WARNING")
GITHUB_ERROR = re.compile(r"##\[error\]", re.IGNORECASE)
ROCM_CI_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"hipErrorOutOfMemory", "hipErrorOutOfMemory"),
    (r"rocsparse_create_handle", "rocsparse_create_handle"),
    (r"\[\s+FAILED\s+\]", "gtest_failed"),
    (r"exit code 127", "exit_code_127"),
    (r"No GPU suite", "no_gpu_suite"),
    (r"ctest.*failed|tests failed", "ctest_failed"),
)
STACK_START = re.compile(
    r"^\s*(at\s+[\w.$]+\(|Traceback|Caused by:|---\s*Crash)",
    re.IGNORECASE,
)


@dataclass
class LogSession:
    """Active log file for a single analysis run."""

    path: Path

    def read_lines(self) -> list[str]:
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            return f.readlines()


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def get_log_stats(session: LogSession) -> str:
    lines = session.read_lines()
    size = session.path.stat().st_size
    counts = {kw: sum(1 for ln in lines if kw in ln.upper()) for kw in ERROR_KEYWORDS}
    gh_errors = sum(1 for ln in lines if GITHUB_ERROR.search(ln))
    counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
    return (
        f"path={session.path}\n"
        f"lines={len(lines)}\n"
        f"bytes={size}\n"
        f"github_error={gh_errors}\n"
        f"keyword_hits: {counts_str}"
    )


def read_log_window(session: LogSession, start_line: int, end_line: int) -> str:
    lines = session.read_lines()
    if not lines:
        return "Log file is empty."
    start = max(1, start_line)
    end = min(len(lines), end_line)
    if start > end:
        return f"Invalid range: start_line={start_line} > end_line={end_line} (file has {len(lines)} lines)"
    chunk = "".join(lines[start - 1 : end])
    header = f"Lines {start}-{end} of {len(lines)} from {session.path.name}\n{'-' * 40}\n"
    return _truncate(header + chunk)


def grep_log(
    session: LogSession,
    pattern: str,
    max_matches: int = 20,
    context_lines: int = 2,
) -> str:
    lines = session.read_lines()
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex '{pattern}': {e}"

    blocks: list[str] = []
    for i, line in enumerate(lines):
        if not rx.search(line):
            continue
        lo = max(0, i - context_lines)
        hi = min(len(lines), i + context_lines + 1)
        block_lines = []
        for j in range(lo, hi):
            prefix = ">>" if j == i else "  "
            block_lines.append(f"{prefix} {j + 1}: {lines[j].rstrip()}")
        blocks.append("\n".join(block_lines))
        if len(blocks) >= max_matches:
            break

    if not blocks:
        return f"No matches for pattern: {pattern}"
    return _truncate(
        f"Matches for /{pattern}/ (showing up to {max_matches}):\n\n" + "\n\n---\n\n".join(blocks)
    )


def grep_error_keyword(session: LogSession, keyword: str, max_matches: int = 15) -> str:
    kw = keyword.upper().strip()
    if kw not in ERROR_KEYWORDS:
        return f"Unsupported keyword '{keyword}'. Use one of: {', '.join(ERROR_KEYWORDS)}"
    return grep_log(session, rf"\b{re.escape(kw)}\b", max_matches=max_matches, context_lines=3)


def extract_stack_traces(session: LogSession, max_traces: int = 10) -> str:
    lines = session.read_lines()
    traces: list[str] = []
    i = 0
    while i < len(lines) and len(traces) < max_traces:
        line = lines[i]
        if not any(k in line.upper() for k in ("ERROR", "EXCEPTION", "FATAL", "TRACE")):
            i += 1
            continue
        start = i
        j = i + 1
        while j < len(lines) and (
            STACK_START.match(lines[j])
            or lines[j].startswith((" ", "\t"))
            or "Caused by:" in lines[j]
        ):
            j += 1
        block = "".join(f"{idx + 1}: {lines[idx]}" for idx in range(start, min(j + 1, len(lines))))
        traces.append(block.strip())
        i = j + 1

    if not traces:
        return "No stack traces detected."
    return _truncate(f"Stack traces ({len(traces)}):\n\n" + "\n\n---\n\n".join(traces))


def chunk_overview(session: LogSession, lines_per_chunk: int = 500) -> str:
    lines = session.read_lines()
    n = len(lines)
    if n == 0:
        return "Empty log."
    chunk_count = (n + lines_per_chunk - 1) // lines_per_chunk
    parts = []
    for c in range(min(chunk_count, 20)):
        start = c * lines_per_chunk + 1
        end = min((c + 1) * lines_per_chunk, n)
        window = lines[start - 1 : end]
        hits = sum(1 for ln in window if any(k in ln.upper() for k in ERROR_KEYWORDS))
        parts.append(f"chunk {c + 1}: lines {start}-{end}, keyword_hits={hits}")
    if chunk_count > 20:
        parts.append(f"... {chunk_count - 20} more chunks")
    return f"total_lines={n}, chunk_size={lines_per_chunk}, chunks={chunk_count}\n" + "\n".join(parts)


def _extract_highlight_messages(grep_output: str, limit: int = 5) -> list[str]:
    messages = []
    for match in re.finditer(r">>\s+\d+:\s*(.+)", grep_output):
        msg = match.group(1).strip()
        if msg and msg not in messages:
            messages.append(msg)
        if len(messages) >= limit:
            break
    return messages


def run_tool_only_analysis(session: LogSession, kb=None, extra_patterns: list[str] | None = None) -> dict:
    """Deterministic analysis pipeline (no LLM required)."""
    stats = get_log_stats(session)
    overview = chunk_overview(session)
    errors = grep_error_keyword(session, "ERROR", max_matches=25)
    critical = grep_error_keyword(session, "CRITICAL", max_matches=10)
    fatal = grep_error_keyword(session, "FATAL", max_matches=10)
    gh_errors = grep_log(session, r"##\[error\]", max_matches=15, context_lines=2)
    stacks = extract_stack_traces(session)

    rocm_ci_samples: dict[str, str] = {}
    for pattern, label in ROCM_CI_PATTERNS:
        rocm_ci_samples[label] = grep_log(session, pattern, max_matches=12, context_lines=2)

    preset_samples: dict[str, str] = {}
    for pat in extra_patterns or []:
        preset_samples[pat] = grep_log(session, pat, max_matches=10, context_lines=2)

    rag_lookups: list[dict] = []
    if kb is not None:
        signatures = (
            _extract_highlight_messages(errors, 3)
            + _extract_highlight_messages(critical, 2)
            + _extract_highlight_messages(fatal, 2)
            + _extract_highlight_messages(gh_errors, 3)
            + _extract_highlight_messages(
                "\n\n".join(rocm_ci_samples.values()), 5
            )
        )
        for sig in signatures[:5]:
            matches = kb.lookup_known_failure(sig, top_k=2)
            rag_lookups.append(
                {
                    "error_signature": sig,
                    "matches": [m.to_dict() for m in matches],
                    "formatted": kb.format_matches(matches),
                }
            )

    return {
        "mode": "tool_only",
        "stats": stats,
        "chunk_overview": overview,
        "error_samples": errors,
        "critical_samples": critical,
        "fatal_samples": fatal,
        "github_error_samples": gh_errors,
        "rocm_ci_samples": rocm_ci_samples,
        "stack_traces": stacks,
        "preset_pattern_samples": preset_samples,
        "rag_lookups": rag_lookups,
    }


def create_langchain_tools(session: LogSession):
    """Build LangChain StructuredTools for agent mode (optional dependency)."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class GrepInput(BaseModel):
        pattern: str
        max_matches: int = Field(default=20, ge=1, le=100)
        context_lines: int = Field(default=2, ge=0, le=10)

    class WindowInput(BaseModel):
        start_line: int = Field(ge=1)
        end_line: int = Field(ge=1)

    class GrepKeywordInput(BaseModel):
        keyword: str
        max_matches: int = Field(default=15, ge=1, le=50)

    return [
        StructuredTool.from_function(
            func=lambda: get_log_stats(session),
            name="get_log_stats",
            description="Get log file metadata. Call this first.",
        ),
        StructuredTool.from_function(
            func=lambda lines_per_chunk=500: chunk_overview(session, lines_per_chunk),
            name="chunk_overview",
            description="Overview of log chunks for large files.",
        ),
        StructuredTool.from_function(
            func=lambda start_line, end_line: read_log_window(session, start_line, end_line),
            name="read_log_window",
            description="Read a line range from the log (1-based).",
            args_schema=WindowInput,
        ),
        StructuredTool.from_function(
            func=lambda pattern, max_matches=20, context_lines=2: grep_log(
                session, pattern, max_matches, context_lines
            ),
            name="grep_log",
            description="Regex search with line numbers and context.",
            args_schema=GrepInput,
        ),
        StructuredTool.from_function(
            func=lambda keyword, max_matches=15: grep_error_keyword(session, keyword, max_matches),
            name="grep_error_keyword",
            description="Search ERROR, WARNING, CRITICAL, FATAL, EXCEPTION.",
            args_schema=GrepKeywordInput,
        ),
        StructuredTool.from_function(
            func=lambda max_traces=10: extract_stack_traces(session, max_traces),
            name="extract_stack_traces",
            description="Pull exception and stack trace blocks.",
        ),
    ]
