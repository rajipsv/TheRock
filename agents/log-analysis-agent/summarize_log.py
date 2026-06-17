#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Generate executive summary from log analysis report.json (template or LLM)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env
from llm import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL, invoke_llm_backend, llm_env_config

load_agent_env()

DEFAULT_INPUT = AGENT_DIR / "out" / "report.json"
DEFAULT_OUTPUT = AGENT_DIR / "out" / "executive_summary.md"
LLM_BRIEF_HEADING = "## Triage brief (LLM)\n\n"
SYSTEM_MESSAGE = (
    "You summarize CI log triage reports for validation engineers. "
    "Use only facts from the provided JSON. Do not invent errors or fixes."
)


def template_log_summary(report: dict) -> str:
    lines = [
        "# Log Analysis Executive Summary",
        "",
        f"**Log:** `{report.get('log_path', '')}`",
    ]
    if report.get("github_run_id"):
        lines.append(
            f"**GitHub run:** [{report.get('github_run_id')}]({report.get('html_url', '')}) "
            f"({report.get('repo', '')}, job: {report.get('job_name', report.get('github_job_id', ''))})"
        )
    lines.extend([
        f"**Mode:** {report.get('mode', 'tool_only')}",
        f"**Preset:** {report.get('preset', 'custom')}",
        f"**Errors found:** {len(report.get('errors', []))}",
        "",
        "## Summary",
        report.get("summary") or report.get("executive_summary") or "No summary available.",
        "",
    ])

    errors = report.get("errors") or []
    if errors:
        lines.append("## Top errors")
        for err in errors[:5]:
            if isinstance(err, dict):
                lines.append(
                    f"- **Line {err.get('line_number', '?')}** ({err.get('severity', '?')}): "
                    f"{err.get('message', '')[:120]}"
                )
                if err.get("recommendation"):
                    lines.append(f"  - Recommendation: {err['recommendation']}")
            else:
                lines.append(f"- {err}")
        if len(errors) > 5:
            lines.append(f"- ... and {len(errors) - 5} more")

    rag = report.get("rag_lookups") or []
    if rag:
        lines.append("")
        lines.append("## Knowledge base matches")
        for item in rag[:3]:
            sig = item.get("error_signature", "")
            matches = item.get("matches") or []
            if matches:
                m0 = matches[0]
                lines.append(
                    f"- `{sig[:80]}` → {m0.get('pattern', '')} "
                    f"({m0.get('category', '')}, score={m0.get('score', '')})"
                )

    return "\n".join(lines) + "\n"


def llm_prompt(report: dict) -> str:
    compact = {
        "log_path": report.get("log_path"),
        "mode": report.get("mode"),
        "preset": report.get("preset"),
        "errors_count": report.get("errors_count", len(report.get("errors", []))),
        "summary": report.get("summary"),
        "errors": (report.get("errors") or [])[:8],
        "rag_lookups": (report.get("rag_lookups") or [])[:3],
        "stats": report.get("stats"),
    }
    return (
        "Write a concise triage brief (3-5 bullet points) for validation engineers. "
        "Focus on root causes and recommended next steps from the data below.\n\n"
        + json.dumps(compact, indent=2)
    )


def generate_log_summary(
    report: dict,
    *,
    backend: str = "template",
    model: str | None = None,
    base_url: str | None = None,
    llm_mode: str = "brief",
) -> str:
    if backend == "template":
        return template_log_summary(report)

    cfg = llm_env_config()
    resolved_model = model or cfg["model"]
    resolved_base = base_url or (
        cfg["base_url"] if backend == "vllm" else "http://localhost:11434"
    )

    try:
        llm_text = invoke_llm_backend(
            backend,
            llm_prompt(report),
            model=resolved_model,
            base_url=resolved_base,
            system=SYSTEM_MESSAGE,
        )
    except Exception as exc:
        print(f"LLM summary failed ({backend}): {exc}", file=sys.stderr)
        return template_log_summary(report)

    if not llm_text.strip():
        return template_log_summary(report)

    if llm_mode == "brief":
        return template_log_summary(report) + "\n" + LLM_BRIEF_HEADING + llm_text + "\n"
    return llm_text + "\n"


def llm_summary(report: dict, provider: str = "openai") -> str:
    """Backward-compatible wrapper."""
    backend = "vllm" if provider == "vllm" else provider
    return generate_log_summary(report, backend=backend, llm_mode="standalone")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate executive summary from log report.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=("template", "openai", "vllm", "ollama"),
        default=os.environ.get("LOG_SUMMARY_BACKEND", "template"),
        help="Summary backend (default: template or LOG_SUMMARY_BACKEND env)",
    )
    parser.add_argument(
        "--llm",
        choices=("template", "openai"),
        help="Deprecated alias for --backend (template|openai only)",
    )
    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_VLLM_BASE_URL,
        help="Ollama base URL or OpenAI-compatible API (vLLM)",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("brief", "standalone"),
        default="brief",
        help="brief: template + LLM section (default); standalone: LLM only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    backend = args.backend
    if args.llm:
        backend = args.llm

    if not args.input.is_file():
        print(f"Report not found: {args.input}", file=sys.stderr)
        return 1

    report = json.loads(args.input.read_text(encoding="utf-8"))
    summary = generate_log_summary(
        report,
        backend=backend,
        model=args.model,
        base_url=args.base_url,
        llm_mode=args.llm_mode,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    report["executive_summary"] = summary
    args.input.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
