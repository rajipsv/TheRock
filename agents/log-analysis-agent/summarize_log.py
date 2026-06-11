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

load_agent_env()

DEFAULT_INPUT = AGENT_DIR / "out" / "report.json"
DEFAULT_OUTPUT = AGENT_DIR / "out" / "executive_summary.md"


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


def llm_summary(report: dict, provider: str = "openai") -> str:
    prompt = (
        "Write a concise executive summary (3-5 bullet points) for validation engineers "
        "triaging this CI log failure. Focus on root causes and recommended next steps.\n\n"
        + json.dumps(report, indent=2)[:12000]
    )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You summarize CI log triage reports for engineers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or template_log_summary(report)

    return template_log_summary(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate executive summary from log report.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm", choices=("template", "openai"), default="template")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Report not found: {args.input}", file=sys.stderr)
        return 1

    report = json.loads(args.input.read_text(encoding="utf-8"))
    if args.llm == "openai":
        summary = llm_summary(report, provider="openai")
    else:
        summary = template_log_summary(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
