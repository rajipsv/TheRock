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
from log_agent import infer_primary_root_cause, rank_errors_for_root_cause
from llm import DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL, default_summary_backend, invoke_llm_backend, llm_env_config

load_agent_env()

DEFAULT_INPUT = AGENT_DIR / "out" / "report.json"
DEFAULT_OUTPUT = AGENT_DIR / "out" / "executive_summary.md"
LLM_BRIEF_HEADING = "## Triage brief (LLM)\n\n"
SYSTEM_MESSAGE = (
    "You summarize CI log triage reports for validation engineers. "
    "Use only facts from the provided JSON. Do not invent errors or fixes. "
    "IGNORE environment setup banners (e.g. 'Compoments check' / 'Components check' "
    "with Passed/Warning/Fatal counts) — they are runner health noise, not the test root cause. "
    "Prioritize primary_root_cause and root_cause_errors: GPU OOM (hipErrorOutOfMemory), "
    "failed test suites (bsric0, gtest FAILED), rocsparse_create_handle, and final exit codes."
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

    errors = rank_errors_for_root_cause(report.get("errors") or [], limit=5)
    primary = report.get("primary_root_cause") or infer_primary_root_cause(report.get("errors") or [])
    if primary:
        lines.append("## Primary root cause (ranked)")
        lines.append(
            f"- **Line {primary.get('line_number', '?')}** ({primary.get('severity', '?')}): "
            f"{primary.get('message', '')[:160]}"
        )
        if primary.get("recommendation"):
            lines.append(f"  - Recommendation: {primary['recommendation']}")
        lines.append("")

    if errors:
        lines.append("## Top errors (ranked)")
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
    all_errors = report.get("errors") or []
    ranked = rank_errors_for_root_cause(all_errors, limit=8)
    primary = report.get("primary_root_cause") or infer_primary_root_cause(all_errors)
    compact = {
        "log_path": report.get("log_path"),
        "job_name": report.get("job_name"),
        "github_run_id": report.get("github_run_id"),
        "github_job_id": report.get("github_job_id"),
        "mode": report.get("mode"),
        "preset": report.get("preset"),
        "errors_count": report.get("errors_count", len(all_errors)),
        "summary": report.get("summary"),
        "primary_root_cause": primary,
        "root_cause_errors": ranked,
        "stats": report.get("stats"),
        "rag_lookups": (report.get("rag_lookups") or [])[:3],
        "instructions": (
            "Write 3-5 bullets: (1) primary root cause, (2) failed test/component if any, "
            "(3) recommended next steps. Do NOT cite Compoments/Components check banners as root cause."
        ),
    }
    return (
        "Write a concise triage brief for validation engineers.\n\n"
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
        default=default_summary_backend(),
        help="Summary backend (default: vllm when USE_VLLM=1, else template)",
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
