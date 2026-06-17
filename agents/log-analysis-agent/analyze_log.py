#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Log Analysis Agent — CI/build log triage (tool-only default, optional LangGraph agent)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

from failure_kb import get_default_kb
from github_logs import (
    download_job_log_text,
    get_run,
    github_metadata_for_run,
    infer_preset_for_run,
    list_jobs,
    select_failed_jobs,
    write_job_log,
)
from log_agent import LogAnalysisAgent, errors_from_tool_only, summary_from_tool_only
from log_tools import LogSession, run_tool_only_analysis
from presets import PRESET_NAMES, get_preset

DEFAULT_OUT = AGENT_DIR / "out"
DEFAULT_REPO = "ROCm/TheRock"


def build_report(
    log_path: Path,
    *,
    preset_name: str = "custom",
    kb_dir: Path | None = None,
    use_agent: bool = False,
    model: str = "gpt-4o-mini",
    max_iterations: int = 16,
    github_meta: dict | None = None,
) -> dict:
    preset = get_preset(preset_name)
    kb = get_default_kb(kb_dir)
    extra = list(preset.extra_patterns)

    if use_agent:
        agent = LogAnalysisAgent(model_name=model, max_iterations=max_iterations, kb=kb)
        result = agent.analyze(log_path, use_llm=True, extra_patterns=extra)
        tool_data = result.get("tool_analysis") or {}
        report = {
            "log_path": str(log_path.resolve()),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "mode": result.get("mode", "agent"),
            "model": result.get("model"),
            "preset": preset.name,
            "preset_label": preset.label,
            "stats": tool_data.get("stats") or "",
            "chunk_overview": tool_data.get("chunk_overview", ""),
            "errors": result.get("errors", []),
            "errors_count": result.get("errors_count", len(result.get("errors", []))),
            "stack_traces": tool_data.get("stack_traces") or result.get("stack_traces", ""),
            "rag_lookups": result.get("rag_lookups", []),
            "summary": result.get("summary", ""),
            "executive_summary": "",
            "tool_analysis": tool_data,
            "agent_trace": result.get("agent_trace", []),
            "raw_agent_response": result.get("raw_agent_response"),
        }
    else:
        session = LogSession(path=log_path.resolve())
        tool_data = run_tool_only_analysis(session, kb=kb, extra_patterns=extra)
        errors = errors_from_tool_only(tool_data, kb)
        report = {
            "log_path": str(log_path.resolve()),
            "timestamp": datetime.now().isoformat(),
            "mode": "tool_only",
            "preset": preset.name,
            "preset_label": preset.label,
            "stats": tool_data.get("stats", ""),
            "chunk_overview": tool_data.get("chunk_overview", ""),
            "errors": errors,
            "errors_count": len(errors),
            "stack_traces": tool_data.get("stack_traces", ""),
            "rag_lookups": tool_data.get("rag_lookups", []),
            "summary": summary_from_tool_only(tool_data),
            "executive_summary": "",
            "tool_analysis": tool_data,
        }

    if github_meta:
        report.update(github_meta)
    return report


def write_html(report: dict, output_dir: Path) -> Path:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(AGENT_DIR / "report_templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    html_path = output_dir / "report.html"
    html_path.write_text(template.render(report=report), encoding="utf-8")
    return html_path


def write_outputs(
    report: dict,
    output_dir: Path,
    write_summary: bool = True,
    summary_backend: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, output_dir)

    if write_summary:
        from summarize_log import generate_log_summary

        backend = summary_backend or os.environ.get("LOG_SUMMARY_BACKEND", "template")
        summary = generate_log_summary(report, backend=backend)
        report["executive_summary"] = summary.strip()
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_dir / "executive_summary.md").write_text(summary, encoding="utf-8")


def analyze_github_run(
    run_id: int,
    *,
    repo: str = DEFAULT_REPO,
    job_id: int | None = None,
    output_dir: Path,
    preset: str = "auto",
    kb_dir: Path | None = None,
    use_agent: bool = False,
    model: str = "gpt-4o-mini",
    max_iterations: int = 16,
    max_jobs: int = 3,
    write_summary: bool = True,
    summary_backend: str | None = None,
) -> list[dict]:
    """Download failed job log(s) from a GitHub Actions run and analyze."""
    run = get_run(repo, run_id)
    jobs = list_jobs(repo, run_id)

    if job_id is not None:
        selected = [j for j in jobs if j.id == job_id]
        if not selected:
            raise RuntimeError(f"Job {job_id} not found in run {run_id}")
    else:
        selected = select_failed_jobs(jobs, max_jobs=max_jobs)
        if not selected:
            raise RuntimeError(f"No jobs found for run {run_id}")

    preset_name = infer_preset_for_run(run, preset)
    reports: list[dict] = []

    for job in selected:
        log_text = download_job_log_text(repo, job.id)
        job_out = output_dir if len(selected) == 1 else output_dir / f"job-{job.id}"
        log_path = write_job_log(log_text, job_out, job.id)
        meta = github_metadata_for_run(run, job, repo)
        report = build_report(
            log_path,
            preset_name=preset_name,
            kb_dir=kb_dir,
            use_agent=use_agent,
            model=model,
            max_iterations=max_iterations,
            github_meta=meta,
        )
        write_outputs(
            report,
            job_out,
            write_summary=write_summary,
            summary_backend=summary_backend,
        )
        reports.append(report)

    return reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGENTS_030 Log Analysis Agent for TheRock CI/build logs"
    )
    parser.add_argument("--log", help="Path to local build/CI log file")
    parser.add_argument("--github-run-id", type=int, help="GitHub Actions workflow run ID")
    parser.add_argument("--github-job-id", type=int, help="Specific job ID within the run")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repo owner/name (default: ROCm/TheRock)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--preset",
        default="auto",
        help="Preset or 'auto' to infer from workflow name",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use LangGraph ReAct agent (needs requirements-agent.txt + LLM: OpenAI, NVIDIA, or vLLM)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
    )
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--max-jobs", type=int, default=3, help="Max failed jobs to analyze per run")
    parser.add_argument("--kb-dir", type=Path, default=None)
    parser.add_argument(
        "--record-resolution",
        nargs=2,
        metavar=("SIGNATURE", "RESOLUTION"),
        help="Add a learned resolution to the KB and exit",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip executive_summary.md")
    parser.add_argument(
        "--summary-backend",
        choices=("template", "openai", "vllm", "ollama"),
        default=None,
        help="Executive summary backend (default: template or LOG_SUMMARY_BACKEND env)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.record_resolution:
        kb = get_default_kb(args.kb_dir)
        sig, res = args.record_resolution
        entry = kb.record_resolution(sig, res)
        print(f"Recorded: {entry}")
        return 0

    if args.github_run_id:
        if args.preset != "auto" and args.preset not in PRESET_NAMES:
            try:
                get_preset(args.preset)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 2
        try:
            reports = analyze_github_run(
                args.github_run_id,
                repo=args.repo,
                job_id=args.github_job_id,
                output_dir=args.output_dir,
                preset=args.preset,
                kb_dir=args.kb_dir,
                use_agent=args.agent,
                model=args.model,
                max_iterations=args.max_iterations,
                max_jobs=args.max_jobs,
                write_summary=not args.no_summary,
                summary_backend=args.summary_backend,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        for report in reports:
            print(f"Mode: {report['mode']}")
            print(f"Errors: {report.get('errors_count', len(report.get('errors', [])))}")
            print(f"Workflow: {report.get('workflow_name')} job={report.get('job_name')}")
        print(f"Report: {args.output_dir / 'report.json'}")
        return 0

    if not args.log:
        print("Provide --log or --github-run-id", file=sys.stderr)
        return 2

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    preset_name = args.preset if args.preset != "auto" else "custom"
    try:
        get_preset(preset_name)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    report = build_report(
        log_path,
        preset_name=preset_name,
        kb_dir=args.kb_dir,
        use_agent=args.agent,
        model=args.model,
        max_iterations=args.max_iterations,
    )
    write_outputs(
        report,
        args.output_dir,
        write_summary=not args.no_summary,
        summary_backend=args.summary_backend,
    )

    print(f"Mode: {report['mode']}")
    print(f"Errors: {report.get('errors_count', len(report.get('errors', [])))}")
    print(f"Report: {args.output_dir / 'report.json'}")
    print(f"HTML: {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
