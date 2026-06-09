#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Generate executive summary from report.json (template, Ollama, OpenAI, or vLLM)."""

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


def template_summary(report: dict) -> str:
    lines = [
        "# Change Impact Executive Summary",
        "",
        f"**Range:** `{report.get('start_sha', '')[:8]}` → `{report.get('end_sha', '')[:8]}`",
        f"**Severity:** {report.get('severity')} (blast radius {report.get('blast_radius_score')}/100)",
        "",
        "## What changed",
    ]
    paths = report.get("changed_files", [])
    if paths:
        lines.append("")
        lines.append("### Changed files (git diff)")
        for path in paths[:20]:
            lines.append(f"- `{path}`")
        if len(paths) > 20:
            lines.append(f"- ... and {len(paths) - 20} more files")

    manifest_items = [
        i for i in report.get("changed_components", []) if i.get("kind") != "path"
    ]
    if manifest_items:
        lines.append("")
        lines.append("### Manifest / submodule changes")
        for item in manifest_items[:15]:
            parent = f"{item['parent']}/" if item.get("parent") else ""
            lines.append(
                f"- `{parent}{item['name']}` ({item['status']}, {item['kind']})"
            )
        if len(manifest_items) > 15:
            lines.append(f"- ... and {len(manifest_items) - 15} more")

    content = report.get("content_insights") or {}
    if content.get("notes"):
        lines.append("")
        lines.append("### Content insights (CI / packaging)")
        for note in content["notes"]:
            lines.append(f"- {note}")
    timeout_changes = (content.get("test_matrix_changes") or {}).get(
        "timeout_changes", []
    )
    if timeout_changes:
        lines.append("")
        lines.append("### GHA wrapper timeout changes")
        for change in timeout_changes:
            lines.append(
                f"- `{change['job']}`: {change['old_minutes']} → "
                f"{change['new_minutes']} minutes"
            )

    comp_paths = report.get("changed_paths_in_components") or {}
    superrepo_diffs = report.get("superrepo_diffs") or {}
    commit_meta: dict[str, dict] = {}
    for info in superrepo_diffs.values():
        for comp, meta in (info.get("components") or {}).items():
            commit_meta[comp] = meta

    if comp_paths:
        lines.append("")
        lines.append("### Changed superrepo components")
        for comp, info in list(comp_paths.items())[:20]:
            file_count = info.get("file_count", 0)
            commit_count = commit_meta.get(comp, {}).get("commit_count")
            detail = f"{file_count} file(s)"
            if commit_count:
                detail += f", {commit_count} commit(s) in range"
            lines.append(f"- `{comp}`: {detail}")
            for p in info.get("sample_paths", [])[:3]:
                lines.append(f"  - `{p}`")

    lines.extend(
        [
            "",
            "## Blast radius",
            "",
        ]
    )
    for r in report.get("rationale", []):
        lines.append(f"- {r}")

    lines.extend(
        [
            "",
            f"**Affected build stages:** {', '.join(report.get('affected_build_stages', [])) or 'none'}",
            f"**Rollout:** {report.get('rollout_strategy', '')}",
            "",
            "## CI recommendations (assistant — apply labels manually)",
            "",
        ]
    )
    ci = report.get("ci_recommendations", {})
    lines.append(f"- **test_type:** `{ci.get('test_type')}` — {ci.get('test_type_reason', '')}")
    inference = ci.get("suite_inference")
    if inference == "superrepo_file_diff":
        lines.append(
            "- **Suite inference:** from changed superrepo components "
            "(file paths or per-directory commits)"
        )
    elif inference == "unresolved":
        lines.append(
            "- **Suite inference:** unresolved — superrepo SHA changed but inner "
            "components unknown (set GITHUB_TOKEN, use --full-manifest)"
        )
    lines.append(f"- **Suggested PR labels:** {', '.join(ci.get('suggested_pr_labels', []))}")
    lines.append(f"- **Suggested test suites:** {', '.join(ci.get('suggested_test_suites', []))}")
    disabled = ci.get("disabled_test_jobs", [])
    if disabled:
        lines.append(f"- **Disabled CI jobs (do not label):** {', '.join(disabled)}")
    lines.append(f"- _{ci.get('notes', '')}_")
    return "\n".join(lines) + "\n"


def llm_prompt(report: dict) -> str:
    compact = {
        "severity": report.get("severity"),
        "blast_radius_score": report.get("blast_radius_score"),
        "changed_components": report.get("changed_components", [])[:20],
        "affected_build_stages": report.get("affected_build_stages"),
        "rollout_strategy": report.get("rollout_strategy"),
        "ci_recommendations": report.get("ci_recommendations"),
        "rationale": report.get("rationale"),
    }
    return (
        "You are a ROCm/TheRock release engineer. Write a concise executive summary "
        "(3-5 bullets) for reviewers based on this structured change impact JSON. "
        "Mention severity, rollout, and CI label recommendations. Do not invent facts.\n\n"
        f"{json.dumps(compact, indent=2)}"
    )


def call_openai_compatible(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str | None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai package: pip install openai")

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return response.choices[0].message.content or ""


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    import requests

    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize change impact report")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=["template", "ollama", "openai", "vllm"],
        default="template",
    )
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL or OpenAI-compatible API (vLLM)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.input.read_text(encoding="utf-8"))

    if args.backend == "template":
        summary = template_summary(report)
    elif args.backend == "ollama":
        summary = call_ollama(llm_prompt(report), args.model, args.base_url)
    elif args.backend == "openai":
        summary = call_openai_compatible(
            llm_prompt(report),
            base_url="https://api.openai.com/v1",
            model=args.model,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif args.backend == "vllm":
        summary = call_openai_compatible(
            llm_prompt(report),
            base_url=args.base_url,
            model=args.model,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    report["executive_summary"] = summary
    args.input.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
