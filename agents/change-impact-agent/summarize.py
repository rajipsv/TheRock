#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Generate executive summary from report.json (template, Ollama, OpenAI, or vLLM)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

DEFAULT_INPUT = AGENT_DIR / "out" / "report.json"
DEFAULT_OUTPUT = AGENT_DIR / "out" / "executive_summary.md"

LABEL_PATTERN = re.compile(r"\b(test(?:_filter)?:[\w-]+)\b")
SEVERITY_PATTERN = re.compile(
    r"\b(CRITICAL|HIGH|MEDIUM-HIGH|MEDIUM|LOW)\b", re.IGNORECASE
)
SCORE_PATTERN = re.compile(
    r"(?:score|blast\s+radius)\s*(?:of\s*)?(\d+)\s*/\s*100",
    re.IGNORECASE,
)
FALLBACK_FOOTNOTE = "\n\n_LLM summary rejected; showing deterministic template._\n"


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
    lines.extend(
        [
            "",
            "## Topology warnings",
            "",
            "_Deterministic gaps between changed files and BUILD_TOPOLOGY.toml — review at PR time._",
            "",
        ]
    )
    topology_warnings = report.get("topology_warnings") or []
    if topology_warnings:
        for warning in topology_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No topology gaps detected for this change range.")
    return "\n".join(lines) + "\n"


def allowed_pr_labels(report: dict) -> set[str]:
    return set(report.get("ci_recommendations", {}).get("suggested_pr_labels", []))


def extract_mentioned_labels(summary: str) -> set[str]:
    return set(LABEL_PATTERN.findall(summary))


def llm_system_message(report: dict) -> str:
    labels = ", ".join(sorted(allowed_pr_labels(report))) or "none"
    severity = report.get("severity", "UNKNOWN")
    score = report.get("blast_radius_score", 0)
    return (
        "You summarize pre-computed change impact JSON for ROCm/TheRock reviewers. "
        f"Severity must be exactly {severity}. "
        f"Blast radius score must be exactly {score}/100. "
        f"You may only mention these PR labels: {labels}. "
        "Do not invent build stages, test suites, topology facts, or extra labels. "
        "If no labels are listed, do not suggest test:* or test_filter:* labels."
    )


def llm_prompt(report: dict) -> str:
    compact = {
        "severity": report.get("severity"),
        "blast_radius_score": report.get("blast_radius_score"),
        "changed_components": report.get("changed_components", [])[:20],
        "affected_build_stages": report.get("affected_build_stages"),
        "rollout_strategy": report.get("rollout_strategy"),
        "ci_recommendations": report.get("ci_recommendations"),
        "topology_warnings": report.get("topology_warnings", []),
        "rationale": report.get("rationale"),
    }
    return (
        "Write a concise executive summary (3-5 bullets) for reviewers based on this "
        "structured change impact JSON. Mention severity, rollout, and CI label "
        "recommendations exactly as given. Do not invent facts.\n\n"
        f"{json.dumps(compact, indent=2)}"
    )


def llm_correction_prompt(report: dict, summary: str, errors: list[str]) -> str:
    compact = {
        "severity": report.get("severity"),
        "blast_radius_score": report.get("blast_radius_score"),
        "ci_recommendations": report.get("ci_recommendations"),
        "rollout_strategy": report.get("rollout_strategy"),
    }
    return (
        "Your previous summary violated constraints:\n"
        + "\n".join(f"- {err}" for err in errors)
        + "\n\nRewrite the executive summary (3-5 bullets). Fix every violation.\n\n"
        "Previous summary:\n"
        f"{summary}\n\n"
        "Authoritative JSON:\n"
        f"{json.dumps(compact, indent=2)}"
    )


def validate_llm_summary(summary: str, report: dict) -> list[str]:
    errors: list[str] = []
    allowed = allowed_pr_labels(report)
    mentioned = extract_mentioned_labels(summary)
    invented = mentioned - allowed
    if invented:
        errors.append(
            "Invented PR labels: "
            + ", ".join(sorted(invented))
            + f" (allowed: {', '.join(sorted(allowed)) or 'none'})"
        )

    expected_severity = str(report.get("severity", "")).upper()
    for match in SEVERITY_PATTERN.finditer(summary):
        token = match.group(1).upper()
        if token != expected_severity:
            errors.append(
                f"Severity mismatch: summary mentions {token}, "
                f"report severity is {expected_severity}"
            )
            break

    expected_score = report.get("blast_radius_score")
    if expected_score is not None:
        for match in SCORE_PATTERN.finditer(summary):
            cited = int(match.group(1))
            if cited != expected_score:
                errors.append(
                    f"Blast radius score mismatch: summary cites {cited}/100, "
                    f"report score is {expected_score}/100"
                )
                break

    return errors


def call_openai_compatible(
    prompt: str,
    base_url: str,
    model: str,
    api_key: str | None,
    system: str | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai package: pip install openai")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
    )
    return response.choices[0].message.content or ""


def call_ollama(
    prompt: str,
    model: str,
    base_url: str,
    system: str | None = None,
) -> str:
    import requests

    if system:
        prompt = f"System: {system}\n\nUser: {prompt}"

    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _invoke_llm(
    backend: str,
    prompt: str,
    *,
    model: str,
    base_url: str,
    system: str | None,
) -> str:
    if backend == "ollama":
        return call_ollama(prompt, model, base_url, system=system)
    if backend == "openai":
        return call_openai_compatible(
            prompt,
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            system=system,
        )
    if backend == "vllm":
        return call_openai_compatible(
            prompt,
            base_url=base_url,
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            system=system,
        )
    raise ValueError(f"Unknown backend: {backend}")


def generate_llm_summary(
    report: dict,
    backend: str,
    *,
    model: str,
    base_url: str,
    validate: bool = True,
    max_retries: int = 1,
    fallback_template: bool = True,
    llm_call: Callable[[str], str] | None = None,
) -> tuple[str, bool]:
    """Return (summary, used_fallback). used_fallback is True when template replaced LLM."""
    system = llm_system_message(report)
    prompt = llm_prompt(report)

    def default_call(user_prompt: str) -> str:
        return _invoke_llm(
            backend,
            user_prompt,
            model=model,
            base_url=base_url,
            system=system,
        )

    caller = llm_call or default_call
    summary = caller(prompt)

    if not validate:
        return summary, False

    attempts = 0
    while True:
        errors = validate_llm_summary(summary, report)
        if not errors:
            return summary, False
        if attempts >= max_retries:
            break
        attempts += 1
        correction = llm_correction_prompt(report, summary, errors)
        print(
            f"LLM summary validation failed (attempt {attempts}); retrying...",
            file=sys.stderr,
        )
        summary = caller(correction)

    print(
        "LLM summary validation failed after retries: "
        + "; ".join(validate_llm_summary(summary, report)),
        file=sys.stderr,
    )
    if fallback_template:
        return template_summary(report) + FALLBACK_FOOTNOTE, True
    return summary, False


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
    parser.add_argument(
        "--validate-llm",
        dest="validate_llm",
        action="store_true",
        default=True,
        help="Validate LLM output against report.json (default: on)",
    )
    parser.add_argument(
        "--no-validate-llm",
        dest="validate_llm",
        action="store_false",
        help="Skip LLM summary validation",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=1,
        help="Retries after validation failure (LLM backends only)",
    )
    parser.add_argument(
        "--fallback-template",
        dest="fallback_template",
        action="store_true",
        default=True,
        help="Fall back to template summary if validation fails (default: on)",
    )
    parser.add_argument(
        "--no-fallback-template",
        dest="fallback_template",
        action="store_false",
        help="Exit with error if LLM validation fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.input.read_text(encoding="utf-8"))

    if args.backend == "template":
        summary = template_summary(report)
    else:
        summary, used_fallback = generate_llm_summary(
            report,
            args.backend,
            model=args.model,
            base_url=args.base_url,
            validate=args.validate_llm,
            max_retries=args.llm_max_retries,
            fallback_template=args.fallback_template,
        )
        if args.validate_llm and not used_fallback:
            errors = validate_llm_summary(summary, report)
            if errors and not args.fallback_template:
                print("LLM summary validation failed:", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    report["executive_summary"] = summary
    args.input.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
