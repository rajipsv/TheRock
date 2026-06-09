#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Change Impact Agent — manifest diff + topology → impact report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from ci_mapping import build_ci_recommendations
from impact_graph import analyze_impact, impact_to_dict
from manifest_bridge import (
    changeset_to_dict,
    compare_manifest,
    find_therock_root,
)

DEFAULT_OUT = AGENT_DIR / "out"


def build_report(
    start_ref: str,
    end_ref: str,
    repo_root: Path | None = None,
) -> dict:
    repo_root = repo_root or find_therock_root()
    changeset = compare_manifest(start_ref, end_ref, repo_root)
    impact = analyze_impact(changeset.items, repo_root)
    ci = build_ci_recommendations(changeset.items, impact, repo_root)

    report = {
        **changeset_to_dict(changeset),
        **impact_to_dict(impact),
        "ci_recommendations": ci,
        "executive_summary": "",
    }
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGENTS_030 Change Impact Analysis for TheRock"
    )
    parser.add_argument("--start", help="Start git ref (required unless --pr-base-ref)")
    parser.add_argument("--end", required=True, help="End git ref (e.g. main, HEAD)")
    parser.add_argument(
        "--pr-base-ref",
        help="PR base branch; resolves merge-base as start (alternative to --start)",
    )
    parser.add_argument(
        "--therock-root",
        type=Path,
        default=None,
        help="Path to TheRock repo root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for report.json and report.html",
    )
    return parser.parse_args(argv)


def resolve_start_ref(args: argparse.Namespace, repo_root: Path) -> str:
    if args.start:
        return args.start
    if args.pr_base_ref:
        from manifest_bridge import resolve_git_ref

        end_sha = resolve_git_ref(args.end, repo_root)
        import subprocess

        result = subprocess.run(
            ["git", "merge-base", args.pr_base_ref, end_sha],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    raise ValueError("Provide --start or --pr-base-ref")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.therock_root or find_therock_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ref = resolve_start_ref(args, repo_root)
    print(f"Analyzing {start_ref} -> {args.end} ...")

    report = build_report(start_ref, args.end, repo_root)

    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = write_html(report, output_dir)

    print(f"Severity: {report['severity']} (score {report['blast_radius_score']})")
    print(f"Changed items: {len(report['changed_components'])}")
    print(f"Affected stages: {', '.join(report['affected_build_stages']) or 'none'}")
    print(f"Suggested labels: {', '.join(report['ci_recommendations']['suggested_pr_labels'])}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
