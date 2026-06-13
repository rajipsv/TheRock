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

from env_loader import load_agent_env

load_agent_env()

from ci_mapping import build_ci_recommendations
from impact_graph import analyze_impact, impact_to_dict
from topology_audit import audit_topology_gaps
from manifest_bridge import (
    changeset_to_dict,
    compare_manifest,
    find_therock_root,
)
from component_diff_bridge import build_component_path_changes
from content_diff import analyze_content_diffs
from path_bridge import (
    changed_files_to_dict,
    get_changed_paths,
    paths_to_changed_items,
)

DEFAULT_OUT = AGENT_DIR / "out"


def build_report(
    start_ref: str,
    end_ref: str,
    repo_root: Path | None = None,
    full_manifest: bool = False,
) -> dict:
    repo_root = repo_root or find_therock_root()
    changeset = compare_manifest(
        start_ref, end_ref, repo_root, full_manifest=full_manifest
    )
    changed_paths = get_changed_paths(start_ref, end_ref, repo_root)
    path_items = paths_to_changed_items(changed_paths)
    all_items = changeset.items + path_items

    content_insights = analyze_content_diffs(
        start_ref, end_ref, changed_paths, repo_root
    )
    component_paths = build_component_path_changes(changeset.items)

    impact = analyze_impact(all_items, repo_root)
    topology_warnings = audit_topology_gaps(all_items, changed_paths, repo_root)
    if topology_warnings:
        impact.rationale.extend(topology_warnings)
    if content_insights.get("notes"):
        impact.rationale.extend(content_insights["notes"])
    ci, rollout_strategy = build_ci_recommendations(
        all_items,
        impact,
        repo_root,
        content_insights=content_insights,
        changed_paths_in_components=component_paths.get(
            "changed_paths_in_components"
        ),
        superrepo_diffs=component_paths.get("superrepo_diffs"),
    )

    manifest_dict = changeset_to_dict(changeset)
    path_dict = changed_files_to_dict(changed_paths)
    # Include path rows in changed_components for unified display
    manifest_dict["changed_components"] = manifest_dict["changed_components"] + [
        {
            "name": i.name,
            "kind": i.kind,
            "status": i.status,
            "parent": i.parent,
            "old_sha": i.old_sha,
            "new_sha": i.new_sha,
        }
        for i in path_items
    ]
    report = {
        **manifest_dict,
        **path_dict,
        **impact_to_dict(impact),
        "content_insights": content_insights,
        **component_paths,
        "ci_recommendations": ci,
        "topology_warnings": topology_warnings,
        "executive_summary": "",
    }
    if rollout_strategy:
        report["rollout_strategy"] = rollout_strategy
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
    parser.add_argument(
        "--end",
        help="End git ref (e.g. main, HEAD). Not needed with --pr",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Upstream PR number — fetches pull/N/head to pr-N and analyzes vs base",
    )
    parser.add_argument(
        "--upstream-repo",
        default="ROCm/TheRock",
        help="Repo for --pr fetch (default: ROCm/TheRock)",
    )
    parser.add_argument(
        "--pr-base-ref",
        help="PR base branch; resolves merge-base as start (alternative to --start)",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Force git fetch for --pr even if local pr-N ref exists",
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
    parser.add_argument(
        "--full-manifest",
        action="store_true",
        help="Use GitHub API for superrepo component drill-down (needs GITHUB_TOKEN)",
    )
    return parser.parse_args(argv)


def resolve_start_ref(args: argparse.Namespace, repo_root: Path) -> str:
    if args.start:
        return args.start
    if args.pr_base_ref:
        from github_pr import ensure_upstream_ref_fetched, git_merge_base
        from manifest_bridge import resolve_git_ref

        base_ref = ensure_upstream_ref_fetched(
            args.pr_base_ref,
            repo_root,
            upstream_repo=args.upstream_repo,
            depth=200,
        )
        end_sha = resolve_git_ref(args.end, repo_root)
        merge_base = git_merge_base(repo_root, base_ref, end_sha)
        if merge_base is None:
            # Shallow fork clone or stale local branch — refetch upstream base deeper.
            base_ref = ensure_upstream_ref_fetched(
                args.pr_base_ref,
                repo_root,
                upstream_repo=args.upstream_repo,
                force=True,
                depth=500,
            )
            merge_base = git_merge_base(repo_root, base_ref, end_sha)
        if merge_base is None:
            raise RuntimeError(
                f"git merge-base failed for base={base_ref} end={end_sha}. "
                f"Re-run upstream fetch (e.g. git fetch {args.upstream_repo} "
                f"{args.pr_base_ref}:upstream-{args.pr_base_ref.replace('/', '-')} "
                "--depth=500) and retry."
            )
        return merge_base
    raise ValueError("Provide --start or --pr-base-ref")


def prepare_pr_args(args: argparse.Namespace, repo_root: Path) -> None:
    """Resolve --pr into end ref + default pr-base-ref after fetching upstream head."""
    if args.pr is None:
        if not args.end:
            raise ValueError("Provide --end or --pr")
        return

    from github_pr import _token, ensure_pr_fetched, get_pull_request, pr_local_ref

    if not _token():
        raise SystemExit(
            "GITHUB_TOKEN is required for --pr (upstream PR fetch). "
            "Create agents/change-impact-agent/.env from .env.example "
            "(PAT with public_repo; Contents read for superrepo drill-down)."
        )

    ensure_pr_fetched(
        args.pr,
        repo_root,
        upstream_repo=args.upstream_repo,
        force=args.refetch,
    )
    args.end = pr_local_ref(args.pr)
    if not args.pr_base_ref:
        try:
            pr_info = get_pull_request(args.pr, args.upstream_repo)
            args.pr_base_ref = pr_info.base_ref
        except Exception:
            args.pr_base_ref = "main"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.therock_root or find_therock_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_pr_args(args, repo_root)
    if args.full_manifest:
        from github_pr import _token

        if not _token():
            raise SystemExit(
                "GITHUB_TOKEN is required for --full-manifest (superrepo component drill-down). "
                "Create agents/change-impact-agent/.env from .env.example."
            )
    start_ref = resolve_start_ref(args, repo_root)
    print(f"Analyzing {start_ref} -> {args.end} ...")

    report = build_report(
        start_ref, args.end, repo_root, full_manifest=args.full_manifest
    )
    if args.pr is not None:
        report["pr_number"] = args.pr
        report["upstream_repo"] = args.upstream_repo

    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = write_html(report, output_dir)

    print(f"Severity: {report['severity']} (score {report['blast_radius_score']})")
    print(f"Changed files: {report.get('changed_file_count', 0)}")
    print(f"Changed items (manifest + paths): {len(report['changed_components'])}")
    print(f"Affected stages: {', '.join(report['affected_build_stages']) or 'none'}")
    print(f"Suggested labels: {', '.join(report['ci_recommendations']['suggested_pr_labels'])}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
