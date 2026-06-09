#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""List open upstream ROCm/TheRock PRs and optionally run change-impact analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

from github_pr import (
    DEFAULT_UPSTREAM,
    ensure_pr_fetched,
    get_pull_request,
    list_open_pull_requests,
)
from manifest_bridge import find_therock_root

DEFAULT_OUT = AGENT_DIR / "out"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and analyze open upstream TheRock pull requests"
    )
    parser.add_argument(
        "--upstream-repo",
        default=DEFAULT_UPSTREAM,
        help="Upstream GitHub repo (default: ROCm/TheRock)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Analyze a single PR number (skips listing)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="Max open PRs to list or analyze (default: 10)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analyze.py for each selected PR",
    )
    parser.add_argument(
        "--pr-base-ref",
        default="main",
        help="Base branch for merge-base start ref (default: main)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Root output directory (per-PR subdirs pr-N/)",
    )
    parser.add_argument(
        "--full-manifest",
        action="store_true",
        help="Pass --full-manifest to analyze.py (needs GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--therock-root",
        type=Path,
        default=None,
        help="Path to TheRock repo root",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Force git fetch even if local pr-N ref exists",
    )
    return parser.parse_args(argv)


def print_pr_table(prs: list) -> None:
    if not prs:
        print("No open pull requests found.")
        return
    print(f"{'PR':>6}  {'Author':<20}  {'Base':<8}  Title")
    print("-" * 80)
    for pr in prs:
        title = pr.title[:48] + ("..." if len(pr.title) > 48 else "")
        print(f"#{pr.number:<5}  {pr.author:<20}  {pr.base_ref:<8}  {title}")


def run_analyze(
    pr_number: int,
    repo_root: Path,
    output_root: Path,
    pr_base_ref: str,
    upstream_repo: str,
    full_manifest: bool,
    refetch: bool,
) -> int:
    out_dir = output_root / f"pr-{pr_number}"
    ensure_pr_fetched(
        pr_number,
        repo_root,
        upstream_repo=upstream_repo,
        force=refetch,
    )
    cmd = [
        sys.executable,
        str(AGENT_DIR / "analyze.py"),
        "--pr",
        str(pr_number),
        "--pr-base-ref",
        pr_base_ref,
        "--upstream-repo",
        upstream_repo,
        "--output-dir",
        str(out_dir),
        "--therock-root",
        str(repo_root),
    ]
    if full_manifest:
        cmd.append("--full-manifest")
    print(f"\n--- PR #{pr_number} -> {out_dir}")
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.therock_root or find_therock_root()

    try:
        if args.pr is not None:
            pr_info = get_pull_request(args.pr, args.upstream_repo)
            prs = [pr_info]
            if not args.analyze:
                print_pr_table(prs)
        else:
            print(f"Fetching open PRs from {args.upstream_repo} (max {args.max})...")
            prs = list_open_pull_requests(args.upstream_repo, max_results=args.max)
            print_pr_table(prs)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"GitHub API error: {exc}", file=sys.stderr)
        return 1

    if not args.analyze:
        if args.pr is None:
            print("\nUse --analyze to run change-impact analysis on listed PRs.")
        else:
            print("\nUse --analyze to run change-impact analysis on this PR.")
        return 0

    errors = 0
    for pr in prs:
        base_ref = args.pr_base_ref or pr.base_ref
        code = run_analyze(
            pr.number,
            repo_root,
            args.output_dir,
            base_ref,
            args.upstream_repo,
            args.full_manifest,
            args.refetch,
        )
        if code != 0:
            errors += 1

    if errors:
        print(f"\n{errors} PR(s) failed analysis.")
        return 1
    print(f"\nDone. Reports under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
