#!/usr/bin/env python3
# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Poll recent failed GitHub Actions runs and optionally analyze with log-analysis-agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

from analyze_log import analyze_github_run
from github_logs import (
    INGESTED_STATE_FILE,
    load_ingested_run_ids,
    list_failed_runs,
    mark_run_ingested,
)
from presets import get_preset

DEFAULT_OUT = AGENT_DIR / "out"
DEFAULT_REPO = "ROCm/TheRock"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll failed GitHub Actions runs and analyze logs"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo")
    parser.add_argument(
        "--preset",
        default="custom",
        help="Filter by preset: therock_multi_arch, therock_install, therock_pytorch, therock_unit_tests, custom",
    )
    parser.add_argument("--max-runs", type=int, default=3, help="Max new runs to process")
    parser.add_argument("--per-page", type=int, default=30, help="Failed runs to fetch from API")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analyze_log on each matched run (default: list only)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=f"Dedup state file (default: output-dir/{INGESTED_STATE_FILE})",
    )
    parser.add_argument("--skip-ingested", action="store_true", default=True)
    parser.add_argument("--no-skip-ingested", action="store_false", dest="skip_ingested")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        get_preset(args.preset)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    state_path = args.state_file or (args.output_dir / INGESTED_STATE_FILE)
    ingested = load_ingested_run_ids(state_path) if args.skip_ingested else set()

    try:
        runs = list_failed_runs(
            args.repo,
            per_page=args.per_page,
            preset=args.preset if args.preset != "custom" else None,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.preset != "custom":
        from presets import preset_matches_workflow_name

        runs = [r for r in runs if preset_matches_workflow_name(args.preset, r.name)]

    processed = 0
    for run in runs:
        if processed >= args.max_runs:
            break
        if run.id in ingested:
            print(f"Skip run {run.id} (already ingested)")
            continue

        print(f"Run {run.id}: {run.name} ({run.conclusion}) — {run.html_url}")

        if args.analyze:
            out = args.output_dir / f"run-{run.id}"
            try:
                reports = analyze_github_run(
                    run.id,
                    repo=args.repo,
                    output_dir=out,
                    preset=args.preset if args.preset != "custom" else "auto",
                    use_agent=args.agent,
                    max_jobs=args.max_jobs,
                )
                for report in reports:
                    print(
                        f"  Analyzed job {report.get('github_job_id')}: "
                        f"{report.get('errors_count', 0)} errors"
                    )
            except RuntimeError as e:
                print(f"  Failed: {e}", file=sys.stderr)
                continue

        mark_run_ingested(state_path, run.id)
        processed += 1

    if processed == 0:
        print("No new failed runs to process.")
        return 0

    print(f"Processed {processed} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
