#!/usr/bin/env python3
"""List jobs for a GitHub Actions run."""

import argparse
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

import os
import requests

GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def list_jobs(repo: str, run_id: int) -> list[dict]:
    owner, name = repo.split("/", 1)
    jobs: list[dict] = []
    url = f"{GITHUB_API}/repos/{owner}/{name}/actions/runs/{run_id}/jobs?per_page=100"
    while url:
        resp = requests.get(url, headers=_headers(), timeout=120)
        if not resp.ok:
            raise RuntimeError(f"GitHub API jobs: HTTP {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        jobs.extend(data.get("jobs", []))
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            section = part.strip().split(";")
            if len(section) >= 2 and 'rel="next"' in section[1]:
                next_url = section[0].strip().removeprefix("<").removesuffix(">")
                break
        url = next_url
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--repo", default="ROCm/TheRock")
    parser.add_argument("--failed-only", action="store_true")
    args = parser.parse_args()

    jobs = list_jobs(args.repo, args.run_id)
    if args.failed_only:
        jobs = [j for j in jobs if j.get("conclusion") in ("failure", "cancelled")]

    print(f"jobs={len(jobs)}")
    for j in jobs:
        print(
            f"{j['id']}\t{j.get('conclusion','')}\t{j.get('name','')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
