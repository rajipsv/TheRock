# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""GitHub API helpers for upstream ROCm/TheRock pull requests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

GITHUB_API = "https://api.github.com"
DEFAULT_UPSTREAM = "ROCm/TheRock"


@dataclass
class PullRequestInfo:
    number: int
    title: str
    author: str
    head_sha: str
    base_ref: str
    labels: list[str]
    html_url: str


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def pr_local_ref(pr_number: int) -> str:
    return f"pr-{pr_number}"


def fetch_url_for_repo(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def upstream_tracking_ref(branch: str) -> str:
    """Local ref for an upstream branch (fork single-branch clones often lack main)."""
    return f"upstream-{branch.replace('/', '-')}"


def ref_exists(repo_root: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def ensure_upstream_ref_fetched(
    ref: str,
    repo_root: Path,
    upstream_repo: str = DEFAULT_UPSTREAM,
    force: bool = False,
) -> str:
    """Fetch upstream branch/tag if missing locally. Returns a ref usable for merge-base."""
    if not force and ref_exists(repo_root, ref):
        return ref

    local_ref = upstream_tracking_ref(ref)
    if not force and ref_exists(repo_root, local_ref):
        return local_ref

    fetch_spec = f"{ref}:{local_ref}"
    subprocess.run(
        ["git", "fetch", fetch_url_for_repo(upstream_repo), fetch_spec],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return local_ref


def get_pull_request(
    pr_number: int,
    upstream_repo: str = DEFAULT_UPSTREAM,
) -> PullRequestInfo:
    """Fetch metadata for a single pull request."""
    owner, repo = upstream_repo.split("/", 1)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=_headers(), timeout=60)
    if response.status_code == 403 and not _token():
        raise RuntimeError(
            "GitHub API rate limit — set GITHUB_TOKEN for upstream PR metadata"
        )
    response.raise_for_status()
    item = response.json()
    user = item.get("user") or {}
    labels = [lbl.get("name", "") for lbl in item.get("labels", [])]
    return PullRequestInfo(
        number=item["number"],
        title=item.get("title", ""),
        author=user.get("login", ""),
        head_sha=(item.get("head") or {}).get("sha", ""),
        base_ref=(item.get("base") or {}).get("ref", "main"),
        labels=labels,
        html_url=item.get("html_url", ""),
    )


def ensure_pr_fetched(
    pr_number: int,
    repo_root: Path,
    upstream_repo: str = DEFAULT_UPSTREAM,
    force: bool = False,
) -> str:
    """Fetch upstream pull/N/head into local ref pr-N. Returns local ref name."""
    local_ref = pr_local_ref(pr_number)
    if not force:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", local_ref],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return local_ref

    fetch_ref = f"pull/{pr_number}/head:{local_ref}"
    cmd = [
        "git",
        "fetch",
        fetch_url_for_repo(upstream_repo),
        fetch_ref,
    ]
    subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    return local_ref


def list_open_pull_requests(
    upstream_repo: str = DEFAULT_UPSTREAM,
    max_results: int = 30,
) -> list[PullRequestInfo]:
    """List open pull requests on upstream (paginated up to max_results)."""
    owner, repo = upstream_repo.split("/", 1)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
    params: dict[str, Any] = {"state": "open", "per_page": min(max_results, 100)}
    prs: list[PullRequestInfo] = []

    while url and len(prs) < max_results:
        response = requests.get(url, headers=_headers(), params=params, timeout=60)
        if response.status_code == 403 and not _token():
            raise RuntimeError(
                "GitHub API rate limit — set GITHUB_TOKEN for listing upstream PRs"
            )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            break
        for item in batch:
            if len(prs) >= max_results:
                break
            user = item.get("user") or {}
            labels = [lbl.get("name", "") for lbl in item.get("labels", [])]
            prs.append(
                PullRequestInfo(
                    number=item["number"],
                    title=item.get("title", ""),
                    author=user.get("login", ""),
                    head_sha=(item.get("head") or {}).get("sha", ""),
                    base_ref=(item.get("base") or {}).get("ref", "main"),
                    labels=labels,
                    html_url=item.get("html_url", ""),
                )
            )
        url = None
        link = response.headers.get("Link", "")
        for part in link.split(","):
            if "rel=\"next\"" in part:
                url = part.split(";")[0].strip().strip("<>")
                params = {}
                break

    return prs
