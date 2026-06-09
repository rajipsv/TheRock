# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""GitHub compare API — file paths changed inside superrepo submodule SHAs."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from manifest_bridge import ChangedItem, find_therock_root

SUPERREPO_GITHUB: dict[str, str] = {
    "rocm-libraries": "ROCm/rocm-libraries",
    "rocm-systems": "ROCm/rocm-systems",
}

MAX_FILES_PER_COMPARE = 200


def _github_get(url: str) -> dict[str, Any]:
    repo_root = find_therock_root()
    build_tools = repo_root / "build_tools"
    if str(build_tools) not in sys.path:
        sys.path.insert(0, str(build_tools))
    from github_actions.github_actions_api import gha_send_request

    return gha_send_request(url)


def _component_from_path(path: str) -> str | None:
    norm = path.replace("\\", "/")
    for prefix in ("projects/", "shared/"):
        if norm.startswith(prefix):
            rest = norm[len(prefix):]
            return rest.split("/")[0] if rest else None
    return None


def _ensure_manifest_diff_imports() -> None:
    repo_root = find_therock_root()
    build_tools = repo_root / "build_tools"
    if str(build_tools) not in sys.path:
        sys.path.insert(0, str(build_tools))


def _changed_components_by_commits(
    superrepo_name: str,
    old_sha: str,
    new_sha: str,
    api_base: str,
) -> dict[str, int]:
    """
    Detect changed components via per-directory commit history (same as manifest diff).
    Returns {component_name: commit_count}. Tolerates per-directory API failures.
    """
    if not old_sha or not new_sha or old_sha == new_sha:
        return {}

    _ensure_manifest_diff_imports()
    import urllib.parse

    from generate_manifest_diff_report import (
        determine_status,
        fetch_commits_in_range,
        fetch_superrepo_components,
        MAX_PAGES,
        PER_PAGE,
    )

    status, fetch_start, fetch_end = determine_status(old_sha, new_sha, api_base)
    if status != "changed":
        return {}

    start_components = fetch_superrepo_components(superrepo_name, old_sha, api_base)
    end_components = fetch_superrepo_components(superrepo_name, new_sha, api_base)
    start_set = set(start_components)
    end_set = set(end_components)
    added_paths = end_set - start_set
    removed_paths = start_set - end_set
    all_components = start_set | end_set
    if not all_components:
        return {}

    all_commits = fetch_commits_in_range(
        superrepo_name, fetch_start, fetch_end, api_base
    )
    commit_shas_in_range = {c["sha"] for c in all_commits}

    allocation: dict[str, list[dict]] = {}
    for comp_path in all_components:
        directory = comp_path + "/" if not comp_path.endswith("/") else comp_path
        comp_key = comp_path.rstrip("/")
        allocation[comp_key] = []
        page = 1
        while page <= MAX_PAGES:
            params = {
                "sha": fetch_end,
                "path": directory,
                "per_page": PER_PAGE,
                "page": page,
            }
            url = f"{api_base}/commits?{urllib.parse.urlencode(params)}"
            try:
                data = _github_get(url)
            except Exception as exc:
                print(
                    f"Warning: commit query stopped for {comp_key} "
                    f"(page {page}): {exc}"
                )
                break
            if not isinstance(data, list) or not data:
                break
            commits_found = 0
            for commit in data:
                sha = commit.get("sha")
                if sha in commit_shas_in_range:
                    allocation[comp_key].append(commit)
                    commits_found += 1
                    if sha == fetch_start:
                        break
            if commits_found == 0 and page > 1:
                break
            if len(data) < PER_PAGE:
                break
            page += 1

    changed: dict[str, int] = {}
    for comp_path in all_components:
        comp_key = comp_path.rstrip("/")
        comp_name = comp_path.split("/")[-1]
        if comp_path in added_paths:
            changed[comp_name] = len(allocation.get(comp_key, [])) or 1
        elif comp_path not in removed_paths and allocation.get(comp_key):
            changed[comp_name] = len(allocation[comp_key])
    return changed


def compare_superrepo_paths(
    github_repo: str,
    old_sha: str,
    new_sha: str,
) -> list[str]:
    if not old_sha or not new_sha or old_sha == new_sha:
        return []
    url = f"https://api.github.com/repos/{github_repo}/compare/{old_sha}...{new_sha}"
    try:
        data = _github_get(url)
    except Exception as exc:
        print(f"Warning: superrepo compare failed for {github_repo}: {exc}")
        return []

    files = data.get("files") or []
    paths: list[str] = []
    for entry in files[:MAX_FILES_PER_COMPARE]:
        filename = entry.get("filename")
        if filename:
            paths.append(filename)
    if len(files) > MAX_FILES_PER_COMPARE:
        print(
            f"Warning: truncated superrepo file list to {MAX_FILES_PER_COMPARE} paths"
        )
    return paths


def build_component_path_changes(
    manifest_items: list[ChangedItem],
) -> dict[str, Any]:
    by_superrepo: dict[str, dict[str, Any]] = {}
    component_files: dict[str, list[str]] = defaultdict(list)

    for item in manifest_items:
        if item.kind != "superrepo" or not item.old_sha or not item.new_sha:
            continue
        gh_repo = SUPERREPO_GITHUB.get(item.name)
        if not gh_repo:
            continue

        paths = compare_superrepo_paths(gh_repo, item.old_sha, item.new_sha)
        grouped: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            comp = _component_from_path(path)
            if comp:
                grouped[comp].append(path)
                component_files[comp].append(path)

        api_base = f"https://api.github.com/repos/{gh_repo}"
        commit_components: dict[str, int] = {}
        if not grouped:
            try:
                commit_components = _changed_components_by_commits(
                    item.name, item.old_sha, item.new_sha, api_base
                )
            except Exception as exc:
                print(
                    f"Warning: commit-based component detection failed for "
                    f"{item.name}: {exc}"
                )

        components_meta: dict[str, dict[str, Any]] = {}
        for comp, files in sorted(grouped.items()):
            components_meta[comp] = {
                "file_count": len(files),
                "sample_paths": files[:8],
                "detection": "file_compare",
            }
        for comp, commit_count in sorted(commit_components.items()):
            if comp in components_meta:
                components_meta[comp]["commit_count"] = commit_count
                components_meta[comp]["detection"] = "file_compare+commits"
            else:
                components_meta[comp] = {
                    "file_count": 0,
                    "sample_paths": [],
                    "commit_count": commit_count,
                    "detection": "commit_allocation",
                }
                component_files[comp] = []

        by_superrepo[item.name] = {
            "github_repo": gh_repo,
            "old_sha": item.old_sha,
            "new_sha": item.new_sha,
            "total_files": len(paths),
            "components": components_meta,
        }

    for item in manifest_items:
        if item.kind != "component" or not item.parent:
            continue
        comp = item.name
        if comp in component_files:
            continue
        parent_item = next(
            (
                i
                for i in manifest_items
                if i.kind == "superrepo" and i.name == item.parent
            ),
            None,
        )
        if not parent_item or not parent_item.old_sha or not parent_item.new_sha:
            continue
        gh_repo = SUPERREPO_GITHUB.get(item.parent)
        if not gh_repo:
            continue
        prefix = f"projects/{comp}/"
        alt_prefix = f"shared/{comp}/"
        all_paths = compare_superrepo_paths(
            gh_repo, parent_item.old_sha, parent_item.new_sha
        )
        filtered = [
            p for p in all_paths if p.startswith(prefix) or p.startswith(alt_prefix)
        ]
        if filtered:
            component_files[comp] = filtered

    return {
        "superrepo_diffs": by_superrepo,
        "changed_paths_in_components": {
            comp: {
                "file_count": len(paths),
                "sample_paths": sorted(set(paths))[:12],
            }
            for comp, paths in sorted(component_files.items())
        },
    }
