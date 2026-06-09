# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Line-level diff insights for known TheRock CI / packaging files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from manifest_bridge import resolve_git_ref

_MATRIX_KEY_RE = re.compile(r'"([a-zA-Z0-9_-]+)":\s*\{')
_TIMEOUT_RE = re.compile(r'"timeout_minutes":\s*(\d+)')
_EXCLUDE_LINE_RE = re.compile(r'["\']([^"\']+)["\']')


def _git_diff_file(
    start_sha: str, end_sha: str, path: str, repo_root: Path
) -> str:
    result = subprocess.run(
        ["git", "diff", start_sha, end_sha, "--", path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _git_file_at(ref: str, path: str, repo_root: Path) -> str:
    sha = resolve_git_ref(ref, repo_root)
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _active_matrix_keys(source: str) -> set[str]:
    keys: set[str] = set()
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        match = _MATRIX_KEY_RE.search(line)
        if match:
            keys.add(match.group(1))
    return keys


def _matrix_timeouts(source: str) -> dict[str, int]:
    """Map test_matrix keys to GHA timeout_minutes (wrapper ceiling, not per-ctest)."""
    timeouts: dict[str, int] = {}
    current_key: str | None = None
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        key_match = _MATRIX_KEY_RE.match(stripped)
        if key_match:
            current_key = key_match.group(1)
            continue
        if current_key:
            timeout_match = _TIMEOUT_RE.search(stripped)
            if timeout_match:
                timeouts[current_key] = int(timeout_match.group(1))
    return timeouts


def analyze_test_matrix_diff(
    start_ref: str,
    end_ref: str,
    repo_root: Path,
    path: str = "build_tools/github_actions/fetch_test_configurations.py",
) -> dict[str, Any]:
    start_sha = resolve_git_ref(start_ref, repo_root)
    end_sha = resolve_git_ref(end_ref, repo_root)
    diff = _git_diff_file(start_sha, end_sha, path, repo_root)
    if not diff:
        return {}

    start_keys = _active_matrix_keys(_git_file_at(start_ref, path, repo_root))
    end_keys = _active_matrix_keys(_git_file_at(end_ref, path, repo_root))
    disabled = sorted(start_keys - end_keys)
    enabled = sorted(end_keys - start_keys)

    notes: list[str] = []
    if disabled:
        notes.append(
            f"Test matrix jobs removed or commented out: {', '.join(disabled)}"
        )
    if enabled:
        notes.append(f"Test matrix jobs added or re-enabled: {', '.join(enabled)}")

    start_timeouts = _matrix_timeouts(_git_file_at(start_ref, path, repo_root))
    end_timeouts = _matrix_timeouts(_git_file_at(end_ref, path, repo_root))
    timeout_changes: list[dict[str, Any]] = []
    timeout_changed_jobs: list[str] = []
    all_jobs = set(start_timeouts) | set(end_timeouts)
    for job in sorted(all_jobs):
        old = start_timeouts.get(job)
        new = end_timeouts.get(job)
        if old != new and old is not None and new is not None:
            timeout_changes.append({"job": job, "old_minutes": old, "new_minutes": new})
            timeout_changed_jobs.append(job)
            notes.append(
                f"GHA wrapper timeout for `{job}`: {old} → {new} minutes "
                "(per-test limits still from test_categories.yaml)"
            )

    return {
        "file": path,
        "disabled_test_jobs": disabled,
        "enabled_test_jobs": enabled,
        "timeout_changes": timeout_changes,
        "timeout_changed_jobs": timeout_changed_jobs,
        "notes": notes,
    }


def analyze_artifact_toml_diff(
    start_ref: str,
    end_ref: str,
    repo_root: Path,
    path: str,
) -> dict[str, Any]:
    start_sha = resolve_git_ref(start_ref, repo_root)
    end_sha = resolve_git_ref(end_ref, repo_root)
    diff = _git_diff_file(start_sha, end_sha, path, repo_root)
    if not diff:
        return {}

    added_excludes: list[str] = []
    removed_excludes: list[str] = []

    for line in diff.splitlines():
        if not line or line.startswith("@@") or line.startswith("diff "):
            continue
        is_add = line.startswith("+") and not line.startswith("+++")
        is_remove = line.startswith("-") and not line.startswith("---")
        if not (is_add or is_remove):
            continue
        body = line[1:].strip()
        if "exclude" not in body and "**" not in body and "share/" not in body:
            continue
        for match in _EXCLUDE_LINE_RE.finditer(body):
            pattern = match.group(1)
            if "/" in pattern or "**" in pattern:
                if is_add:
                    added_excludes.append(pattern)
                elif is_remove:
                    removed_excludes.append(pattern)

    notes: list[str] = []
    if added_excludes:
        notes.append(f"Artifact exclude patterns added: {', '.join(added_excludes)}")
    if removed_excludes:
        notes.append(f"Artifact exclude patterns removed: {', '.join(removed_excludes)}")

    return {
        "file": path,
        "added_excludes": added_excludes,
        "removed_excludes": removed_excludes,
        "notes": notes,
    }


def analyze_content_diffs(
    start_ref: str,
    end_ref: str,
    changed_paths: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    insights: dict[str, Any] = {
        "test_matrix_changes": None,
        "artifact_toml_changes": [],
        "disabled_test_jobs": [],
        "timeout_changed_jobs": [],
        "notes": [],
    }

    for path in changed_paths:
        norm = path.replace("\\", "/")
        if norm.endswith("fetch_test_configurations.py"):
            matrix = analyze_test_matrix_diff(start_ref, end_ref, repo_root, norm)
            if matrix:
                insights["test_matrix_changes"] = matrix
                insights["disabled_test_jobs"] = matrix.get("disabled_test_jobs", [])
                insights["timeout_changed_jobs"] = matrix.get(
                    "timeout_changed_jobs", []
                )
                insights["notes"].extend(matrix.get("notes", []))
        elif norm.startswith("ml-libs/artifact-") and norm.endswith(".toml"):
            tom = analyze_artifact_toml_diff(start_ref, end_ref, repo_root, norm)
            if tom:
                insights["artifact_toml_changes"].append(tom)
                insights["notes"].extend(tom.get("notes", []))

    insights["notes"] = list(dict.fromkeys(insights["notes"]))
    return insights
