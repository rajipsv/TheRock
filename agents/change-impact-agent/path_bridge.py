# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Git path diff — detect CI, packaging, and topology-related file changes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from manifest_bridge import ChangedItem, resolve_git_ref


def get_changed_paths(
    start_ref: str,
    end_ref: str,
    repo_root: Path,
) -> list[str]:
    """Return relative paths changed between two refs (merge-base → head for PRs)."""
    start_sha = resolve_git_ref(start_ref, repo_root)
    end_sha = resolve_git_ref(end_ref, repo_root)
    result = subprocess.run(
        ["git", "diff", "--name-only", start_sha, end_sha],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"git diff failed for {start_sha}..{end_sha}: {result.stderr.strip()}"
        )
    return [p.strip() for p in result.stdout.splitlines() if p.strip()]


def paths_to_changed_items(paths: list[str]) -> list[ChangedItem]:
    return [
        ChangedItem(name=path, kind="path", status="changed")
        for path in paths
    ]


def changed_files_to_dict(paths: list[str]) -> dict[str, Any]:
    return {
        "changed_files": paths,
        "changed_file_count": len(paths),
    }
