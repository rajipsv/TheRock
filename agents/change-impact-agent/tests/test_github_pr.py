"""Tests for upstream ref fetch helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from github_pr import (
    ensure_upstream_ref_fetched,
    git_merge_base,
    upstream_tracking_ref,
)


def test_upstream_tracking_ref_slashes() -> None:
    assert upstream_tracking_ref("main") == "upstream-main"
    assert upstream_tracking_ref("release/6.4") == "upstream-release-6.4"


def test_ensure_upstream_ref_uses_tracking_ref_not_local_main(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".git").mkdir()

    def fake_run(cmd, cwd=None, check=True, capture_output=True, text=True):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            ref = cmd[3]
            if ref in {"main", "upstream-main"}:
                result.returncode = 0
            else:
                result.returncode = 1
        return result

    with patch("github_pr.subprocess.run", side_effect=fake_run) as run:
        ref = ensure_upstream_ref_fetched("main", repo_root)

    assert ref == "upstream-main"
    fetch_calls = [c.args[0] for c in run.call_args_list if c.args[0][1] == "fetch"]
    assert not fetch_calls  # upstream-main already present — no fetch needed


def test_ensure_upstream_ref_fetches_when_tracking_missing(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".git").mkdir()

    def fake_run(cmd, cwd=None, check=True, capture_output=True, text=True):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            ref = cmd[3]
            if ref == "main":
                result.returncode = 0
            else:
                result.returncode = 1
        return result

    with patch("github_pr.subprocess.run", side_effect=fake_run) as run:
        ref = ensure_upstream_ref_fetched("main", repo_root, depth=200)

    assert ref == "upstream-main"
    fetch_calls = [c.args[0] for c in run.call_args_list if c.args[0][1] == "fetch"]
    assert len(fetch_calls) == 1
    assert fetch_calls[0] == [
        "git",
        "fetch",
        "https://github.com/ROCm/TheRock.git",
        "main:upstream-main",
        "--depth=200",
    ]


def test_ensure_upstream_ref_accepts_existing_upstream_main(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".git").mkdir()

    def fake_run(cmd, cwd=None, check=True, capture_output=True, text=True):
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            if cmd[3] == "upstream-main":
                result.returncode = 0
            else:
                result.returncode = 1
        return result

    with patch("github_pr.subprocess.run", side_effect=fake_run) as run:
        ref = ensure_upstream_ref_fetched("upstream-main", repo_root)

    assert ref == "upstream-main"
    fetch_calls = [c for c in run.call_args_list if c.args[0][1] == "fetch"]
    assert not fetch_calls


def test_git_merge_base_returns_none_on_failure(tmp_path: Path) -> None:
    with patch("github_pr.subprocess.run") as run:
        run.return_value = MagicMock(returncode=1, stdout="", stderr="fatal")
        assert git_merge_base(tmp_path, "upstream-main", "pr-5480") is None
