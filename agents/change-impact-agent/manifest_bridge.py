# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Bridge to TheRock manifest diff — resolves git refs and returns structured changes."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPERREPO_NAMES = frozenset({"rocm-libraries", "rocm-systems"})


@dataclass
class ChangedItem:
    name: str
    kind: str  # submodule | superrepo | component
    status: str
    parent: str | None = None
    old_sha: str = ""
    new_sha: str = ""


@dataclass
class ManifestChangeSet:
    start_ref: str
    end_ref: str
    start_sha: str
    end_sha: str
    items: list[ChangedItem] = field(default_factory=list)
    manifest_mode: str = "local_git_pins"
    notes: list[str] = field(default_factory=list)

    @property
    def changed_names(self) -> list[str]:
        return [i.name for i in self.items]


def find_therock_root(start: Path | None = None) -> Path:
    """Locate TheRock repository root from this file or a starting path."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        root = candidate
        while root != root.parent:
            if (root / "BUILD_TOPOLOGY.toml").exists() and (root / "build_tools").exists():
                return root
            root = root.parent
    raise RuntimeError("Could not locate TheRock repository root (BUILD_TOPOLOGY.toml)")


def resolve_git_ref(ref: str, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not resolve git ref '{ref}': {result.stderr.strip()}")
    return result.stdout.strip()


def _ensure_build_tools_path(repo_root: Path) -> None:
    build_tools = repo_root / "build_tools"
    if str(build_tools) not in sys.path:
        sys.path.insert(0, str(build_tools))


def _pins_from_manifest(manifest: dict) -> dict[str, str]:
    pins: dict[str, str] = {}
    for row in manifest.get("submodules", []):
        name = row.get("submodule_name") or row.get("name")
        pin = row.get("pin_sha")
        if name and pin:
            pins[name] = pin
    return pins


def compare_manifest_local(
    start_ref: str, end_ref: str, repo_root: Path
) -> ManifestChangeSet:
    """Compare submodule gitlink SHAs locally — no GitHub API (fast, demo-friendly)."""
    start_sha = resolve_git_ref(start_ref, repo_root)
    end_sha = resolve_git_ref(end_ref, repo_root)

    _ensure_build_tools_path(repo_root)
    from generate_therock_manifest import build_manifest_schema

    start_pins = _pins_from_manifest(build_manifest_schema(repo_root, start_sha))
    end_pins = _pins_from_manifest(build_manifest_schema(repo_root, end_sha))

    items: list[ChangedItem] = []
    all_names = set(start_pins) | set(end_pins)

    for name in sorted(all_names):
        old_sha = start_pins.get(name)
        new_sha = end_pins.get(name)
        if old_sha == new_sha:
            continue
        if old_sha is None:
            status = "added"
        elif new_sha is None:
            status = "removed"
        else:
            status = "changed"

        kind = "superrepo" if name in SUPERREPO_NAMES else "submodule"
        items.append(
            ChangedItem(
                name=name,
                kind=kind,
                status=status,
                old_sha=old_sha or "",
                new_sha=new_sha or "",
            )
        )

    notes = [
        "Local mode: compared top-level submodule SHAs via git (no GitHub API).",
        "Superrepo internal component list requires --full-manifest and GITHUB_TOKEN.",
        "Pick a range that spans submodule bumps (e.g. main~6..main not main~5..main).",
    ]

    return ManifestChangeSet(
        start_ref=start_ref,
        end_ref=end_ref,
        start_sha=start_sha,
        end_sha=end_sha,
        items=items,
        manifest_mode="local_git_pins",
        notes=notes,
    )


def compare_manifest_full(
    start_ref: str, end_ref: str, repo_root: Path
) -> ManifestChangeSet:
    """Full manifest diff via generate_manifest_diff_report (uses GitHub API)."""
    start_sha = resolve_git_ref(start_ref, repo_root)
    end_sha = resolve_git_ref(end_ref, repo_root)

    _ensure_build_tools_path(repo_root)
    from generate_manifest_diff_report import compare_manifests

    diff = compare_manifests(start_sha, end_sha)
    items: list[ChangedItem] = []

    for status_key in ("added", "removed", "changed", "reverted"):
        names = diff.get_status_groups().get(status_key, [])
        for name in names:
            entry = diff.all_items.get(name)
            if entry is None:
                continue
            kind = "superrepo" if name in diff.superrepos else "submodule"
            old_sha = getattr(entry, "start_sha", "") or ""
            new_sha = getattr(entry, "end_sha", "") or getattr(entry, "sha", "") or ""
            items.append(
                ChangedItem(
                    name=name,
                    kind=kind,
                    status=status_key,
                    old_sha=old_sha,
                    new_sha=new_sha,
                )
            )

    for superrepo in diff.superrepos.values():
        for status_key in ("added", "removed", "changed"):
            comp_names = {
                "added": superrepo.added_components,
                "removed": superrepo.removed_components,
                "changed": superrepo.changed_components,
            }[status_key]
            for comp_name in comp_names:
                if any(
                    i.name == comp_name and i.parent == superrepo.name for i in items
                ):
                    continue
                items.append(
                    ChangedItem(
                        name=comp_name,
                        kind="component",
                        status=status_key,
                        parent=superrepo.name,
                    )
                )

    return ManifestChangeSet(
        start_ref=start_ref,
        end_ref=end_ref,
        start_sha=start_sha,
        end_sha=end_sha,
        items=items,
        manifest_mode="full_github_manifest_diff",
        notes=["Full mode: includes superrepo component drill-down via GitHub API."],
    )


def compare_manifest(
    start_ref: str,
    end_ref: str,
    repo_root: Path,
    full_manifest: bool = False,
) -> ManifestChangeSet:
    """Default: fast local git pin diff. Use full_manifest=True for GitHub deep dive."""
    if full_manifest:
        try:
            return compare_manifest_full(start_ref, end_ref, repo_root)
        except Exception as exc:
            print(f"Warning: full manifest diff failed ({exc}); falling back to local mode.")
    return compare_manifest_local(start_ref, end_ref, repo_root)


def changeset_to_dict(changeset: ManifestChangeSet) -> dict[str, Any]:
    return {
        "start_ref": changeset.start_ref,
        "end_ref": changeset.end_ref,
        "start_sha": changeset.start_sha,
        "end_sha": changeset.end_sha,
        "manifest_mode": changeset.manifest_mode,
        "manifest_notes": changeset.notes,
        "changed_components": [
            {
                "name": i.name,
                "kind": i.kind,
                "status": i.status,
                "parent": i.parent,
                "old_sha": i.old_sha,
                "new_sha": i.new_sha,
            }
            for i in changeset.items
        ],
    }
