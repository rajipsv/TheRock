# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Deterministic audit: flag manifest/path changes not covered by BUILD_TOPOLOGY.toml."""

from __future__ import annotations

import sys
from pathlib import Path

from ci_mapping import _known_test_keys, _suggest_test_suites
from manifest_bridge import ChangedItem, find_therock_root

BUILD_RELEVANT_PREFIXES = (
    "math-libs/",
    "compiler/",
    "profiler/",
    "ml-libs/",
    "rocm-systems/",
)

CI_PATH_MARKERS = (
    "fetch_test_configurations.py",
    ".github/workflows/",
    "BUILD_TOPOLOGY.toml",
    "build_tools/",
)

SKIP_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ipynb")


def _load_topology(repo_root: Path):
    build_tools = repo_root / "build_tools"
    if str(build_tools) not in sys.path:
        sys.path.insert(0, str(build_tools))
    from _therock_utils.build_topology import get_topology

    return get_topology(repo_root / "BUILD_TOPOLOGY.toml")


def _submodule_to_source_sets(topology) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for source_set in topology.get_source_sets():
        for submodule in topology.get_submodules_for_source_set(source_set.name):
            mapping.setdefault(submodule.name, []).append(source_set.name)
    return mapping


def should_skip_path(path: str) -> bool:
    """Exclude docs, media, and most .github paths from topology gap checks."""
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    name = Path(normalized).name.lower()

    if lower.endswith(SKIP_SUFFIXES):
        return True
    if name.startswith("readme"):
        return True
    if lower.startswith("docs/"):
        return True
    if "/notebook/" in lower or lower.startswith("agents/") and "/notebook/" in lower:
        return True
    if lower.startswith(".github/"):
        return not any(marker in lower for marker in (".github/workflows/", "fetch_test_configurations"))
    return False


def is_build_relevant_path(path: str) -> bool:
    lower = path.replace("\\", "/").lower()
    if any(lower.startswith(prefix) for prefix in BUILD_RELEVANT_PREFIXES):
        return True
    if "hipdnn" in lower:
        return True
    if any(marker in lower for marker in CI_PATH_MARKERS):
        return True
    return False


def path_mapped_by_topology(path: str, submodule_map: dict[str, list[str]]) -> bool:
    lower = path.replace("\\", "/").lower()
    if any(
        lower.startswith(prefix)
        for prefix in ("math-libs/", "compiler/", "profiler/", "ml-libs/", "rocm-systems/")
    ):
        return True
    if "hipdnn" in lower:
        return True
    top = lower.split("/")[0]
    if top in submodule_map:
        return True
    if any(marker in lower for marker in CI_PATH_MARKERS):
        return True
    return False


def path_mapped_by_ci(path: str, known_keys: set[str]) -> bool:
    items = [ChangedItem(name=path, kind="path", status="changed")]
    return bool(_suggest_test_suites(items, known_keys))


def audit_topology_gaps(
    changed_items: list[ChangedItem],
    changed_paths: list[str],
    repo_root: Path | None = None,
) -> list[str]:
    """Return natural-language warnings for topology coverage gaps."""
    repo_root = repo_root or find_therock_root()
    topology = _load_topology(repo_root)
    submodule_map = _submodule_to_source_sets(topology)
    known_keys = _known_test_keys(repo_root)
    warnings: list[str] = []

    for path in changed_paths:
        norm = path.replace("\\", "/")
        if norm.endswith("BUILD_TOPOLOGY.toml") or norm == "BUILD_TOPOLOGY.toml":
            warnings.append(
                "BUILD_TOPOLOGY.toml was modified — verify source_set, artifact group, "
                "and build stage entries match the code change."
            )
            break

    seen_submodules: set[str] = set()
    for item in changed_items:
        if item.kind not in ("submodule", "superrepo"):
            continue
        if item.name in seen_submodules:
            continue
        seen_submodules.add(item.name)
        if item.name not in submodule_map:
            warnings.append(
                f"Warning: Submodule/superrepo `{item.name}` changed but is not listed in "
                "BUILD_TOPOLOGY.toml — blast radius may be understated; add a topology entry "
                "in a follow-up PR."
            )

    seen_paths: set[str] = set()
    for path in changed_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if should_skip_path(path):
            continue
        if not is_build_relevant_path(path):
            continue
        if path_mapped_by_topology(path, submodule_map):
            continue
        if path_mapped_by_ci(path, known_keys):
            continue
        warnings.append(
            f"Warning: Changed path `{path}` is under a build-related area but has no "
            "BUILD_TOPOLOGY.toml mapping or recognized test-suite keyword — review whether "
            "topology or CI mapping needs an update."
        )

    return warnings
