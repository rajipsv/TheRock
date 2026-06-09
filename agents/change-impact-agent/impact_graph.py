# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Build topology impact analysis for manifest changes."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from manifest_bridge import ChangedItem, find_therock_root


@dataclass
class ImpactResult:
    affected_source_sets: list[str] = field(default_factory=list)
    affected_artifact_groups: list[str] = field(default_factory=list)
    affected_build_stages: list[str] = field(default_factory=list)
    severity: str = "LOW"
    blast_radius_score: int = 0
    rollout_strategy: str = ""
    rationale: list[str] = field(default_factory=list)


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


def _reverse_group_dependencies(topology) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for group in topology.get_artifact_groups():
        for dep in group.artifact_group_deps:
            reverse.setdefault(dep, []).append(group.name)
    return reverse


def _expand_dependent_groups(seed: Iterable[str], reverse_deps: dict[str, list[str]]) -> set[str]:
    result = set(seed)
    queue = list(seed)
    while queue:
        current = queue.pop()
        for dependent in reverse_deps.get(current, []):
            if dependent not in result:
                result.add(dependent)
                queue.append(dependent)
    return result


def _score_severity(
    changed_items: list[ChangedItem],
    source_sets: set[str],
    groups: set[str],
) -> tuple[str, int, list[str]]:
    rationale: list[str] = []
    names_lower = " ".join(
        f"{i.parent or ''}/{i.name}".lower() for i in changed_items
    )

    if any(
        k in names_lower
        for k in ("llvm", "compiler", "hipify", "spirv", "amd-llvm")
    ):
        rationale.append("Compiler toolchain change detected")
        return "CRITICAL", 95, rationale

    if any(k in names_lower for k in ("rocm-systems", "rocr", "hip-runtime", "hsa")):
        rationale.append("Core runtime / rocm-systems change detected")
        return "HIGH", 85, rationale

    if "compiler" in groups or "core-runtime" in groups:
        rationale.append("Affected artifact groups include core/compiler paths")
        return "HIGH", 80, rationale

    path_text = " ".join(
        i.name.lower() for i in changed_items if i.kind == "path"
    )
    combined = f"{names_lower} {path_text}"

    if any(k in combined for k in ("miopen", "hipdnn", "ml-lib", "artifact-hipdnn")):
        rationale.append("ML library / hipDNN path change detected")
        return "MEDIUM-HIGH", 70, rationale

    if "fetch_test_configurations" in combined or "github_actions" in combined:
        rationale.append("CI test matrix or GitHub Actions configuration changed")
        if not any(i.kind in ("submodule", "superrepo", "component") for i in changed_items):
            return "MEDIUM", 40, rationale
        return "MEDIUM", 50, rationale

    if any(k in combined for k in ("packaging", "artifact-", ".toml")):
        rationale.append("Packaging or artifact manifest path change detected")
        return "MEDIUM", 45, rationale

    if "math-libs" in source_sets or any(
        k in names_lower for k in ("blas", "fft", "sparse", "solver", "rocblas", "hipblas")
    ):
        rationale.append("Math library change detected")
        return "MEDIUM", 55, rationale

    if any(k in names_lower for k in ("patch", "doc", "readme")):
        rationale.append("Low-risk path or documentation change")
        return "LOW", 20, rationale

    if changed_items:
        rationale.append("Changes detected with moderate default scoring")
        return "MEDIUM", 45, rationale

    rationale.append("No manifest or path changes in range")
    return "LOW", 10, rationale


def _rollout_for_severity(severity: str) -> str:
    if severity in ("CRITICAL", "HIGH"):
        return "Canary one GPU family (e.g. gfx110X) then full multi_arch_ci matrix"
    if severity in ("MEDIUM", "MEDIUM-HIGH"):
        return "Canary gfx family + component-specific test labels"
    return "Quick or standard tests; manifest-diff sibling job sufficient"


def analyze_impact(changed_items: list[ChangedItem], repo_root: Path | None = None) -> ImpactResult:
    repo_root = repo_root or find_therock_root()
    topology = _load_topology(repo_root)

    submodule_map = _submodule_to_source_sets(topology)
    source_set_to_groups = topology.get_source_set_to_artifact_groups()
    group_to_stages = topology.get_artifact_group_to_build_stages()
    reverse_deps = _reverse_group_dependencies(topology)

    source_sets: set[str] = set()
    path_seed_groups: set[str] = set()
    for item in changed_items:
        if item.kind in ("submodule", "superrepo"):
            for ss in submodule_map.get(item.name, []):
                source_sets.add(ss)
        if item.parent in submodule_map:
            for ss in submodule_map.get(item.parent, []):
                source_sets.add(ss)
        if item.kind == "component" and item.parent == "rocm-libraries":
            source_sets.add("rocm-libraries")
            source_sets.add("math-libs")
        if item.kind == "component" and item.parent == "rocm-systems":
            source_sets.add("rocm-systems")
        if item.kind == "path":
            path = item.name.replace("\\", "/").lower()
            if path.startswith("math-libs/"):
                source_sets.add("math-libs")
                path_seed_groups.add("math-libs")
            if path.startswith("ml-libs/") or "hipdnn" in path:
                path_seed_groups.add("ml-libs")
            if path.startswith("profiler/"):
                path_seed_groups.add("profiler")
            if path.startswith("compiler/"):
                source_sets.add("compilers")
                path_seed_groups.add("compiler")
    seed_groups: set[str] = set(path_seed_groups)
    for ss in source_sets:
        for group in source_set_to_groups.get(ss, []):
            seed_groups.add(group)

    has_manifest_change = any(
        i.kind in ("submodule", "superrepo", "component") for i in changed_items
    )
    # Path-only PRs (CI/packaging/config): map direct groups, skip full reverse blast radius.
    if has_manifest_change:
        all_groups = _expand_dependent_groups(seed_groups, reverse_deps)
    else:
        all_groups = set(seed_groups)

    stages: set[str] = set()
    for group in all_groups:
        for stage in group_to_stages.get(group, []):
            stages.add(stage)

    severity, score, rationale = _score_severity(changed_items, source_sets, all_groups)

    return ImpactResult(
        affected_source_sets=sorted(source_sets),
        affected_artifact_groups=sorted(all_groups),
        affected_build_stages=sorted(stages),
        severity=severity,
        blast_radius_score=score,
        rollout_strategy=_rollout_for_severity(severity),
        rationale=rationale,
    )


def impact_to_dict(impact: ImpactResult) -> dict:
    return {
        "affected_source_sets": impact.affected_source_sets,
        "affected_artifact_groups": impact.affected_artifact_groups,
        "affected_build_stages": impact.affected_build_stages,
        "severity": impact.severity,
        "blast_radius_score": impact.blast_radius_score,
        "rollout_strategy": impact.rollout_strategy,
        "rationale": impact.rationale,
    }
