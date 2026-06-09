# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Map impact analysis to TheRock CI test labels and test_type recommendations."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from impact_graph import ImpactResult
from manifest_bridge import ChangedItem, find_therock_root

_COMPONENT_TO_TEST_KEY: dict[str, str] = {
    "rocblas": "rocblas",
    "hipblaslt": "hipblaslt",
    "hipblas": "hipblas",
    "miopen": "miopen",
    "rocfft": "rocfft",
    "rocprim": "rocprim",
    "rocthrust": "rocthrust",
    "hipfft": "hipfft",
    "rocsparse": "rocsparse",
    "hipsparse": "hipsparse",
    "rocsolver": "rocsolver",
    "hipsolver": "hipsolver",
    "rocrand": "rocrand",
    "hiprand": "hiprand",
    "rccl": "rccl",
    "hipsparselt": "hipsparselt",
    "rocroller": "rocroller",
    "tensilelite": "tensilelite",
    "hipify": "hip-tests",
    "hip": "hip-tests",
    "rocprofiler": "rocprofiler",
    "rocprofiler-systems": "rocprofiler-systems",
    "hipdnn": "hipdnn",
    "hipdnn_install": "hipdnn_install",
    "hipdnn-integration-tests": "hipdnn-integration-tests",
    "hipdnn-samples": "hipdnn-samples",
    "rocjpeg": "rocjpeg",
    "rocdecode": "rocdecode",
    "amdsmi": "amdsmi",
}

_SUPERREPO_NAMES = frozenset({"rocm-libraries", "rocm-systems"})


def _known_test_keys(repo_root: Path) -> set[str]:
    if str(repo_root / "build_tools") not in sys.path:
        sys.path.insert(0, str(repo_root / "build_tools"))
    try:
        from github_actions.fetch_test_configurations import test_matrix

        return set(test_matrix.keys())
    except ImportError:
        return set(_COMPONENT_TO_TEST_KEY.values())


def _keyword_matches(text_norm: str, keyword: str) -> bool:
    if keyword not in text_norm:
        return False
    # Prefer longer keys (hipdnn before hip → hip-tests).
    for other, _ in _COMPONENT_TO_TEST_KEY.items():
        if other == keyword or len(other) <= len(keyword):
            continue
        if other in text_norm and keyword in other:
            return False
    return True


def _suggest_test_suites(changed_items: list[ChangedItem], known_keys: set[str]) -> list[str]:
    suggested: set[str] = set()
    keywords_sorted = sorted(
        _COMPONENT_TO_TEST_KEY.items(), key=lambda pair: -len(pair[0])
    )
    for item in changed_items:
        name = item.name.lower()
        name_norm = name.replace("-", "").replace("_", "")
        for keyword, key in keywords_sorted:
            if _keyword_matches(name_norm, keyword) and key in known_keys:
                suggested.add(key)
        for part in re.split(r"[/\\]", name):
            part_norm = part.lower().replace("-", "").replace("_", "")
            for keyword, key in keywords_sorted:
                if _keyword_matches(part_norm, keyword) and key in known_keys:
                    suggested.add(key)
    return sorted(suggested)


def _is_coarse_superrepo_bump(changed_items: list[ChangedItem]) -> bool:
    """True when only top-level superrepo/submodule pins changed (no inner component names)."""
    manifest_items = [
        i
        for i in changed_items
        if i.kind in ("submodule", "superrepo", "component")
    ]
    if not manifest_items:
        return False
    has_component = any(i.kind == "component" for i in manifest_items)
    if has_component:
        return False
    return any(
        i.kind == "superrepo" and i.name in _SUPERREPO_NAMES for i in manifest_items
    )


def _component_names_from_path_data(
    changed_paths_in_components: dict | None,
    superrepo_diffs: dict | None,
) -> list[str]:
    names: set[str] = set()
    for comp in (changed_paths_in_components or {}):
        names.add(comp)
    for info in (superrepo_diffs or {}).values():
        for comp in (info.get("components") or {}):
            names.add(comp)
    return sorted(names)


def _suggest_test_suites_for_component_names(
    component_names: list[str],
    known_keys: set[str],
) -> list[str]:
    if not component_names:
        return []
    items = [
        ChangedItem(
            name=name,
            kind="component",
            status="changed",
            parent="rocm-libraries",
        )
        for name in component_names
    ]
    return _suggest_test_suites(items, known_keys)


def _rollout_strategy_for_ci(
    impact: ImpactResult,
    test_suites: list[str],
    suite_inference: str,
) -> str:
    if impact.severity in ("CRITICAL", "HIGH"):
        return "Canary one GPU family (e.g. gfx110X) then full multi_arch_ci matrix"
    if impact.severity in ("MEDIUM", "MEDIUM-HIGH"):
        if not test_suites:
            return (
                "Canary gfx family + test_filter:standard "
                "(no test:* inferred — set GITHUB_TOKEN and use --full-manifest)"
            )
        if suite_inference == "unresolved":
            return (
                "Canary gfx family + test_filter:standard "
                "(superrepo SHA changed but inner components unknown)"
            )
        return "Canary gfx family + component-specific test labels"
    return "Quick or standard tests; manifest-diff sibling job sufficient"


def _suggest_test_type(
    changed_items: list[ChangedItem],
    severity: str,
    content_insights: dict | None = None,
) -> tuple[str, str]:
    if not changed_items:
        return "quick", "No manifest or path changes in range"
    submodule_like = [
        i for i in changed_items if i.kind in ("submodule", "superrepo") or i.parent
    ]
    path_items = [i for i in changed_items if i.kind == "path"]
    ci_only = path_items and not submodule_like
    content_insights = content_insights or {}

    if ci_only and content_insights.get("disabled_test_jobs"):
        return (
            "quick",
            "CI config change with test jobs disabled — quick sanity per test_filtering.md",
        )
    if ci_only and (
        content_insights.get("test_matrix_changes")
        or content_insights.get("artifact_toml_changes")
    ):
        return (
            "quick",
            "CI or packaging manifest change only (no submodule bump)",
        )
    if ci_only and all(
        i.name.replace("\\", "/").startswith("third-party/") for i in path_items
    ):
        return (
            "quick",
            "Third-party packaging/config change — quick sanity per test_filtering.md",
        )

    if severity in ("CRITICAL", "HIGH"):
        return "full", f"Severity {severity} — recommend full test depth"
    if submodule_like:
        return "standard", "Submodule or superrepo component change (matches TheRock CI policy)"
    if path_items:
        ci_paths = [
            i.name for i in path_items
            if "github_actions" in i.name or "fetch_test_configurations" in i.name
        ]
        if ci_paths:
            return (
                "quick",
                "CI test matrix or workflow configuration changed — quick sanity + targeted suites",
            )
        return "standard", "Repository path changes (packaging/config/topology manifests)"
    return "quick", "Default quick tests for low-impact change"


def _suggest_pr_labels(test_suites: list[str], test_type: str, severity: str) -> list[str]:
    labels = [f"test:{s}" for s in test_suites]
    labels.append(f"test_filter:{test_type}")
    if severity in ("CRITICAL", "HIGH"):
        labels.append("ci:run-all-archs")
    return sorted(set(labels))


def build_ci_recommendations(
    changed_items: list[ChangedItem],
    impact: ImpactResult,
    repo_root: Path | None = None,
    content_insights: dict | None = None,
    changed_paths_in_components: dict | None = None,
    superrepo_diffs: dict | None = None,
) -> dict:
    repo_root = repo_root or find_therock_root()
    known_keys = _known_test_keys(repo_root)
    content_insights = content_insights or {}
    disabled_jobs = set(content_insights.get("disabled_test_jobs", []))

    keyword_suites = _suggest_test_suites(changed_items, known_keys)
    suite_inference = "keyword" if keyword_suites else "none"

    test_suites = list(keyword_suites)
    path_component_names = _component_names_from_path_data(
        changed_paths_in_components, superrepo_diffs
    )
    path_suites = _suggest_test_suites_for_component_names(
        path_component_names, known_keys
    )
    if path_suites:
        test_suites = sorted(set(test_suites) | set(path_suites))
        suite_inference = "mixed" if keyword_suites else "superrepo_file_diff"

    if _is_coarse_superrepo_bump(changed_items) and not test_suites:
        suite_inference = "unresolved"

    for job in content_insights.get("timeout_changed_jobs", []):
        if job in known_keys:
            test_suites.append(job)
            if suite_inference == "none":
                suite_inference = "keyword"
    test_suites = sorted(set(test_suites))
    test_suites = [s for s in test_suites if s not in disabled_jobs]

    test_type, test_type_reason = _suggest_test_type(
        changed_items, impact.severity, content_insights
    )
    labels = _suggest_pr_labels(test_suites, test_type, impact.severity)
    rollout_strategy = _rollout_strategy_for_ci(impact, test_suites, suite_inference)

    notes = (
        "TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless "
        "test:* labels are set. These labels narrow scope to recommended suites. "
        "Recommendations are test_matrix jobs + test_filter depth — not individual ctest cases."
    )
    if suite_inference == "superrepo_file_diff":
        notes += (
            " test:* labels inferred from changed superrepo components "
            "(file compare or commit allocation). Use --full-manifest for manifest drill-down."
        )
    if suite_inference == "unresolved":
        notes += (
            " Superrepo pin changed but inner components could not be resolved. "
            "Set GITHUB_TOKEN and re-run with --full-manifest."
        )
    if disabled_jobs:
        notes += (
            f" Do not label disabled jobs: {', '.join(sorted(disabled_jobs))}."
        )

    return {
        "test_type": test_type,
        "test_type_reason": test_type_reason,
        "suggested_pr_labels": labels,
        "suggested_test_suites": test_suites,
        "suite_inference": suite_inference,
        "rollout_strategy": rollout_strategy,
        "disabled_test_jobs": sorted(disabled_jobs),
        "notes": notes,
        "affected_build_stages": impact.affected_build_stages,
    }
