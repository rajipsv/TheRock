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
    "rocjpeg": "rocjpeg",
    "rocdecode": "rocdecode",
    "amdsmi": "amdsmi",
}


def _known_test_keys(repo_root: Path) -> set[str]:
    if str(repo_root / "build_tools") not in sys.path:
        sys.path.insert(0, str(repo_root / "build_tools"))
    try:
        from github_actions.fetch_test_configurations import test_matrix

        return set(test_matrix.keys())
    except ImportError:
        return set(_COMPONENT_TO_TEST_KEY.values())


def _suggest_test_suites(changed_items: list[ChangedItem], known_keys: set[str]) -> list[str]:
    suggested: set[str] = set()
    for item in changed_items:
        name = item.name.lower()
        for keyword, key in _COMPONENT_TO_TEST_KEY.items():
            if keyword in name.replace("-", "").replace("_", "") and key in known_keys:
                suggested.add(key)
        for part in re.split(r"[/\\]", name):
            part_norm = part.lower().replace("-", "").replace("_", "")
            for keyword, key in _COMPONENT_TO_TEST_KEY.items():
                if keyword in part_norm and key in known_keys:
                    suggested.add(key)
    return sorted(suggested)


def _suggest_test_type(
    changed_items: list[ChangedItem],
    severity: str,
) -> tuple[str, str]:
    if not changed_items:
        return "quick", "No manifest changes in range"
    submodule_like = [
        i for i in changed_items if i.kind in ("submodule", "superrepo") or i.parent
    ]
    if severity in ("CRITICAL", "HIGH"):
        return "full", f"Severity {severity} — recommend full test depth"
    if submodule_like:
        return "standard", "Submodule or superrepo component change (matches TheRock CI policy)"
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
) -> dict:
    repo_root = repo_root or find_therock_root()
    known_keys = _known_test_keys(repo_root)
    test_suites = _suggest_test_suites(changed_items, known_keys)
    test_type, test_type_reason = _suggest_test_type(changed_items, impact.severity)
    labels = _suggest_pr_labels(test_suites, test_type, impact.severity)

    return {
        "test_type": test_type,
        "test_type_reason": test_type_reason,
        "suggested_pr_labels": labels,
        "suggested_test_suites": test_suites,
        "notes": (
            "TheRock CI defaults to all test suites (PROJECTS_TO_TEST=*) unless "
            "test:* labels are set. These labels narrow scope to recommended suites."
        ),
        "affected_build_stages": impact.affected_build_stages,
    }
