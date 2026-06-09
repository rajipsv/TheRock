# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from ci_mapping import (
    _component_names_from_path_data,
    _is_coarse_superrepo_bump,
    _rollout_strategy_for_ci,
    _suggest_test_suites_for_component_names,
    build_ci_recommendations,
)
from impact_graph import ImpactResult
from manifest_bridge import ChangedItem


class CiMappingTest(unittest.TestCase):
    def test_coarse_superrepo_detected(self):
        items = [
            ChangedItem(
                name="rocm-libraries",
                kind="superrepo",
                status="changed",
            )
        ]
        self.assertTrue(_is_coarse_superrepo_bump(items))

    def test_component_superrepo_not_coarse(self):
        items = [
            ChangedItem(
                name="miopen",
                kind="component",
                status="changed",
                parent="rocm-libraries",
            )
        ]
        self.assertFalse(_is_coarse_superrepo_bump(items))

    def test_component_names_from_path_data(self):
        names = _component_names_from_path_data(
            {"miopen": {"file_count": 3}, "rocblas": {"file_count": 1}},
            {
                "rocm-libraries": {
                    "components": {"hipfft": {"file_count": 2}},
                }
            },
        )
        self.assertEqual(names, ["hipfft", "miopen", "rocblas"])

    def test_suggest_suites_for_component_names_only_matching(self):
        known = {"miopen", "rocblas", "hipdnn"}
        suites = _suggest_test_suites_for_component_names(
            ["miopen", "some-unknown-lib"], known
        )
        self.assertEqual(suites, ["miopen"])

    def test_rollout_without_suites(self):
        impact = ImpactResult(severity="MEDIUM")
        rollout = _rollout_strategy_for_ci(impact, [], "unresolved")
        self.assertIn("--full-manifest", rollout)

    def test_build_ci_coarse_without_path_data(self):
        items = [
            ChangedItem(
                name="rocm-libraries",
                kind="superrepo",
                status="changed",
            )
        ]
        impact = ImpactResult(
            severity="MEDIUM",
            affected_artifact_groups=["math-libs", "ml-libs"],
            affected_build_stages=["math-libs"],
        )
        ci, _ = build_ci_recommendations(items, impact, repo_root=None)
        self.assertEqual(ci["suite_inference"], "unresolved")
        self.assertEqual(ci["suggested_test_suites"], [])
        self.assertEqual(ci["suggested_pr_labels"], ["test_filter:standard"])

    def test_build_ci_from_superrepo_file_diff(self):
        items = [
            ChangedItem(
                name="rocm-libraries",
                kind="superrepo",
                status="changed",
            )
        ]
        impact = ImpactResult(severity="MEDIUM")
        ci, _ = build_ci_recommendations(
            items,
            impact,
            repo_root=None,
            changed_paths_in_components={
                "miopen": {"file_count": 5, "sample_paths": ["projects/miopen/a.cpp"]},
                "rocblas": {"file_count": 2, "sample_paths": []},
            },
        )
        self.assertEqual(ci["suite_inference"], "superrepo_file_diff")
        self.assertEqual(ci["suggested_test_suites"], ["miopen", "rocblas"])
        self.assertIn("test:miopen", ci["suggested_pr_labels"])
        self.assertNotIn("test:hipdnn", ci["suggested_pr_labels"])

    def test_third_party_path_suggests_quick(self):
        items = [
            ChangedItem(
                name="third-party/openmpi/CMakeLists.txt",
                kind="path",
                status="changed",
            )
        ]
        impact = ImpactResult(severity="MEDIUM")
        ci, _ = build_ci_recommendations(items, impact, repo_root=None)
        self.assertEqual(ci["test_type"], "quick")
        self.assertIn("test_filter:quick", ci["suggested_pr_labels"])

    def test_build_ci_from_manifest_component_items(self):
        items = [
            ChangedItem(
                name="rocm-libraries",
                kind="superrepo",
                status="changed",
            ),
            ChangedItem(
                name="miopen",
                kind="component",
                status="changed",
                parent="rocm-libraries",
            ),
        ]
        impact = ImpactResult(severity="MEDIUM")
        ci, _ = build_ci_recommendations(items, impact, repo_root=None)
        self.assertEqual(ci["suite_inference"], "keyword")
        self.assertEqual(ci["suggested_test_suites"], ["miopen"])


if __name__ == "__main__":
    unittest.main()
