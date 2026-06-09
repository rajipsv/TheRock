# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from impact_graph import analyze_impact
from manifest_bridge import ChangedItem, find_therock_root


class ImpactGraphTest(unittest.TestCase):
    def test_math_lib_component_raises_severity(self):
        repo_root = find_therock_root()
        items = [
            ChangedItem(
                name="rocblas",
                kind="component",
                status="changed",
                parent="rocm-libraries",
            )
        ]
        impact = analyze_impact(items, repo_root)
        self.assertIn(impact.severity, ("MEDIUM", "MEDIUM-HIGH", "HIGH", "CRITICAL"))
        self.assertGreater(impact.blast_radius_score, 40)
        self.assertTrue(impact.affected_build_stages)

    def test_empty_changes_low_severity(self):
        repo_root = find_therock_root()
        impact = analyze_impact([], repo_root)
        self.assertEqual(impact.severity, "LOW")

    def test_hipdnn_path_change_detected(self):
        repo_root = find_therock_root()
        items = [
            ChangedItem(
                name="ml-libs/artifact-hipdnn.toml",
                kind="path",
                status="changed",
            ),
            ChangedItem(
                name="build_tools/github_actions/fetch_test_configurations.py",
                kind="path",
                status="changed",
            ),
        ]
        impact = analyze_impact(items, repo_root)
        self.assertIn(impact.severity, ("MEDIUM", "MEDIUM-HIGH", "HIGH"))
        self.assertGreater(impact.blast_radius_score, 40)
        self.assertIn("ml-libs", impact.affected_artifact_groups)


if __name__ == "__main__":
    unittest.main()
