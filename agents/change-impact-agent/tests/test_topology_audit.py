# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from manifest_bridge import ChangedItem, find_therock_root
from topology_audit import (
    audit_topology_gaps,
    is_build_relevant_path,
    path_mapped_by_ci,
    path_mapped_by_topology,
    should_skip_path,
)


class TopologyAuditHelpersTest(unittest.TestCase):
    def test_skip_readme_and_docs(self):
        self.assertTrue(should_skip_path("README.md"))
        self.assertTrue(should_skip_path("docs/architecture/overview.md"))
        self.assertTrue(should_skip_path("assets/logo.png"))

    def test_skip_github_unless_workflow(self):
        self.assertTrue(should_skip_path(".github/ISSUE_TEMPLATE/bug.md"))
        self.assertFalse(should_skip_path(".github/workflows/multi_arch_ci.yml"))

    def test_build_relevant_prefixes(self):
        self.assertTrue(is_build_relevant_path("math-libs/rocblas/CMakeLists.txt"))
        self.assertFalse(is_build_relevant_path("third-party/openmpi/CMakeLists.txt"))


class TopologyAuditIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = find_therock_root()
        self.submodule_map = {"rocm-libraries": ["math-libs"], "rocm-systems": ["rocm-systems"]}

    @patch("topology_audit._load_topology")
    @patch("topology_audit._submodule_to_source_sets")
    @patch("topology_audit._known_test_keys")
    def test_unmapped_submodule_warns(
        self, mock_known_keys, mock_submodule_map, mock_load_topology
    ):
        mock_load_topology.return_value = MagicMock()
        mock_submodule_map.return_value = self.submodule_map
        mock_known_keys.return_value = {"miopen"}

        items = [
            ChangedItem(name="new-math-lib", kind="submodule", status="changed"),
        ]
        warnings = audit_topology_gaps(items, [], self.repo_root)
        self.assertEqual(len(warnings), 1)
        self.assertIn("new-math-lib", warnings[0])
        self.assertIn("BUILD_TOPOLOGY.toml", warnings[0])

    @patch("topology_audit._load_topology")
    @patch("topology_audit._submodule_to_source_sets")
    @patch("topology_audit._known_test_keys")
    def test_mapped_submodule_silent(
        self, mock_known_keys, mock_submodule_map, mock_load_topology
    ):
        mock_load_topology.return_value = MagicMock()
        mock_submodule_map.return_value = self.submodule_map
        mock_known_keys.return_value = {"miopen"}

        items = [
            ChangedItem(name="rocm-libraries", kind="superrepo", status="changed"),
        ]
        warnings = audit_topology_gaps(items, [], self.repo_root)
        self.assertEqual(warnings, [])

    @patch("topology_audit._load_topology")
    @patch("topology_audit._submodule_to_source_sets")
    @patch("topology_audit._known_test_keys")
    def test_readme_path_skipped(
        self, mock_known_keys, mock_submodule_map, mock_load_topology
    ):
        mock_load_topology.return_value = MagicMock()
        mock_submodule_map.return_value = self.submodule_map
        mock_known_keys.return_value = set()

        warnings = audit_topology_gaps([], ["math-libs/rocblas/README.md"], self.repo_root)
        self.assertEqual(warnings, [])

    @patch("topology_audit._load_topology")
    @patch("topology_audit._submodule_to_source_sets")
    @patch("topology_audit._known_test_keys")
    def test_build_topology_edit_warns(
        self, mock_known_keys, mock_submodule_map, mock_load_topology
    ):
        mock_load_topology.return_value = MagicMock()
        mock_submodule_map.return_value = self.submodule_map
        mock_known_keys.return_value = set()

        warnings = audit_topology_gaps([], ["BUILD_TOPOLOGY.toml"], self.repo_root)
        self.assertEqual(len(warnings), 1)
        self.assertIn("BUILD_TOPOLOGY.toml was modified", warnings[0])

    @patch("topology_audit._load_topology")
    @patch("topology_audit._submodule_to_source_sets")
    @patch("topology_audit._known_test_keys")
    def test_unmapped_build_path_warns(
        self, mock_known_keys, mock_submodule_map, mock_load_topology
    ):
        mock_load_topology.return_value = MagicMock()
        mock_submodule_map.return_value = self.submodule_map
        mock_known_keys.return_value = set()

        path = "math-libs/brand-new-lib/src/foo.c"
        with patch("topology_audit.path_mapped_by_topology", return_value=False):
            with patch("topology_audit.path_mapped_by_ci", return_value=False):
                warnings = audit_topology_gaps([], [path], self.repo_root)
        self.assertEqual(len(warnings), 1)
        self.assertIn(path, warnings[0])

    def test_math_libs_path_mapped_by_topology(self):
        self.assertTrue(
            path_mapped_by_topology("math-libs/rocblas/CMakeLists.txt", self.submodule_map)
        )

    def test_ci_keyword_path_mapped(self):
        known = {"miopen"}
        self.assertTrue(
            path_mapped_by_ci("projects/miopen/src/foo.cpp", known)
        )


if __name__ == "__main__":
    unittest.main()
