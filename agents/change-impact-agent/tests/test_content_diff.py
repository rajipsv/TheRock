# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from content_diff import _active_matrix_keys, analyze_content_diffs
from manifest_bridge import find_therock_root


class ContentDiffTest(unittest.TestCase):
    def test_active_matrix_keys_ignores_commented(self):
        source = '''
    "hipdnn": {
        "job_name": "hipdnn",
    },
    # "hipdnn_python_bindings": {
    #     "job_name": "hipdnn_python_bindings",
    # },
'''
        keys = _active_matrix_keys(source)
        self.assertIn("hipdnn", keys)
        self.assertNotIn("hipdnn_python_bindings", keys)

    def test_pr5688_content_if_ref_available(self):
        repo_root = find_therock_root()
        try:
            from manifest_bridge import resolve_git_ref

            resolve_git_ref("pr-5688", repo_root)
            start = "31b738c3f3fc0bd7be498befe72a7920b2324853"
            end = "pr-5688"
        except ValueError:
            self.skipTest("pr-5688 ref not fetched locally")

        paths = [
            "build_tools/github_actions/fetch_test_configurations.py",
            "ml-libs/artifact-hipdnn.toml",
        ]
        insights = analyze_content_diffs(start, end, paths, repo_root)
        disabled = insights.get("disabled_test_jobs", [])
        self.assertIn("hipdnn_python_bindings", disabled)

    def test_pr5572_miopen_timeout_if_ref_available(self):
        repo_root = find_therock_root()
        try:
            from manifest_bridge import resolve_git_ref

            resolve_git_ref("pr-5572", repo_root)
            start = "7b7b238e3e0c2e98436ea230b5114a5b6b946abf"
            end = "pr-5572"
        except ValueError:
            self.skipTest("pr-5572 ref not fetched locally")

        paths = ["build_tools/github_actions/fetch_test_configurations.py"]
        insights = analyze_content_diffs(start, end, paths, repo_root)
        self.assertIn("miopen", insights.get("timeout_changed_jobs", []))
        changes = (insights.get("test_matrix_changes") or {}).get(
            "timeout_changes", []
        )
        miopen = next((c for c in changes if c["job"] == "miopen"), None)
        self.assertIsNotNone(miopen)
        self.assertEqual(miopen["old_minutes"], 60)
        self.assertEqual(miopen["new_minutes"], 120)


if __name__ == "__main__":
    unittest.main()
