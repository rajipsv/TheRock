# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from failure_kb import FailureKnowledgeBase, get_default_kb


class FailureKbTest(unittest.TestCase):
    def test_lookup_connection_timeout(self):
        kb = get_default_kb()
        matches = kb.lookup_known_failure("Connection failed: Connection timeout after 30s", top_k=2)
        self.assertTrue(matches)
        ids = {m.pattern_id for m in matches}
        self.assertIn("db_connection_timeout", ids)

    def test_lookup_oom(self):
        kb = get_default_kb()
        matches = kb.lookup_known_failure("java.lang.OutOfMemoryError: Java heap space", top_k=1)
        self.assertTrue(matches)
        self.assertEqual(matches[0].pattern_id, "oom_heap")

    def test_lookup_runner_shutdown_130(self):
        kb = get_default_kb()
        matches = kb.lookup_known_failure("Process completed with exit code 130", top_k=1)
        self.assertTrue(matches)
        self.assertEqual(matches[0].pattern_id, "gha_runner_shutdown_130")

    def test_format_matches_empty(self):
        kb = FailureKnowledgeBase()
        text = kb.format_matches([])
        self.assertIn("No known failure patterns matched", text)

    def test_list_categories(self):
        kb = get_default_kb()
        text = kb.list_categories()
        self.assertIn("Total patterns:", text)


if __name__ == "__main__":
    unittest.main()
