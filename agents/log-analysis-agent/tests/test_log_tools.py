# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from log_tools import (
    LogSession,
    extract_stack_traces,
    get_log_stats,
    grep_error_keyword,
    grep_log,
    run_tool_only_analysis,
)

FIXTURE = AGENT_DIR / "tests" / "fixtures" / "example.log"


class LogToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = LogSession(path=FIXTURE)

    def test_get_log_stats(self):
        stats = get_log_stats(self.session)
        self.assertIn("lines=21", stats)
        self.assertIn("ERROR=", stats)

    def test_grep_error_keyword(self):
        out = grep_error_keyword(self.session, "ERROR", max_matches=5)
        self.assertIn("Connection failed", out)
        self.assertIn(">>", out)

    def test_grep_log_pattern(self):
        out = grep_log(self.session, r"OutOfMemory", max_matches=3)
        self.assertIn("OutOfMemoryError", out)

    def test_extract_stack_traces(self):
        out = extract_stack_traces(self.session, max_traces=5)
        self.assertIn("PaymentProcessor", out)
        self.assertIn("OutOfMemoryError", out)

    def test_run_tool_only_analysis(self):
        result = run_tool_only_analysis(self.session, kb=None)
        self.assertEqual(result["mode"], "tool_only")
        self.assertIn("error_samples", result)
        self.assertIn("stack_traces", result)


if __name__ == "__main__":
    unittest.main()
