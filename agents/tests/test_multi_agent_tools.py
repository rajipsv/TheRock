# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_DIR))

from multi_agent_tools import (
    DEFAULT_DEMO_LOG,
    DEFAULT_DEMO_PR,
    list_demo_assets,
    run_change_impact_for_demo_pr,
    run_change_impact_for_pr,
    run_infrastructure_triage_loop,
    run_log_analysis_for_demo_run,
    run_log_analysis_for_path,
)


class MultiAgentToolsTest(unittest.TestCase):
    def test_list_demo_assets(self):
        text = list_demo_assets()
        self.assertIn("5572", text)

    def test_change_impact_sample(self):
        summary = run_change_impact_for_demo_pr(DEFAULT_DEMO_PR, use_vllm_summary=False)
        self.assertIn("Severity", summary)
        self.assertIn("change-impact-agent", summary)
        self.assertIn("test:", summary.lower())

    def test_log_analysis_sample(self):
        summary = run_log_analysis_for_path(str(DEFAULT_DEMO_LOG), use_vllm_summary=False)
        self.assertIn("tool_only", summary)
        self.assertIn("Errors", summary)
        self.assertTrue(
            "rocsparse" in summary.lower()
            or "HIP" in summary
            or "27710372755" in summary
            or "hipErrorOutOfMemory" in summary
            or "exit code" in summary.lower()
        )

    def test_log_analysis_demo_run(self):
        summary = run_log_analysis_for_demo_run(use_vllm_summary=False)
        self.assertIn("tool_only pass", summary)
        self.assertIn("27710372755", summary)
        self.assertIn("81992436725", summary)
        self.assertIn("re-analyzed", summary)

    def test_full_loop(self):
        summary = run_infrastructure_triage_loop(
            DEFAULT_DEMO_PR, str(DEFAULT_DEMO_LOG), use_sample=True, use_vllm_summary=False
        )
        self.assertIn("Pre-merge", summary)
        self.assertIn("Post-CI", summary)


if __name__ == "__main__":
    unittest.main()
