# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from llm import sanitize_llm_text
from summarize_log import generate_log_summary, template_log_summary


SAMPLE_REPORT = {
    "log_path": "/tmp/job.log",
    "mode": "tool_only",
    "preset": "custom",
    "summary": "Build failed at link stage.",
    "errors": [
        {
            "line_number": 42,
            "severity": "HIGH",
            "message": "undefined reference to hipMalloc",
            "recommendation": "Check ROCm install paths.",
        }
    ],
    "errors_count": 1,
}


class SummarizeLogTest(unittest.TestCase):
    def test_template_summary_includes_errors(self):
        text = template_log_summary(SAMPLE_REPORT)
        self.assertIn("hipMalloc", text)
        self.assertIn("tool_only", text)

    def test_sanitize_strips_thinking_blocks(self):
        open_tag, close_tag = "<think>", "</think>"
        wrapped = f"{open_tag}internal chain of thought{close_tag}\n- Root cause: linker error"
        self.assertEqual(sanitize_llm_text(wrapped), "- Root cause: linker error")

    @patch("summarize_log.invoke_llm_backend")
    def test_vllm_brief_mode_appends_llm_section(self, mock_invoke):
        mock_invoke.return_value = "- Verify ROCm paths\n- Re-run with verbose link"
        text = generate_log_summary(SAMPLE_REPORT, backend="vllm", llm_mode="brief")
        self.assertIn("Log Analysis Executive Summary", text)
        self.assertIn("Triage brief (LLM)", text)
        self.assertIn("Verify ROCm paths", text)
        mock_invoke.assert_called_once()
        self.assertEqual(mock_invoke.call_args.args[0], "vllm")

    @patch("summarize_log.invoke_llm_backend", side_effect=RuntimeError("connection refused"))
    def test_llm_failure_falls_back_to_template(self, _mock_invoke):
        text = generate_log_summary(SAMPLE_REPORT, backend="vllm")
        self.assertIn("Log Analysis Executive Summary", text)
        self.assertNotIn("Triage brief (LLM)", text)


if __name__ == "__main__":
    unittest.main()
