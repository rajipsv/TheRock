# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from log_agent import infer_primary_root_cause, rank_errors_for_root_cause
from llm import configure_vllm_env, default_summary_backend, sanitize_llm_text, use_vllm_summary_enabled
from summarize_log import generate_log_summary, llm_prompt, template_log_summary


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

    def test_default_summary_backend_respects_use_vllm_flag(self):
        import os

        for key in ("USE_VLLM", "USE_VLLM_SUMMARY", "LOG_SUMMARY_BACKEND"):
            os.environ.pop(key, None)
        self.assertEqual(default_summary_backend(), "template")
        os.environ["USE_VLLM"] = "1"
        self.assertEqual(default_summary_backend(), "vllm")
        self.assertTrue(use_vllm_summary_enabled())
        configure_vllm_env(use_vllm=False)
        self.assertEqual(default_summary_backend(), "template")
        configure_vllm_env(use_vllm=True)
        self.assertEqual(default_summary_backend(), "vllm")
        for key in ("USE_VLLM", "USE_VLLM_SUMMARY", "LOG_SUMMARY_BACKEND"):
            os.environ.pop(key, None)

    @patch("summarize_log.invoke_llm_backend", side_effect=RuntimeError("connection refused"))
    def test_llm_failure_falls_back_to_template(self, _mock_invoke):
        text = generate_log_summary(SAMPLE_REPORT, backend="vllm")
        self.assertIn("Log Analysis Executive Summary", text)
        self.assertNotIn("Triage brief (LLM)", text)

    def test_rank_errors_deprioritizes_components_check_banner(self):
        errors = [
            {
                "line_number": 1769,
                "severity": "HIGH",
                "message": "Compoments check 12 Passed, 0 Warning, 6 Fatal Error",
                "kb_pattern_id": "rocm_hip_gpu_oom",
            },
            {
                "line_number": 26551,
                "severity": "CRITICAL",
                "message": "hipErrorOutOfMemory - test_bsric0.cpp failed",
                "kb_pattern_id": "rocm_hip_gpu_oom",
                "type": "HIPERROROUTOFMEMORY",
            },
        ]
        ranked = rank_errors_for_root_cause(errors)
        self.assertIn("hipErrorOutOfMemory", ranked[0]["message"])
        primary = infer_primary_root_cause(errors)
        self.assertIsNotNone(primary)
        self.assertIn("hipErrorOutOfMemory", primary["message"])

    def test_llm_prompt_uses_root_cause_not_setup_banner(self):
        report = {
            "errors": [
                {
                    "line_number": 1769,
                    "severity": "HIGH",
                    "message": "Compoments check 6 Fatal Error",
                },
                {
                    "line_number": 26551,
                    "severity": "CRITICAL",
                    "message": "name 'hipErrorOutOfMemory', description 'out of memory'",
                    "kb_pattern_id": "rocm_hip_gpu_oom",
                },
            ],
            "errors_count": 2,
            "summary": "tool_only pass",
        }
        prompt = llm_prompt(report)
        self.assertIn("primary_root_cause", prompt)
        self.assertIn("hipErrorOutOfMemory", prompt)
        self.assertIn("Do NOT cite Compoments", prompt)
        self.assertNotIn('"errors":', prompt)


if __name__ == "__main__":
    unittest.main()
