# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from summarize import (
    FALLBACK_FOOTNOTE,
    REVIEWER_BRIEF_HEADING,
    extract_mentioned_labels,
    generate_llm_summary,
    llm_prompt,
    sanitize_llm_text,
    template_summary,
    validate_llm_summary,
)

SAMPLE_REPORT = {
    "start_sha": "abc1234567890",
    "end_sha": "def9876543210",
    "severity": "MEDIUM",
    "blast_radius_score": 40,
    "changed_files": ["build_tools/github_actions/fetch_test_configurations.py"],
    "changed_components": [],
    "affected_build_stages": [],
    "rollout_strategy": "Canary gfx family + test_filter:quick",
    "rationale": ["CI test matrix configuration changed"],
    "topology_warnings": [],
    "content_insights": {
        "notes": ["GHA wrapper timeout for miopen: 60 → 120 minutes"],
        "test_matrix_changes": {
            "timeout_changes": [
                {"job": "miopen", "old_minutes": 60, "new_minutes": 120}
            ]
        },
    },
    "ci_recommendations": {
        "test_type": "quick",
        "test_type_reason": "CI matrix change",
        "suggested_pr_labels": ["test:miopen", "test_filter:quick"],
        "suggested_test_suites": ["miopen"],
        "notes": "Apply labels manually.",
    },
}


class ExtractLabelsTest(unittest.TestCase):
    def test_parses_test_and_filter_labels(self):
        text = "Apply test:miopen and test_filter:quick for this PR."
        self.assertEqual(
            extract_mentioned_labels(text),
            {"test:miopen", "test_filter:quick"},
        )

    def test_ignores_unrelated_text(self):
        self.assertEqual(extract_mentioned_labels("No labels here."), set())


class ValidateSummaryTest(unittest.TestCase):
    def test_valid_summary_passes(self):
        summary = (
            "Severity MEDIUM (score 40/100). "
            "Suggested labels: test:miopen, test_filter:quick."
        )
        self.assertEqual(validate_llm_summary(summary, SAMPLE_REPORT), [])

    def test_invented_label_fails(self):
        summary = "Please add test:rocblas and test_filter:quick."
        errors = validate_llm_summary(summary, SAMPLE_REPORT)
        self.assertTrue(any("Invented PR labels" in e for e in errors))
        self.assertIn("test:rocblas", errors[0])

    def test_wrong_severity_fails(self):
        summary = "This is a HIGH severity change with test:miopen."
        errors = validate_llm_summary(summary, SAMPLE_REPORT)
        self.assertTrue(any("Severity mismatch" in e for e in errors))

    def test_wrong_score_fails(self):
        summary = "Blast radius score 99/100; use test_filter:quick."
        errors = validate_llm_summary(summary, SAMPLE_REPORT)
        self.assertTrue(any("Blast radius score mismatch" in e for e in errors))


class SanitizeLlmTextTest(unittest.TestCase):
    def test_removes_redacted_thinking(self):
        raw = (
            "<think>internal reasoning</think>\n"
            "- Apply test:miopen."
        )
        self.assertEqual(sanitize_llm_text(raw), "- Apply test:miopen.")

    def test_removes_think_tags(self):
        t = "think"
        raw = "<" + t + ">hidden</" + t + ">Visible text."
        self.assertEqual(sanitize_llm_text(raw), "Visible text.")


class LlmPromptTest(unittest.TestCase):
    def test_includes_content_insights(self):
        prompt = llm_prompt(SAMPLE_REPORT)
        self.assertIn("content_insights", prompt)
        self.assertIn("timeout_changes", prompt)
        self.assertIn("60", prompt)


class GenerateLlmSummaryTest(unittest.TestCase):
    def test_retry_then_template_fallback(self):
        calls: list[str] = []

        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "HIGH severity. Add test:rocblas."
            return "Still wrong: test:rocblas."

        summary, used_fallback = generate_llm_summary(
            SAMPLE_REPORT,
            "openai",
            model="test",
            base_url="http://localhost",
            llm_mode="standalone",
            max_retries=1,
            llm_call=fake_llm,
        )
        self.assertTrue(used_fallback)
        self.assertEqual(len(calls), 2)
        self.assertIn(FALLBACK_FOOTNOTE.strip(), summary)
        self.assertIn("MEDIUM", summary)

    def test_valid_llm_summary_no_fallback(self):
        def fake_llm(prompt: str) -> str:
            return (
                "MEDIUM severity (score 40/100). "
                "Apply test:miopen and test_filter:quick."
            )

        summary, used_fallback = generate_llm_summary(
            SAMPLE_REPORT,
            "openai",
            model="test",
            base_url="http://localhost",
            llm_mode="standalone",
            llm_call=fake_llm,
        )
        self.assertFalse(used_fallback)
        self.assertIn("test:miopen", summary)

    def test_brief_mode_appends_to_template(self):
        def fake_llm(prompt: str) -> str:
            return "Apply test:miopen and test_filter:quick for miopen timeout."

        summary, used_fallback = generate_llm_summary(
            SAMPLE_REPORT,
            "openai",
            model="test",
            base_url="http://localhost",
            llm_mode="brief",
            llm_call=fake_llm,
        )
        self.assertFalse(used_fallback)
        self.assertIn(REVIEWER_BRIEF_HEADING.strip(), summary)
        self.assertIn("## What changed", summary)
        self.assertIn("test:miopen", summary)

    def test_no_validation_skips_checks(self):
        def fake_llm(prompt: str) -> str:
            return "Invented test:rocblas everywhere."

        summary, used_fallback = generate_llm_summary(
            SAMPLE_REPORT,
            "openai",
            model="test",
            base_url="http://localhost",
            validate=False,
            llm_mode="standalone",
            llm_call=fake_llm,
        )
        self.assertFalse(used_fallback)
        self.assertIn("test:rocblas", summary)


class TemplateSummaryTest(unittest.TestCase):
    def test_template_includes_labels(self):
        text = template_summary(SAMPLE_REPORT)
        self.assertIn("test:miopen", text)
        self.assertIn("MEDIUM", text)

    def test_template_shows_topology_section_when_empty(self):
        text = template_summary(SAMPLE_REPORT)
        self.assertIn("## Topology warnings", text)
        self.assertIn("No topology gaps detected", text)


class MainCliTest(unittest.TestCase):
    @patch("summarize.generate_llm_summary")
    def test_template_backend_skips_llm(self, mock_generate):
        from summarize import main

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            out_path = Path(tmp) / "executive_summary.md"
            import json

            report_path.write_text(json.dumps(SAMPLE_REPORT), encoding="utf-8")
            rc = main(
                [
                    "--backend",
                    "template",
                    "--input",
                    str(report_path),
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            mock_generate.assert_not_called()
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
