# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from github_logs import (
    _extract_log_from_bytes,
    download_job_log_text,
    list_failed_runs,
    load_ingested_run_ids,
    mark_run_ingested,
    normalize_github_repo,
    save_ingested_run_ids,
    select_failed_jobs,
)
from github_logs import WorkflowJob
from presets import preset_matches_workflow_name, workflow_name_to_preset


class PresetMappingTest(unittest.TestCase):
    def test_multi_arch_name(self):
        self.assertEqual(workflow_name_to_preset("Multi-Arch CI"), "therock_multi_arch")

    def test_pytorch_wheels(self):
        self.assertEqual(
            workflow_name_to_preset("Test PyTorch Wheels (Full Suite)"),
            "therock_pytorch",
        )

    def test_unit_tests(self):
        self.assertEqual(workflow_name_to_preset("Unit Tests"), "therock_unit_tests")

    def test_install(self):
        self.assertEqual(
            workflow_name_to_preset("Test Native Linux Packages Install"),
            "therock_install",
        )

    def test_preset_matches(self):
        self.assertTrue(
            preset_matches_workflow_name("therock_multi_arch", "Multi-Arch CI")
        )
        self.assertFalse(
            preset_matches_workflow_name("therock_pytorch", "Multi-Arch CI")
        )


class GithubLogsTest(unittest.TestCase):
    def test_normalize_repo_url(self):
        self.assertEqual(
            normalize_github_repo("https://github.com/ROCm/TheRock.git"),
            "ROCm/TheRock",
        )

    def test_extract_log_from_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("1_build.txt", "ERROR build failed\n")
        text = _extract_log_from_bytes(buf.getvalue())
        self.assertIn("ERROR build failed", text)

    def test_select_failed_jobs(self):
        jobs = [
            WorkflowJob(id=1, name="ok", conclusion="success"),
            WorkflowJob(id=2, name="fail", conclusion="failure"),
            WorkflowJob(id=3, name="fail2", conclusion="failure"),
        ]
        selected = select_failed_jobs(jobs, max_jobs=2)
        self.assertEqual([j.id for j in selected], [2, 3])

    def test_ingested_state_roundtrip(self, tmp_path=None):
        state = AGENT_DIR / "tests" / "_state_test.json"
        if state.exists():
            state.unlink()
        mark_run_ingested(state, 12345)
        self.assertIn(12345, load_ingested_run_ids(state))
        save_ingested_run_ids(state, {1, 2})
        self.assertEqual(load_ingested_run_ids(state), {1, 2})
        state.unlink(missing_ok=True)

    @patch("github_logs._get_json")
    def test_list_failed_runs(self, mock_get_json):
        mock_get_json.return_value = {
            "workflow_runs": [
                {
                    "id": 99,
                    "name": "Multi-Arch CI",
                    "head_branch": "main",
                    "head_sha": "abc",
                    "status": "completed",
                    "conclusion": "failure",
                    "html_url": "https://github.com/ROCm/TheRock/actions/runs/99",
                    "workflow_id": 1,
                }
            ]
        }
        runs = list_failed_runs("ROCm/TheRock", per_page=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].id, 99)

    @patch("github_logs.requests.get")
    @patch("github_logs._get")
    def test_download_job_log_redirect(self, mock_get, mock_requests_get):
        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"Location": "https://logs.example/job.zip"}
        redirect.ok = False

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("log.txt", "##[error]Process completed with exit code 1")

        log_resp = MagicMock()
        log_resp.ok = True
        log_resp.content = buf.getvalue()

        mock_get.return_value = redirect
        mock_requests_get.return_value = log_resp

        text = download_job_log_text("ROCm/TheRock", 555)
        self.assertIn("##[error]", text)


if __name__ == "__main__":
    unittest.main()
