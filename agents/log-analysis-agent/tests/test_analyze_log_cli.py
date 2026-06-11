# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

import json
import subprocess
import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
FIXTURE = AGENT_DIR / "tests" / "fixtures" / "example.log"


class AnalyzeLogCliTest(unittest.TestCase):
    def test_tool_only_end_to_end(self):
        out_dir = AGENT_DIR / "tests" / "_cli_out"
        if out_dir.exists():
            for f in out_dir.iterdir():
                f.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        proc = subprocess.run(
            [
                sys.executable,
                str(AGENT_DIR / "analyze_log.py"),
                "--log",
                str(FIXTURE),
                "--output-dir",
                str(out_dir),
                "--preset",
                "custom",
            ],
            capture_output=True,
            text=True,
            cwd=str(AGENT_DIR),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

        report_path = out_dir / "report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "tool_only")
        self.assertGreater(len(report.get("errors", [])), 0)
        self.assertTrue((out_dir / "report.html").is_file())
        self.assertTrue((out_dir / "executive_summary.md").is_file())

    def test_unknown_preset_exits_2(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(AGENT_DIR / "analyze_log.py"),
                "--log",
                str(FIXTURE),
                "--preset",
                "not_a_preset",
            ],
            capture_output=True,
            text=True,
            cwd=str(AGENT_DIR),
        )
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
