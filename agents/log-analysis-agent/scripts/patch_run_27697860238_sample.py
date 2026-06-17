#!/usr/bin/env python3
"""Patch sample-runs/run-27697860238 report with GitHub run metadata."""

import json
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
OUT = AGENT / "sample-runs" / "run-27697860238"


def main() -> int:
    sys.path.insert(0, str(AGENT))
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    log_rel = OUT / "job-81925995968.log"

    report["log_path"] = str(log_rel).replace("\\", "/")
    report["github_run_id"] = 27697860238
    report["github_job_id"] = 81925995968
    report["repo"] = "ROCm/TheRock"
    report["html_url"] = "https://github.com/ROCm/TheRock/actions/runs/27697860238"
    report["workflow_name"] = "multi_arch_ci.yml"
    report["job_name"] = (
        "Linux::release / Build Multi-Arch Stages / compiler-runtime / Stage - Compiler Runtime"
    )
    report["run_title"] = "Adding kfdtest to The Rock CI."
    report["head_sha"] = "cbc1a15ce2fcf5843637b3ee6dd24121ace4527a"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    from analyze_log import write_html, write_outputs

    write_outputs(report, OUT, write_summary=True, summary_backend="template")
    print(f"Patched {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
