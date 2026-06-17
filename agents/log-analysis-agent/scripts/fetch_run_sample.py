#!/usr/bin/env python3
"""Fetch a GitHub Actions job log and run analyze_log for sample-runs."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

RUN_ID = 27697860238
REPO = "ROCm/TheRock"
DEFAULT_JOB_ID = 81974468002  # kfdtest gfx94X-dcgpu
GITHUB_API = "https://api.github.com"


def download_job_log_text(repo: str, job_id: int) -> str:
    owner, name = repo.split("/", 1)
    resp = requests.get(
        f"{GITHUB_API}/repos/{owner}/{name}/actions/jobs/{job_id}/logs",
        allow_redirects=False,
        timeout=120,
    )
    if resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location")
        if not location:
            raise RuntimeError(f"Job {job_id}: redirect without Location")
        log_resp = requests.get(location, timeout=300)
        log_resp.raise_for_status()
    elif resp.ok:
        log_resp = resp
    else:
        raise RuntimeError(f"Job {job_id} logs: HTTP {resp.status_code}")

    content = log_resp.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            parts = []
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                parts.append(zf.read(name).decode("utf-8", errors="replace"))
            return "\n".join(parts)
    return content.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=RUN_ID)
    parser.add_argument("--job-id", type=int, default=DEFAULT_JOB_ID)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENT_DIR / "sample-runs" / f"run-{RUN_ID}",
    )
    parser.add_argument("--preset", default="therock_multi_arch")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"job-{args.job_id}.log"

    print(f"Downloading job {args.job_id} for run {args.run_id}...")
    log_text = download_job_log_text(args.repo, args.job_id)
    log_path.write_text(log_text, encoding="utf-8")
    print(f"Wrote {log_path} ({len(log_text.splitlines())} lines)")

    from analyze_log import analyze_github_run, build_report, write_outputs

    # Prefer full run metadata when API works without token
    try:
        reports = analyze_github_run(
            args.run_id,
            repo=args.repo,
            job_id=args.job_id,
            output_dir=out,
            preset=args.preset,
        )
        print(f"Analyzed via GitHub run API: {len(reports)} report(s)")
    except Exception as exc:
        print(f"GitHub run analyze fallback ({exc}); analyzing local log...")
        report = build_report(log_path, preset_name=args.preset, use_agent=False)
        report["github_run_id"] = args.run_id
        report["github_job_id"] = args.job_id
        report["repo"] = args.repo
        report["html_url"] = f"https://github.com/{args.repo}/actions/runs/{args.run_id}"
        report["job_name"] = "Linux::release / Test gfx94X-dcgpu / Test kfdtest"
        write_outputs(report, out, write_summary=True, summary_backend="template")

    print(f"Done: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
