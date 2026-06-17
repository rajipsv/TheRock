#!/usr/bin/env python3
"""Build a committed CI log fixture from a saved GitHub Actions run page (markdown)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MD = (
    Path.home()
    / ".cursor"
    / "projects"
    / "c-Users-Rajeswari-gemini-antigravity-scratch-TheRock"
    / "uploads"
    / "27697860238-0.md"
)

RUN_ID = 27697860238
JOB_ID = 81925995968  # Linux compiler-runtime failure (representative)
JOB_NAME = "Linux::release / Build Multi-Arch Stages / compiler-runtime / Stage - Compiler Runtime"

ANNOTATION_RE = re.compile(
    r"/job/(\d+)#step:(\d+):(\d+)\)\s+(.+?)(?:\s+Show more|\s*\|\s*$)",
    re.IGNORECASE,
)
JOB_LINK_RE = re.compile(
    r"\*\*(.+?)\*\*\s*\]\(/ROCm/TheRock/actions/runs/\d+/job/(\d+)#",
)


def parse_annotations(md_text: str, job_id: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line in md_text.splitlines():
        if f"job/{job_id}#" not in line:
            continue
        m = ANNOTATION_RE.search(line)
        if not m:
            continue
        _jid, _step, line_no, msg = m.groups()
        msg = msg.strip()
        if msg:
            rows.append((int(line_no), msg))
    return rows


def build_log_lines(annotations: list[tuple[int, str]], *, job_name: str, prelude: list[str] | None = None) -> list[str]:
    lines = [
        "2026-03-15T08:10:00.0000000Z ##[group]Runner Image",
        "2026-03-15T08:10:01.0000000Z ##[endgroup]",
        f"2026-03-15T08:10:02.0000000Z ##[group]{job_name}",
        f"2026-03-15T08:10:02.5000000Z Run ID {RUN_ID} — Adding kfdtest to The Rock CI (ROCm/TheRock)",
    ]
    if prelude:
        lines.extend(prelude)
    for line_no, msg in sorted(annotations, key=lambda x: x[0]):
        if "Process completed with exit code" in msg:
            level = "error"
        elif msg.upper().startswith("WARNING"):
            level = "warning"
        else:
            level = "error" if "error" in msg.lower() or "failed" in msg.lower() else "notice"
        if msg.startswith("##["):
            lines.append(f"2026-03-15T08:30:{line_no % 60:02d}.0000000Z {msg}")
        elif level == "error":
            lines.append(f"2026-03-15T08:30:{line_no % 60:02d}.0000000Z ##[error]{msg}")
        elif level == "warning":
            lines.append(f"2026-03-15T08:30:{line_no % 60:02d}.0000000Z ##[warning]{msg}")
        else:
            lines.append(f"2026-03-15T08:30:{line_no % 60:02d}.0000000Z {msg}")
    if not any("exit code" in a[1] for a in annotations):
        lines.append("2026-03-15T08:31:00.0000000Z ##[error]Process completed with exit code 1.")
    lines.append("2026-03-15T08:31:01.0000000Z ##[endgroup]")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--out",
        type=Path,
        default=AGENT_DIR / "tests" / "fixtures" / "run-27697860238-compiler-runtime.log",
    )
    parser.add_argument("--job-id", default=str(JOB_ID))
    parser.add_argument("--job-name", default=JOB_NAME)
    parser.add_argument(
        "--fixture-name",
        default="run-27697860238-compiler-runtime.log",
        help="Output filename under tests/fixtures/",
    )
    args = parser.parse_args()

    if not args.md.is_file():
        raise SystemExit(f"Markdown source not found: {args.md}")

    md_text = args.md.read_text(encoding="utf-8", errors="replace")
    annotations = parse_annotations(md_text, args.job_id)
    if not annotations:
        raise SystemExit(f"No annotations found for job {args.job_id}")

    prelude = []
    if "compiler-runtime" in args.job_name:
        prelude = [
            "2026-03-15T08:10:03.0000000Z Syncing repository ROCm/TheRock",
            "2026-03-15T08:12:00.0000000Z [120/341] Configure sub-project compiler-runtime",
            "2026-03-15T08:25:00.0000000Z [121/341] Building sub-project compiler-runtime",
            "2026-03-15T08:29:00.0000000Z ninja: build stopped: subcommand failed.",
            "2026-03-15T08:29:30.0000000Z ERROR: HIP compiler runtime stage failed",
        ]
    elif "kfdtest" in args.job_name.lower():
        prelude = [
            "2026-03-15T09:05:00.0000000Z Installing test dependencies (pip)",
            "2026-03-15T09:06:00.0000000Z Running test suite: kfdtest (gfx94X-dcgpu)",
            "2026-03-15T09:08:00.0000000Z ctest -R kfdtest --output-on-failure",
        ]

    log_text = "\n".join(build_log_lines(annotations, job_name=args.job_name, prelude=prelude)) + "\n"
    out_path = AGENT_DIR / "tests" / "fixtures" / args.fixture_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(log_text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(log_text.splitlines())} lines, {len(annotations)} annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
