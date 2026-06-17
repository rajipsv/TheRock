#!/usr/bin/env python3
"""Debug GitHub Actions job log download size and zip layout."""

import io
import sys
import zipfile
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env

load_agent_env()

import requests
from github_logs import _extract_log_from_bytes, _get, download_job_log_text

job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 81992436725
owner, name = "ROCm", "TheRock"

resp = _get(f"/repos/{owner}/{name}/actions/jobs/{job_id}/logs", allow_redirects=False)
if resp.status_code in (301, 302, 303, 307, 308):
    log_resp = requests.get(resp.headers["Location"], timeout=120)
else:
    log_resp = resp

content = log_resp.content
out = AGENT_DIR / "out" / f"inspect-{job_id}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
lines = [
    f"job_id={job_id}",
    f"http={log_resp.status_code}",
    f"raw_bytes={len(content)}",
    f"content_type={log_resp.headers.get('content-type')}",
]
try:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        lines.append(f"zip_files={zf.namelist()}")
        for fname in zf.namelist():
            data = zf.read(fname)
            lines.append(f"  {fname}: {len(data)} bytes")
except zipfile.BadZipFile as exc:
    lines.append(f"zip_error={exc}")

text = _extract_log_from_bytes(content)
lines.append(f"extracted_len={len(text)}")
for needle in (
    "hipErrorOutOfMemory",
    "bsric0",
    "exit code 127",
    "rocsparse_create_handle",
    "No GPU suite",
):
    lines.append(f"find {needle}={text.find(needle)}")

full = download_job_log_text(f"{owner}/{name}", job_id)
lines.append(f"download_job_log_text_len={len(full)}")
out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
