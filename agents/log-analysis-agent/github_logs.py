# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Fetch GitHub Actions failed runs and job logs for log-analysis-agent."""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from presets import preset_matches_workflow_name, workflow_name_to_preset

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "ROCm/TheRock"
INGESTED_STATE_FILE = ".ingested_run_ids.json"


@dataclass
class WorkflowRun:
    id: int
    name: str
    head_branch: str | None
    head_sha: str | None
    status: str | None
    conclusion: str | None
    html_url: str | None
    workflow_id: int | None
    event: str | None = None
    run_started_at: str | None = None


@dataclass
class WorkflowJob:
    id: int
    name: str
    conclusion: str | None
    started_at: str | None = None
    completed_at: str | None = None


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_github_repo(repo: str) -> str:
    s = repo.strip()
    s = s.replace("https://github.com/", "").replace("http://github.com/", "")
    s = s.rstrip("/").removesuffix(".git")
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


def _repo_parts(repo: str) -> tuple[str, str]:
    normalized = normalize_github_repo(repo)
    owner, name = normalized.split("/", 1)
    if not owner or not name:
        raise ValueError(f"Invalid repo '{repo}' — use owner/repo (e.g. ROCm/TheRock)")
    return owner, name


def _get(path: str, *, allow_redirects: bool = True) -> requests.Response:
    url = f"{GITHUB_API}{path}" if path.startswith("/") else path
    return requests.get(url, headers=_headers(), allow_redirects=allow_redirects, timeout=120)


def _get_json(path: str) -> dict[str, Any]:
    resp = _get(path)
    if not resp.ok:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except Exception:
            pass
        raise RuntimeError(f"GitHub API {path}: HTTP {resp.status_code} {detail}".strip())
    return resp.json()


def list_workflow_ids_for_preset(owner: str, repo: str, preset: str) -> set[int]:
    if preset == "custom":
        return set()
    data = _get_json(f"/repos/{owner}/{repo}/actions/workflows")
    ids: set[int] = set()
    for wf in data.get("workflows", []):
        if preset_matches_workflow_name(preset, wf.get("name", ""), wf.get("path")):
            if wf.get("id") is not None:
                ids.add(int(wf["id"]))
    return ids


def list_failed_runs(
    repo: str,
    *,
    per_page: int = 30,
    preset: str | None = None,
) -> list[WorkflowRun]:
    owner, name = _repo_parts(repo)
    workflow_ids = list_workflow_ids_for_preset(owner, name, preset) if preset else set()
    data = _get_json(
        f"/repos/{owner}/{name}/actions/runs?status=failure&per_page={per_page}"
    )
    runs: list[WorkflowRun] = []
    for r in data.get("workflow_runs", []):
        run = WorkflowRun(
            id=int(r["id"]),
            name=r.get("name") or "unknown",
            head_branch=r.get("head_branch"),
            head_sha=r.get("head_sha"),
            status=r.get("status"),
            conclusion=r.get("conclusion"),
            html_url=r.get("html_url"),
            workflow_id=r.get("workflow_id"),
            event=r.get("event"),
            run_started_at=r.get("run_started_at"),
        )
        if preset and preset != "custom":
            if run.workflow_id and workflow_ids and run.workflow_id in workflow_ids:
                runs.append(run)
            elif preset_matches_workflow_name(preset, run.name):
                runs.append(run)
        else:
            runs.append(run)
    return runs


def get_run(repo: str, run_id: int) -> WorkflowRun:
    owner, name = _repo_parts(repo)
    r = _get_json(f"/repos/{owner}/{name}/actions/runs/{run_id}")
    return WorkflowRun(
        id=int(r["id"]),
        name=r.get("name") or "unknown",
        head_branch=r.get("head_branch"),
        head_sha=r.get("head_sha"),
        status=r.get("status"),
        conclusion=r.get("conclusion"),
        html_url=r.get("html_url"),
        workflow_id=r.get("workflow_id"),
        event=r.get("event"),
        run_started_at=r.get("run_started_at"),
    )


def list_jobs(repo: str, run_id: int) -> list[WorkflowJob]:
    owner, name = _repo_parts(repo)
    data = _get_json(f"/repos/{owner}/{name}/actions/runs/{run_id}/jobs")
    jobs: list[WorkflowJob] = []
    for j in data.get("jobs", []):
        jobs.append(
            WorkflowJob(
                id=int(j["id"]),
                name=j.get("name") or "unknown",
                conclusion=j.get("conclusion"),
                started_at=j.get("started_at"),
                completed_at=j.get("completed_at"),
            )
        )
    return jobs


def select_failed_jobs(jobs: list[WorkflowJob], *, max_jobs: int = 3) -> list[WorkflowJob]:
    failed = [j for j in jobs if j.conclusion in ("failure", "cancelled")]
    if failed:
        return failed[:max_jobs]
    return jobs[:1] if jobs else []


def download_job_log_text(repo: str, job_id: int) -> str:
    owner, name = _repo_parts(repo)
    resp = _get(
        f"/repos/{owner}/{name}/actions/jobs/{job_id}/logs",
        allow_redirects=False,
    )

    if resp.status_code in (301, 302, 303, 307, 308):
        location = resp.headers.get("Location")
        if not location:
            raise RuntimeError(f"Job logs {job_id}: redirect without Location")
        log_resp = requests.get(location, timeout=120)
    elif resp.ok:
        log_resp = resp
    else:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except Exception:
            pass
        raise RuntimeError(
            f"Job logs {job_id}: HTTP {resp.status_code}{f' — {detail}' if detail else ''}"
        )

    if not log_resp.ok:
        raise RuntimeError(f"Job logs {job_id} download: HTTP {log_resp.status_code}")

    content = log_resp.content
    return _extract_log_from_bytes(content)


def _extract_log_from_bytes(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            parts: list[str] = []
            for fname in zf.namelist():
                if fname.endswith("/"):
                    continue
                if fname.endswith(".txt") or "/" not in fname:
                    parts.append(zf.read(fname).decode("utf-8", errors="replace"))
            if parts:
                return "\n".join(parts)
    except zipfile.BadZipFile:
        pass
    return content.decode("utf-8", errors="replace")[:500_000]


def write_job_log(log_text: str, output_dir: Path, job_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"job-{job_id}.log"
    path.write_text(log_text, encoding="utf-8")
    return path


def github_metadata_for_run(run: WorkflowRun, job: WorkflowJob, repo: str) -> dict:
    return {
        "repo": normalize_github_repo(repo),
        "github_run_id": run.id,
        "github_job_id": job.id,
        "workflow_name": run.name,
        "job_name": job.name,
        "html_url": run.html_url,
        "branch": run.head_branch,
        "head_sha": run.head_sha,
        "run_conclusion": run.conclusion,
        "job_conclusion": job.conclusion,
    }


def infer_preset_for_run(run: WorkflowRun, preset_override: str | None = None) -> str:
    if preset_override and preset_override != "auto":
        return preset_override
    return workflow_name_to_preset(run.name)


def load_ingested_run_ids(state_path: Path) -> set[int]:
    if not state_path.is_file():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {int(x) for x in data.get("run_ids", [])}
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


def save_ingested_run_ids(state_path: Path, run_ids: set[int]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"run_ids": sorted(run_ids)}, indent=2),
        encoding="utf-8",
    )


def mark_run_ingested(state_path: Path, run_id: int) -> None:
    known = load_ingested_run_ids(state_path)
    known.add(run_id)
    save_ingested_run_ids(state_path, known)
