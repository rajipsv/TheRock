# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Pydantic AI tools wrapping change-impact-agent and log-analysis-agent."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

AGENTS_DIR = Path(__file__).resolve().parent
CHANGE_IMPACT_DIR = AGENTS_DIR / "change-impact-agent"
LOG_ANALYSIS_DIR = AGENTS_DIR / "log-analysis-agent"
NOTEBOOK_OUT = AGENTS_DIR / "notebook" / "out"

# Bump when Step 5/6 demo API changes (notebook Step 2 checks this).
DEMO_NOTEBOOK_API_REVISION = 2
_REQUIRED_DEMO_API = (
    "run_change_impact_for_demo_pr",
    "run_log_analysis_for_demo_run",
    "ensure_demo_run_samples",
)


def check_demo_notebook_api(module_path: Path | None = None) -> None:
    """Fail fast with git-pull instructions when multi_agent_tools.py is stale."""
    path = (module_path or (AGENTS_DIR / "multi_agent_tools.py")).resolve()
    text = path.read_text(encoding="utf-8")
    missing = [name for name in _REQUIRED_DEMO_API if f"def {name}" not in text]
    if missing:
        raise RuntimeError(
            f"Stale {path} — missing: {', '.join(missing)} "
            f"(need demo API revision {DEMO_NOTEBOOK_API_REVISION}+).\n"
            "Terminal:\n"
            "  cd /workspace/TheRock-old\n"
            "  git fetch origin\n"
            "  git checkout feature/change-impact-agent\n"
            "  git pull origin feature/change-impact-agent\n"
            "  git log -1 --oneline   # c9c176c3a or newer\n"
            "Jupyter: Kernel → Restart, re-run from Step 2."
        )

DEMO_PRS = (5572, 5688, 5480, 5718)
DEFAULT_DEMO_PR = 5572

# Primary log-analysis demo: ROCm/TheRock Multi-Arch CI run (rocSPARSE OOM on Windows gfx110X)
DEFAULT_DEMO_RUN_ID = 27710372755
DEFAULT_DEMO_JOB_ID = 81992436725
DEFAULT_DEMO_LOG = (
    LOG_ANALYSIS_DIR
    / "sample-runs"
    / f"run-{DEFAULT_DEMO_RUN_ID}"
    / f"job-{DEFAULT_DEMO_JOB_ID}"
    / f"job-{DEFAULT_DEMO_JOB_ID}.log"
)
DEFAULT_DEMO_LOG_URL = (
    f"https://github.com/ROCm/TheRock/actions/runs/{DEFAULT_DEMO_RUN_ID}"
)
DEFAULT_DEMO_REPO = "ROCm/TheRock"
DEFAULT_DEMO_RUN_DIR = (
    LOG_ANALYSIS_DIR / "sample-runs" / f"run-{DEFAULT_DEMO_RUN_ID}"
)


def _discover_bundled_job_logs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("job-*/job-*.log"))


def _job_id_from_log_path(log_path: Path) -> int | None:
    match = re.match(r"job-(\d+)", log_path.parent.name)
    return int(match.group(1)) if match else None


def ensure_demo_run_samples(*, allow_git_restore: bool = True) -> Path:
    """
    Ensure bundled GHA run sample logs exist under sample-runs/.

    sample-runs/ only supplies raw .log inputs. Step 6 always re-runs grep + KB analysis
    into agents/notebook/out/ (never reads pre-baked report.json from sample-runs).
    """
    run_dir = DEFAULT_DEMO_RUN_DIR.resolve()
    if _discover_bundled_job_logs(run_dir):
        return run_dir

    if allow_git_restore:
        try:
            root = resolve_therock_root()
            rel = run_dir.relative_to(root)
            proc = subprocess.run(
                ["git", "checkout", "HEAD", "--", str(rel)],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and _discover_bundled_job_logs(run_dir):
                return run_dir
        except (RuntimeError, ValueError, OSError):
            pass

    raise FileNotFoundError(
        f"Demo run sample missing: {run_dir}\n"
        f"Restore bundled logs (not reports — those are regenerated in Step 6):\n"
        f"  git pull\n"
        f"  git checkout HEAD -- agents/log-analysis-agent/sample-runs/run-{DEFAULT_DEMO_RUN_ID}/"
    )


def ensure_demo_log(*, allow_git_restore: bool = True, allow_download: bool = True) -> Path:
    """Return the primary rocSPARSE demo log (restores full run sample if needed)."""
    try:
        ensure_demo_run_samples(allow_git_restore=allow_git_restore)
    except FileNotFoundError:
        pass

    log_path = DEFAULT_DEMO_LOG.resolve()
    if log_path.is_file():
        return log_path

    if allow_git_restore:
        try:
            root = resolve_therock_root()
            rel = log_path.relative_to(root)
            proc = subprocess.run(
                ["git", "checkout", "HEAD", "--", str(rel)],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and log_path.is_file():
                return log_path
        except (RuntimeError, ValueError, OSError):
            pass

    if allow_download:
        _ensure_log_analysis_path()
        from github_logs import download_job_log_text

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_text = download_job_log_text(DEFAULT_DEMO_REPO, DEFAULT_DEMO_JOB_ID)
        log_path.write_text(log_text, encoding="utf-8")
        if log_path.is_file():
            return log_path

    raise FileNotFoundError(
        f"Demo log missing: {log_path}\n"
        f"Restore bundled sample:\n"
        f"  git pull\n"
        f"  git checkout HEAD -- agents/log-analysis-agent/sample-runs/run-{DEFAULT_DEMO_RUN_ID}/\n"
        f"Or set GITHUB_TOKEN in log-analysis-agent/.env and re-run Step 6 (auto-download)."
    )


def _load_bundled_run_meta(run_dir: Path) -> dict | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _github_meta_for_bundled_job(job_id: int, run_meta: dict | None) -> dict | None:
    if not run_meta:
        return {"github_run_id": DEFAULT_DEMO_RUN_ID, "github_job_id": job_id}
    job_info = next(
        (j for j in run_meta.get("job_summaries", []) if j.get("github_job_id") == job_id),
        None,
    )
    meta = {
        "repo": run_meta.get("repo", DEFAULT_DEMO_REPO),
        "github_run_id": run_meta.get("github_run_id", DEFAULT_DEMO_RUN_ID),
        "github_job_id": job_id,
        "html_url": run_meta.get("html_url"),
        "workflow_name": run_meta.get("workflow_name"),
        "branch": run_meta.get("branch"),
        "head_sha": run_meta.get("head_sha"),
        "run_conclusion": run_meta.get("run_conclusion"),
    }
    if job_info:
        meta["job_name"] = job_info.get("job_name")
        meta["job_conclusion"] = job_info.get("job_conclusion")
    return meta


def resolve_therock_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").is_dir() and (candidate / "agents" / "change-impact-agent").is_dir():
            return candidate
        if candidate.name == "TheRock" and (candidate / "agents").is_dir():
            return candidate
    if (AGENTS_DIR.parent / "agents").is_dir():
        return AGENTS_DIR.parent
    raise RuntimeError(
        "Could not find TheRock repo root. Start Jupyter from TheRock/ or agents/notebook/."
    )


def _ensure_change_impact_path() -> None:
    path = str(CHANGE_IMPACT_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _ensure_log_analysis_path() -> None:
    path = str(LOG_ANALYSIS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_env_files() -> None:
    _ensure_change_impact_path()
    _ensure_log_analysis_path()
    from env_loader import load_agent_env as load_ci_env

    load_ci_env()
    # log-analysis env_loader is separate module name — load via path
    log_env = LOG_ANALYSIS_DIR / "env_loader.py"
    if log_env.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location("log_env_loader", log_env)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.load_agent_env()


def _format_change_impact_summary(report: dict) -> str:
    ci = report.get("ci_recommendations") or {}
    labels = ", ".join(ci.get("suggested_pr_labels") or [])
    lines = [
        f"PR #{report.get('pr_number', '?')} change impact",
        f"Severity: {report.get('severity')} (blast radius {report.get('blast_radius_score')}/100)",
        f"Rollout: {report.get('rollout_strategy', 'n/a')}",
        f"Suggested labels: {labels or 'none'}",
        f"Changed files: {report.get('changed_file_count', 0)}",
    ]
    rationale = report.get("rationale") or []
    if rationale:
        lines.append("Rationale:")
        lines.extend(f"  - {note}" for note in rationale[:5])
    return "\n".join(lines)


def _format_log_summary(report: dict) -> str:
    lines = [
        f"Log triage ({report.get('mode', 'tool_only')})",
        f"Preset: {report.get('preset_label', report.get('preset', 'custom'))}",
        f"Errors: {report.get('errors_count', len(report.get('errors', [])))}",
        f"Summary: {report.get('summary', '')[:400]}",
    ]
    primary = report.get("primary_root_cause")
    if isinstance(primary, dict) and primary.get("message"):
        lines.append(
            f"Primary root cause: L{primary.get('line_number')}: "
            f"{primary.get('message', '')[:120]}"
        )
    for err in (report.get("errors") or [])[:3]:
        if isinstance(err, dict):
            lines.append(
                f"  - L{err.get('line_number')}: {err.get('message', '')[:100]} "
                f"→ {err.get('recommendation', '')[:80]}"
            )
    return "\n".join(lines)


def _format_vllm_section(executive_summary: str) -> str:
    text = executive_summary or ""
    marker = "## Triage brief (LLM)"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def load_change_impact_sample(pr_number: int) -> dict:
    """Load committed sample-runs report (no GitHub API)."""
    path = CHANGE_IMPACT_DIR / "sample-runs" / f"pr-{pr_number}" / "report.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No sample for PR {pr_number}. Available: {list(DEMO_PRS)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _git_ref_resolves(ref: str, repo_root: Path) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _build_change_impact_report_live(pr_number: int) -> tuple[dict, str]:
    """
    Re-run change-impact analyze.py pipeline when possible.

    Order: GitHub fetch + merge-base (needs token) → local git refs from sample metadata
    → fall back to committed sample-runs report.json.
    """
    sample = load_change_impact_sample(pr_number)
    _ensure_change_impact_path()
    from analyze import build_report
    from github_pr import (
        _token,
        ensure_pr_fetched,
        ensure_upstream_ref_fetched,
        get_pull_request,
        git_merge_base,
        pr_local_ref,
    )
    from manifest_bridge import find_therock_root, resolve_git_ref

    repo_root = find_therock_root()

    if _token():
        try:
            ensure_pr_fetched(pr_number, repo_root)
            end_ref = pr_local_ref(pr_number)
            try:
                pr_info = get_pull_request(pr_number, DEFAULT_DEMO_REPO)
                base_ref = pr_info.base_ref
            except Exception:
                base_ref = "main"
            base = ensure_upstream_ref_fetched(base_ref, repo_root)
            start_ref = git_merge_base(
                resolve_git_ref(base, repo_root),
                resolve_git_ref(end_ref, repo_root),
                repo_root,
            )
            report = build_report(start_ref, end_ref, repo_root, full_manifest=True)
            report["pr_number"] = pr_number
            report["upstream_repo"] = DEFAULT_DEMO_REPO
            return report, "live_fetch"
        except Exception:
            pass

    start_ref = sample.get("start_ref") or sample.get("start_sha")
    end_ref = sample.get("end_ref") or sample.get("end_sha")
    if (
        start_ref
        and end_ref
        and _git_ref_resolves(str(start_ref), repo_root)
        and _git_ref_resolves(str(end_ref), repo_root)
    ):
        full_manifest = (
            sample.get("manifest_mode") == "full_github_manifest_diff" and bool(_token())
        )
        report = build_report(
            str(start_ref),
            str(end_ref),
            repo_root,
            full_manifest=full_manifest,
        )
        report["pr_number"] = pr_number
        report["upstream_repo"] = DEFAULT_DEMO_REPO
        mode = "live_git_refs+api" if full_manifest else "live_git_refs"
        return report, mode

    return sample, "sample_fallback"


def _write_change_impact_outputs(
    report: dict,
    out_dir: Path,
    *,
    use_vllm_summary: bool,
) -> None:
    from analyze import write_html

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, out_dir)

    _ensure_change_impact_path()
    from summarize import generate_llm_summary, template_summary

    if not use_vllm_summary:
        summary = template_summary(report)
        report["executive_summary"] = summary
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out_dir / "executive_summary.md").write_text(summary, encoding="utf-8")
        return

    _ensure_log_analysis_path()
    from llm import configure_vllm_env, llm_env_config, use_vllm_summary_enabled

    configure_vllm_env(use_vllm=True)
    if not use_vllm_summary_enabled():
        summary = template_summary(report)
    else:
        cfg = llm_env_config()
        summary, _used_fallback = generate_llm_summary(
            report,
            "vllm",
            model=cfg["model"],
            base_url=cfg["base_url"],
            validate=True,
            fallback_template=True,
            llm_mode="brief",
        )
    report["executive_summary"] = summary
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "executive_summary.md").write_text(summary, encoding="utf-8")


def run_change_impact_for_demo_pr(
    pr_number: int = DEFAULT_DEMO_PR,
    *,
    use_vllm_summary: bool = True,
) -> str:
    """
    Step 5 demo: re-run manifest/path/topology analysis when git/API allows.

    Unlike use_sample=True (load-only), this calls analyze.build_report() live and writes
    fresh output under agents/notebook/out/pr-<N>/, then optional vLLM executive brief.
    """
    _load_env_files()
    out_dir = NOTEBOOK_OUT / f"pr-{pr_number}"
    report, analysis_mode = _build_change_impact_report_live(pr_number)
    _write_change_impact_outputs(report, out_dir, use_vllm_summary=use_vllm_summary)

    lines = [
        "=== change-impact-agent: pre-merge analysis ===",
        f"PR #{pr_number} — mode: {analysis_mode}",
        f"Pipeline: manifest_diff → path_diff → content_diff → impact_graph → ci_mapping",
        f"Output: {out_dir.relative_to(AGENTS_DIR)}/report.json",
        "",
        _format_change_impact_summary(report),
    ]
    if analysis_mode == "sample_fallback":
        lines.extend(
            [
                "",
                "Note: could not resolve git refs or fetch PR — showing committed sample-runs JSON.",
                "Set GITHUB_TOKEN in change-impact-agent/.env and ensure pr-* ref exists to re-analyze live.",
            ]
        )
    if use_vllm_summary and report.get("executive_summary"):
        llm_part = _format_vllm_section(report["executive_summary"])
        if llm_part:
            lines.extend(["", "=== vLLM reviewer brief (after analysis JSON) ===", llm_part[:2000]])
    return "\n".join(lines)


def run_change_impact_for_pr(pr_number: int, *, use_sample: bool = True) -> str:
    """
    Pre-merge change impact: blast radius, CI labels, rollout strategy.

    use_sample=True loads committed sample-runs only (no re-analysis).
    For the MI300 notebook demo, prefer run_change_impact_for_demo_pr().
    """
    _load_env_files()
    NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)
    out_dir = NOTEBOOK_OUT / f"pr-{pr_number}"

    if use_sample:
        report = load_change_impact_sample(pr_number)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return _format_change_impact_summary(report)

    _ensure_change_impact_path()
    from analyze import build_report, write_html
    from github_pr import _token, ensure_pr_fetched, get_pull_request, pr_local_ref
    from manifest_bridge import find_therock_root

    if not _token():
        return (
            f"GITHUB_TOKEN missing — cannot fetch PR {pr_number}. "
            f"Use use_sample=True or set agents/change-impact-agent/.env"
        )

    repo_root = find_therock_root()
    ensure_pr_fetched(pr_number, repo_root)
    end_ref = pr_local_ref(pr_number)
    try:
        pr_info = get_pull_request(pr_number, "ROCm/TheRock")
        base_ref = pr_info.base_ref
    except Exception:
        base_ref = "main"

    from github_pr import ensure_upstream_ref_fetched, git_merge_base
    from manifest_bridge import resolve_git_ref

    base = ensure_upstream_ref_fetched(base_ref, repo_root)
    start_ref = git_merge_base(resolve_git_ref(base, repo_root), resolve_git_ref(end_ref, repo_root), repo_root)

    report = build_report(start_ref, end_ref, repo_root, full_manifest=False)
    report["pr_number"] = pr_number
    report["upstream_repo"] = "ROCm/TheRock"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, out_dir)
    return _format_change_impact_summary(report)


def run_log_analysis_for_path(
    log_path: str,
    preset: str = "therock_multi_arch",
    *,
    use_vllm_summary: bool = True,
) -> str:
    """Post-CI log triage: errors, KB matches, optional vLLM executive summary."""
    _load_env_files()
    path = Path(log_path).expanduser().resolve()
    if not path.is_file():
        alt = LOG_ANALYSIS_DIR / log_path
        if alt.is_file():
            path = alt.resolve()
        else:
            return f"Log file not found: {log_path}"

    _ensure_log_analysis_path()
    from analyze_log import build_report, write_outputs
    from llm import configure_vllm_env, default_summary_backend

    if use_vllm_summary:
        configure_vllm_env(use_vllm=True)
    else:
        configure_vllm_env(use_vllm=False)

    out_dir = NOTEBOOK_OUT / f"log-{path.stem}"
    report = build_report(path, preset_name=preset, use_agent=False)
    backend = default_summary_backend() if use_vllm_summary else "template"
    write_outputs(report, out_dir, write_summary=True, summary_backend=backend)
    lines = [
        "=== tool_only analysis (grep + KB) ===",
        _format_log_summary(report),
        f"Artifacts: {out_dir.relative_to(AGENTS_DIR)}/report.json",
    ]
    if use_vllm_summary and report.get("executive_summary"):
        llm_part = _format_vllm_section(report["executive_summary"])
        if llm_part:
            lines.extend(["", "=== vLLM executive brief (after tool_only JSON) ===", llm_part[:2000]])
    return "\n".join(lines)


def run_log_analysis_for_demo_run(
    *,
    preset: str = "therock_multi_arch",
    use_vllm_summary: bool = True,
) -> str:
    """
    Step 6 demo: re-analyze all bundled failed jobs for run 27710372755.

    Each job gets a fresh tool_only pass (grep + KB → report.json). When USE_VLLM is on,
    every failed job also gets an executive_summary.md via vLLM (not just rocSPARSE).
    """
    _load_env_files()
    run_dir = ensure_demo_run_samples()
    log_paths = _discover_bundled_job_logs(run_dir)
    if not log_paths:
        raise FileNotFoundError(f"No job logs under {run_dir}")

    _ensure_log_analysis_path()
    from analyze_log import build_report, write_outputs, write_run_rollup
    from llm import configure_vllm_env, default_summary_backend

    if use_vllm_summary:
        configure_vllm_env(use_vllm=True)
    else:
        configure_vllm_env(use_vllm=False)

    summary_backend = default_summary_backend() if use_vllm_summary else "template"
    run_meta = _load_bundled_run_meta(run_dir)
    out_dir = NOTEBOOK_OUT / f"run-{DEFAULT_DEMO_RUN_ID}"
    reports: list[dict] = []

    for log_path in log_paths:
        job_id = _job_id_from_log_path(log_path)
        job_out = out_dir / f"job-{job_id}" if job_id else out_dir / log_path.parent.name
        report = build_report(
            log_path,
            preset_name=preset,
            use_agent=False,
            github_meta=_github_meta_for_bundled_job(job_id, run_meta) if job_id else None,
        )
        write_outputs(
            report,
            job_out,
            write_summary=True,
            summary_backend=summary_backend,
        )
        reports.append(report)

    if len(reports) > 1 and run_meta:
        run_obj = SimpleNamespace(
            id=run_meta.get("github_run_id", DEFAULT_DEMO_RUN_ID),
            html_url=run_meta.get("html_url"),
            name=run_meta.get("workflow_name"),
            head_branch=run_meta.get("branch"),
            head_sha=run_meta.get("head_sha"),
            conclusion=run_meta.get("run_conclusion"),
        )
        all_jobs = [
            SimpleNamespace(conclusion=j.get("job_conclusion", "failure"))
            for j in run_meta.get("job_summaries", [])
        ]
        write_run_rollup(
            run_meta.get("github_run_id", DEFAULT_DEMO_RUN_ID),
            run_meta.get("repo", DEFAULT_DEMO_REPO),
            run_obj,
            all_jobs,
            reports,
            out_dir,
        )

    lines = [
        "=== log-analysis-agent: tool_only pass (grep + KB) ===",
        f"Run {DEFAULT_DEMO_RUN_ID} — re-analyzed {len(reports)} bundled job log(s)",
        f"Input logs: {run_dir.relative_to(AGENTS_DIR)}/*.log",
        f"Fresh output: {out_dir.relative_to(AGENTS_DIR)}/",
        "",
        "Rollup:",
    ]
    for report in reports:
        job_id = report.get("github_job_id", "?")
        job_name = (report.get("job_name") or "")[:60]
        primary = report.get("primary_root_cause") or {}
        primary_msg = (primary.get("message") or "n/a")[:90]
        focus = " [demo focus — rocSPARSE OOM]" if job_id == DEFAULT_DEMO_JOB_ID else ""
        lines.append(
            f"  job {job_id}{focus} ({job_name}): "
            f"{report.get('errors_count', 0)} errors | primary: {primary_msg}"
        )

    lines.extend(["", "=== Per-job detail ==="])
    for report in reports:
        job_id = report.get("github_job_id", "?")
        lines.append(f"\n--- job {job_id} ---")
        lines.append(_format_log_summary(report))
        lines.append(
            f"Artifacts: {out_dir.relative_to(AGENTS_DIR)}/job-{job_id}/report.json"
        )
        if use_vllm_summary:
            llm_part = _format_vllm_section(report.get("executive_summary") or "")
            if llm_part:
                lines.extend(["vLLM brief:", llm_part[:800]])
            else:
                lines.append("(no vLLM brief — check USE_VLLM / vLLM server)")

    if use_vllm_summary and not any(
        _format_vllm_section(r.get("executive_summary") or "") for r in reports
    ):
        lines.append("")
        lines.append("(vLLM briefs empty — set USE_VLLM=True in Step 2 or start vLLM in Step 1)")

    return "\n".join(lines)


def run_infrastructure_triage_loop(
    pr_number: int,
    log_path: str,
    *,
    preset: str = "therock_multi_arch",
    use_sample: bool = True,
    use_vllm_summary: bool = True,
) -> str:
    """
    Multi-agent infrastructure loop:
    1) change-impact-agent (pre-merge briefing)
    2) log-analysis-agent (post-CI failure triage)
    """
    pre = run_change_impact_for_demo_pr(pr_number, use_vllm_summary=use_vllm_summary)
    if Path(log_path).resolve() == DEFAULT_DEMO_LOG.resolve():
        post = run_log_analysis_for_demo_run(
            preset=preset,
            use_vllm_summary=use_vllm_summary,
        )
    else:
        post = run_log_analysis_for_path(log_path, preset=preset, use_vllm_summary=use_vllm_summary)
    return (
        "=== Pre-merge (change-impact-agent) ===\n"
        f"{pre}\n\n"
        "=== Post-CI failure (log-analysis-agent) ===\n"
        f"{post}\n\n"
        "Artifacts written under agents/notebook/out/"
    )


def list_demo_assets() -> str:
    """List bundled demo PRs and log fixtures for the multi-agent notebook."""
    pr_lines = [f"  - PR #{pr} (sample-runs/pr-{pr}/)" for pr in DEMO_PRS]
    logs = [
        DEFAULT_DEMO_LOG,
        LOG_ANALYSIS_DIR / "tests" / "fixtures" / "run-27697860238-compiler-runtime.log",
        LOG_ANALYSIS_DIR / "tests" / "fixtures" / "run-27697860238-kfdtest.log",
    ]
    log_lines = []
    for p in logs:
        if p == DEFAULT_DEMO_LOG and not p.is_file():
            try:
                p = ensure_demo_log(allow_download=False)
            except FileNotFoundError:
                pass
        if p.is_file():
            label = f" ({DEFAULT_DEMO_LOG_URL})" if p.resolve() == DEFAULT_DEMO_LOG.resolve() else ""
            log_lines.append(f"  - {p}{label}")
    return (
        "Demo PRs (change-impact):\n"
        + "\n".join(pr_lines)
        + "\n\nDemo logs (log-analysis):\n"
        + "\n".join(log_lines)
    )


# --- Pydantic AI wrappers (import when pydantic_ai is installed) ---


def build_agent_model(base_url: str, api_key: str, model: str):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model, provider=provider)


def build_pydantic_tools():
    from pydantic_ai import Tool

    @Tool
    def analyze_upstream_pr(pr_number: int) -> str:
        """Run change-impact-agent on a TheRock PR (pre-merge blast radius + rollout)."""
        return run_change_impact_for_pr(pr_number, use_sample=True)

    @Tool
    def triage_ci_log(log_path: str, preset: str = "therock_multi_arch") -> str:
        """Run log-analysis-agent on a CI/build log (post-failure KB triage)."""
        return run_log_analysis_for_path(log_path, preset=preset)

    @Tool
    def full_infrastructure_triage(pr_number: int, log_path: str) -> str:
        """Run both agents: pre-merge impact briefing then post-CI log triage."""
        default_log = str(DEFAULT_DEMO_LOG)
        return run_infrastructure_triage_loop(
            pr_number,
            log_path or default_log,
            preset="therock_multi_arch",
            use_sample=True,
        )

    @Tool
    def list_demo_data() -> str:
        """List available demo PR numbers and log paths for this notebook."""
        return list_demo_assets()

    return [analyze_upstream_pr, triage_ci_log, full_infrastructure_triage, list_demo_data]


def build_orchestrator_agent(model) -> "Agent":
    from pydantic_ai import Agent

    tools = build_pydantic_tools()
    return Agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are the TheRock Infrastructure Triage Orchestrator (AGENTS_030 multi-agent).\n"
            "You coordinate two specialist agents:\n"
            "1) change-impact-agent — pre-merge PR blast radius, CI labels, rollout strategy\n"
            "2) log-analysis-agent — post-CI log triage with knowledge-base fixes\n\n"
            "Workflow rules:\n"
            "- For 'before merge' or 'what CI to run' → analyze_upstream_pr\n"
            "- For 'CI failed' or log errors → triage_ci_log\n"
            "- For full infrastructure change loop → full_infrastructure_triage\n"
            "- When unsure what demo data exists → list_demo_data\n"
            "Always call tools; do not invent severity scores or error lines."
        ),
    )
