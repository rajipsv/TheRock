# Copyright (c) 2026 Rajeswari / AGENTS_030 hackathon prototype
# SPDX-License-Identifier: MIT

"""Pydantic AI tools wrapping change-impact-agent and log-analysis-agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent
CHANGE_IMPACT_DIR = AGENTS_DIR / "change-impact-agent"
LOG_ANALYSIS_DIR = AGENTS_DIR / "log-analysis-agent"
NOTEBOOK_OUT = AGENTS_DIR / "notebook" / "out"

DEMO_PRS = (5572, 5688, 5480, 5718)
DEFAULT_DEMO_PR = 5572

# Primary log-analysis demo: ROCm/TheRock Multi-Arch CI run (kfdtest PR #8864)
DEFAULT_DEMO_RUN_ID = 27697860238
DEFAULT_DEMO_JOB_ID = 81925995968
DEFAULT_DEMO_LOG = (
    LOG_ANALYSIS_DIR / "sample-runs" / f"run-{DEFAULT_DEMO_RUN_ID}" / f"job-{DEFAULT_DEMO_JOB_ID}.log"
)
DEFAULT_DEMO_LOG_URL = (
    f"https://github.com/ROCm/TheRock/actions/runs/{DEFAULT_DEMO_RUN_ID}"
)


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
    for err in (report.get("errors") or [])[:3]:
        if isinstance(err, dict):
            lines.append(
                f"  - L{err.get('line_number')}: {err.get('message', '')[:100]} "
                f"→ {err.get('recommendation', '')[:80]}"
            )
    return "\n".join(lines)


def load_change_impact_sample(pr_number: int) -> dict:
    """Load committed sample-runs report (no GitHub API)."""
    path = CHANGE_IMPACT_DIR / "sample-runs" / f"pr-{pr_number}" / "report.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No sample for PR {pr_number}. Available: {list(DEMO_PRS)}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_change_impact_for_pr(pr_number: int, *, use_sample: bool = True) -> str:
    """
    Pre-merge change impact: blast radius, CI labels, rollout strategy.
    Uses sample-runs when use_sample=True (default for notebook demo).
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
    preset: str = "custom",
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
    lines = [_format_log_summary(report)]
    if report.get("executive_summary"):
        lines.append("")
        lines.append("--- vLLM executive summary ---")
        lines.append(report["executive_summary"][:2000])
    return "\n".join(lines)


def run_infrastructure_triage_loop(
    pr_number: int,
    log_path: str,
    *,
    preset: str = "custom",
    use_sample: bool = True,
    use_vllm_summary: bool = True,
) -> str:
    """
    Multi-agent infrastructure loop:
    1) change-impact-agent (pre-merge briefing)
    2) log-analysis-agent (post-CI failure triage)
    """
    pre = run_change_impact_for_pr(pr_number, use_sample=use_sample)
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
    log_lines = [f"  - {p} ({DEFAULT_DEMO_LOG_URL})" if p == DEFAULT_DEMO_LOG else f"  - {p}" for p in logs if p.is_file()]
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
    def triage_ci_log(log_path: str, preset: str = "custom") -> str:
        """Run log-analysis-agent on a CI/build log (post-failure KB triage)."""
        return run_log_analysis_for_path(log_path, preset=preset)

    @Tool
    def full_infrastructure_triage(pr_number: int, log_path: str) -> str:
        """Run both agents: pre-merge impact briefing then post-CI log triage."""
        default_log = str(DEFAULT_DEMO_LOG)
        return run_infrastructure_triage_loop(
            pr_number,
            log_path or default_log,
            preset="custom",
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
