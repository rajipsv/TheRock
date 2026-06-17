"""Pydantic AI agent wrappers (Academy-style) for the earnings IR notebook."""

from __future__ import annotations

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from earnings_ir.config import settings
from earnings_ir.dataset import load_transcripts
from earnings_ir.financials import financials_as_text
from earnings_ir.pipeline import run_earnings_ir_pipeline


def build_agent_model() -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.llm_model, provider=provider)


@Tool
def get_demo_financials(ticker: str) -> str:
    """Return demo quarterly financial metrics for a ticker (CSV-backed)."""
    return financials_as_text(ticker.upper())


@Tool
def list_earnings_transcripts(ticker: str) -> str:
    """List available cached earnings call transcripts for a ticker."""
    records = load_transcripts(ticker.upper(), limit=settings.max_transcripts)
    lines = [f"Found {len(records)} transcript(s) for {ticker.upper()}:"]
    for r in records:
        lines.append(f"- {r.quarter} {r.earnings_year}: {r.title} ({r.call_date})")
    return "\n".join(lines)


@Tool
def run_full_ir_pipeline(ticker: str) -> str:
    """Run the full extract -> predict -> draft earnings IR pipeline for a ticker."""
    import asyncio

    result = asyncio.run(run_earnings_ir_pipeline(ticker.upper()))
    return (
        f"LLM used: {result.llm_used}\n"
        f"Predicted questions: {len(result.predicted_questions)}\n\n"
        f"Script excerpt:\n{result.earnings_script[:600]}..."
    )


def build_extraction_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model=model,
        tools=[get_demo_financials, list_earnings_transcripts],
        system_prompt=(
            "You are a data extraction agent for investor relations. "
            "Use tools to fetch demo financials and list earnings transcripts."
        ),
    )


def build_orchestrator_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model=model,
        tools=[run_full_ir_pipeline],
        system_prompt=(
            "You orchestrate an earnings IR workflow. "
            "When asked to prepare earnings materials, call run_full_ir_pipeline(ticker)."
        ),
    )
