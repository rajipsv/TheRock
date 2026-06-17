"""Multi-agent earnings IR pipeline (extract -> predict -> draft)."""

from __future__ import annotations

import json

from earnings_ir.dataset import load_transcripts
from earnings_ir.financials import financials_as_text, load_financials
from earnings_ir.llm import LLMClient
from earnings_ir.models import EarningsIRResult, PredictedQuestion, QACheatSheetItem, TranscriptRecord
from earnings_ir.rag import build_rag_context


def _latest_transcript(records: list[TranscriptRecord]) -> TranscriptRecord:
    return records[0]


def _rule_predicted_questions(financials_text: str, rag_context: str) -> list[PredictedQuestion]:
    base = [
        PredictedQuestion(
            question="How should we think about revenue growth versus prior guidance?",
            rationale="Investors typically probe headline growth and guidance credibility.",
            severity="high",
        ),
        PredictedQuestion(
            question="What is driving gross margin movement this quarter?",
            rationale="Margin questions follow any mix, pricing, or cost commentary.",
            severity="high",
        ),
        PredictedQuestion(
            question="How are you positioned against key competitors in your core markets?",
            rationale="Competitive intensity is a common institutional investor theme.",
            severity="medium",
        ),
        PredictedQuestion(
            question="What is the timeline for new product ramps and related capex?",
            rationale="Capex and product cycle questions appear in prior transcript Q&A patterns.",
            severity="medium",
        ),
    ]
    if "data center" in rag_context.lower() or "data center" in financials_text.lower():
        base.append(
            PredictedQuestion(
                question="Can you quantify data center demand and supply constraints?",
                rationale="Segment-specific demand appears in historical Q&A excerpts.",
                severity="high",
            )
        )
    return base


def _rule_script(ticker: str, quarter: str, year: int, financials_text: str) -> str:
    return (
        f"Good afternoon. Thank you for joining {ticker}'s {quarter} {year} earnings call.\n\n"
        f"Today we will review our quarterly results, operational highlights, and outlook. "
        f"Please note that our remarks may include forward-looking statements.\n\n"
        f"Financial highlights (demo data):\n{financials_text}\n\n"
        "We remain focused on executing our product roadmap, disciplined investments, "
        "and delivering long-term shareholder value. Operator, we are ready for questions."
    )


def _rule_bullets(financials_text: str) -> list[str]:
    return [
        "Quarterly revenue and EPS versus prior year",
        "Gross margin drivers and mix shift",
        "Key segment performance highlights",
        "Capital allocation and investment priorities",
        "Updated guidance and macro assumptions",
    ]


def _rule_cheat_sheet(questions: list[PredictedQuestion]) -> list[QACheatSheetItem]:
    return [
        QACheatSheetItem(
            question=q.question,
            suggested_answer="Anchor on verified metrics from the earnings release; avoid speculation.",
            talking_points=["Cite quarter-specific data", "Link to long-term strategy", "Acknowledge uncertainty"],
        )
        for q in questions[:6]
    ]


async def run_earnings_ir_pipeline(
    ticker: str,
    target_quarter: str | None = None,
    target_year: int | None = None,
) -> EarningsIRResult:
    ticker = ticker.upper()
    llm = LLMClient()
    llm_used = False

    records = load_transcripts(ticker)
    latest = _latest_transcript(records)
    quarter = target_quarter or latest.quarter
    year = target_year or latest.earnings_year
    company = latest.company or ticker

    financials_text = financials_as_text(ticker)
    query_terms = ["guidance", "margin", "revenue", "competition", "outlook", "data center", "capex"]
    rag_context, snippets_used = build_rag_context(records, query_terms)

    extracted_summary = (
        f"Company: {company} ({ticker})\n"
        f"Target earnings period: {quarter} {year}\n"
        f"Transcripts loaded: {len(records)}\n"
        f"Latest call: {latest.title} ({latest.call_date})\n\n"
        f"{financials_text}\n\n"
        f"Historical Q&A / transcript excerpts:\n{rag_context[:4000]}"
    )

    # --- Agent 2: Predictive Analyst ---
    analyst_system = (
        "You are an investor relations analyst. Given financials and earnings transcript excerpts, "
        "predict difficult questions institutional investors may ask on the upcoming/simulated call. "
        "Return JSON array: [{question, rationale, severity}] where severity is high|medium|low."
    )
    analyst_user = extracted_summary
    raw_analyst = await llm.complete(analyst_system, analyst_user, max_tokens=1800)
    parsed_q = LLMClient.parse_json_block(raw_analyst or "")

    predicted: list[PredictedQuestion] = []
    if isinstance(parsed_q, list) and parsed_q:
        llm_used = True
        for item in parsed_q:
            if isinstance(item, dict) and item.get("question"):
                predicted.append(
                    PredictedQuestion(
                        question=str(item["question"]),
                        rationale=str(item.get("rationale", "")),
                        severity=str(item.get("severity", "medium")),
                    )
                )

    if not predicted:
        predicted = _rule_predicted_questions(financials_text, rag_context)

    questions_text = json.dumps([q.model_dump() for q in predicted], indent=2)

    # --- Agent 3: Drafting ---
    drafter_system = (
        "You are an IR drafting assistant. Produce JSON with keys:\n"
        "earnings_script (string, CEO/CFO opening remarks ~300-500 words),\n"
        "presentation_bullets (array of 5-8 investor deck bullet strings),\n"
        "qa_cheat_sheet (array of {question, suggested_answer, talking_points[]}).\n"
        "Use only provided facts; mark unknowns as 'to be confirmed from official release'. "
        "Educational demo tone; include safe-harbor reminder."
    )
    drafter_user = (
        f"{extracted_summary}\n\nPredicted investor questions:\n{questions_text}\n\n"
        f"Draft materials for {company} {quarter} {year} earnings."
    )
    raw_draft = await llm.complete(drafter_system, drafter_user, max_tokens=2500)
    parsed_draft = LLMClient.parse_json_block(raw_draft or "")

    script = _rule_script(ticker, quarter, year, financials_text)
    bullets = _rule_bullets(financials_text)
    cheat_sheet = _rule_cheat_sheet(predicted)

    if isinstance(parsed_draft, dict):
        llm_used = True
        if parsed_draft.get("earnings_script"):
            script = str(parsed_draft["earnings_script"])
        if isinstance(parsed_draft.get("presentation_bullets"), list):
            bullets = [str(b) for b in parsed_draft["presentation_bullets"]]
        if isinstance(parsed_draft.get("qa_cheat_sheet"), list):
            cheat_sheet = []
            for item in parsed_draft["qa_cheat_sheet"]:
                if isinstance(item, dict) and item.get("question"):
                    cheat_sheet.append(
                        QACheatSheetItem(
                            question=str(item["question"]),
                            suggested_answer=str(item.get("suggested_answer", "")),
                            talking_points=[str(x) for x in item.get("talking_points", [])],
                        )
                    )
            if not cheat_sheet:
                cheat_sheet = _rule_cheat_sheet(predicted)

    data_source = "Rogersurf/earnings-call-transcripts"
    if len(records) <= 2:
        data_source += " + bundled fallback cache"

    return EarningsIRResult(
        ticker=ticker,
        target_quarter=quarter,
        target_year=year,
        company=company,
        extracted_summary=extracted_summary,
        financials_text=financials_text,
        transcript_snippets_used=snippets_used,
        predicted_questions=predicted,
        earnings_script=script,
        presentation_bullets=bullets,
        qa_cheat_sheet=cheat_sheet,
        llm_used=llm_used,
        data_source=data_source,
    )
