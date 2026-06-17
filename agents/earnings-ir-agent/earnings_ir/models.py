from pydantic import BaseModel, Field


class TranscriptRecord(BaseModel):
    ticker: str
    company: str
    quarter: str
    earnings_year: int
    call_date: str
    title: str
    transcript: str
    source_url: str = ""


class FinancialQuarter(BaseModel):
    ticker: str
    quarter: str
    year: int
    revenue_usd_b: float
    gross_margin_pct: float
    eps_usd: float
    yoy_revenue_growth_pct: float
    guidance_summary: str = ""


class PredictedQuestion(BaseModel):
    question: str
    rationale: str
    severity: str = "medium"


class QACheatSheetItem(BaseModel):
    question: str
    suggested_answer: str
    talking_points: list[str] = Field(default_factory=list)


class EarningsIRResult(BaseModel):
    ticker: str
    target_quarter: str
    target_year: int
    company: str
    extracted_summary: str
    financials_text: str
    transcript_snippets_used: int
    predicted_questions: list[PredictedQuestion]
    earnings_script: str
    presentation_bullets: list[str]
    qa_cheat_sheet: list[QACheatSheetItem]
    llm_used: bool
    data_source: str
    disclaimer: str = (
        "Research/education demo only. Not for investor distribution. "
        "Dataset: Rogersurf/earnings-call-transcripts (research-and-educational-use-only)."
    )
