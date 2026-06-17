"""Demo quarterly financials (CSV). Replace with EDGAR ingestion in production."""

from __future__ import annotations

import csv
from pathlib import Path

from earnings_ir.models import FinancialQuarter

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_CSV_PATH = _PACKAGE_ROOT / "data" / "sample_financials.csv"


def load_financials(ticker: str) -> list[FinancialQuarter]:
    ticker = ticker.upper()
    if not _CSV_PATH.is_file():
        return []

    rows: list[FinancialQuarter] = []
    with open(_CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker", "").upper() != ticker:
                continue
            rows.append(
                FinancialQuarter(
                    ticker=ticker,
                    quarter=row["quarter"],
                    year=int(row["year"]),
                    revenue_usd_b=float(row["revenue_usd_b"]),
                    gross_margin_pct=float(row["gross_margin_pct"]),
                    eps_usd=float(row["eps_usd"]),
                    yoy_revenue_growth_pct=float(row["yoy_revenue_growth_pct"]),
                    guidance_summary=row.get("guidance_summary", ""),
                )
            )

    rows.sort(key=lambda r: (r.year, r.quarter), reverse=True)
    return rows


def financials_as_text(ticker: str, quarters: int = 4) -> str:
    data = load_financials(ticker)[:quarters]
    if not data:
        return f"No demo financials on file for {ticker}."

    lines = [f"Demo quarterly financials for {ticker} (synthetic/educational):"]
    for q in data:
        lines.append(
            f"- {q.quarter} {q.year}: revenue ${q.revenue_usd_b}B, "
            f"gross margin {q.gross_margin_pct}%, EPS ${q.eps_usd}, "
            f"YoY revenue growth {q.yoy_revenue_growth_pct}%. "
            f"Guidance: {q.guidance_summary}"
        )
    return "\n".join(lines)
