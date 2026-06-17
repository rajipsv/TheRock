#!/usr/bin/env python3
"""CLI entry point for earnings IR demo."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earnings_ir.pipeline import run_earnings_ir_pipeline


async def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Earnings Call IR demo")
    parser.add_argument("--ticker", default="AMD", help="Stock ticker (default: AMD)")
    parser.add_argument("--quarter", default=None, help="Target quarter e.g. Q4")
    parser.add_argument("--year", type=int, default=None, help="Target year e.g. 2023")
    parser.add_argument("-o", "--output", type=Path, help="Write JSON result to file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    result = await run_earnings_ir_pipeline(args.ticker, args.quarter, args.year)
    payload = result.model_dump()

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2 if args.pretty else None), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(f"=== {result.company} ({result.ticker}) {result.target_quarter} {result.target_year} ===")
        print(f"LLM used: {result.llm_used} | Data: {result.data_source}")
        print("\n--- Predicted investor questions ---")
        for i, q in enumerate(result.predicted_questions, 1):
            print(f"{i}. [{q.severity}] {q.question}")
        print("\n--- Earnings script (excerpt) ---")
        print(result.earnings_script[:800], "...")
        print("\n--- Presentation bullets ---")
        for b in result.presentation_bullets:
            print(f"  • {b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
