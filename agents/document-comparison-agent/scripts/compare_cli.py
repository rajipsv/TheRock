#!/usr/bin/env python3
"""CLI helper to compare two policy PDFs without a UI."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.pipeline import run_comparison


async def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy vs modernized policy PDFs")
    parser.add_argument("legacy_pdf", type=Path, help="Path to legacy policy PDF")
    parser.add_argument("modernized_pdf", type=Path, help="Path to modernized policy PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write full JSON result to file (default: stdout summary only)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    if not args.legacy_pdf.is_file():
        print(f"Error: not found: {args.legacy_pdf}", file=sys.stderr)
        return 1
    if not args.modernized_pdf.is_file():
        print(f"Error: not found: {args.modernized_pdf}", file=sys.stderr)
        return 1

    result = await run_comparison(
        args.legacy_pdf.name,
        args.legacy_pdf.read_bytes(),
        args.modernized_pdf.name,
        args.modernized_pdf.read_bytes(),
    )

    payload = result.model_dump()
    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2 if args.pretty else None),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    else:
        print(result.executive_summary)
        print("\nStats:", json.dumps(result.stats, indent=2))
        print(f"\nAlignment score: {result.alignment_score}")
        if result.format_warnings:
            print("\nFormat warnings:")
            for warning in result.format_warnings:
                print(f"  - {warning}")
        print(f"\nLLM used: {result.llm_used}")
        for impact in result.regulatory_impacts:
            print(f"\n[{impact.severity.value.upper()}] {impact.title}")
            print(f"  {impact.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
