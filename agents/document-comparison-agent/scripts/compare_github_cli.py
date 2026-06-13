#!/usr/bin/env python3
"""CLI to compare policy PDF pairs from GitHub GDPR-similarity-comparison dataset."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.github_regulatory import download_dataset, list_pairs
from app.services.pipeline import run_comparison_from_github


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy vs modernized policy PDFs from GitHub dataset"
    )
    parser.add_argument(
        "--pair-id",
        type=str,
        help="Pair id: europe-brazil or europe-india",
    )
    parser.add_argument("--list-pairs", action="store_true", help="List available pairs")
    parser.add_argument("--download", action="store_true", help="Download PDFs from GitHub first")
    parser.add_argument("-o", "--output", type=Path, help="Write JSON result to file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    if args.download:
        download_dataset(force=False)

    if args.list_pairs:
        for p in list_pairs():
            cached = "cached" if p.legacy_cached and p.modernized_cached else "not cached"
            print(f"{p.pair_id}: {p.legacy_label} vs {p.modernized_label} [{cached}]")
            print(f"  {p.description}")
        return 0

    if not args.pair_id:
        parser.error("Provide --pair-id or use --list-pairs")

    result = await run_comparison_from_github(args.pair_id)
    payload = result.model_dump()

    if args.output:
        args.output.write_text(
            json.dumps(payload, indent=2 if args.pretty else None),
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    else:
        print(result.executive_summary)
        print(f"\nPair: {result.dataset_pair_id}")
        print(f"Legacy: {result.dataset_legacy_label}")
        print(f"Modernized: {result.dataset_modernized_label}")
        print("\nStats:", json.dumps(result.stats, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
