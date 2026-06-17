"""Load earnings call transcripts from Hugging Face or local cache."""

from __future__ import annotations

import json
from pathlib import Path

from earnings_ir.config import settings
from earnings_ir.models import TranscriptRecord

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK_PATH = _PACKAGE_ROOT / "data" / "fallback_amd_transcripts.json"


def _cache_path(ticker: str) -> Path:
    cache_dir = Path(settings.hf_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _PACKAGE_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker.upper()}_transcripts.json"


def _row_to_record(row: dict) -> TranscriptRecord:
    return TranscriptRecord(
        ticker=str(row.get("ticker", "")).upper(),
        company=str(row.get("company", "")),
        quarter=str(row.get("quarter", "")),
        earnings_year=int(row.get("earnings_year") or row.get("year") or 0),
        call_date=str(row.get("call_date", "")),
        title=str(row.get("title", "")),
        transcript=str(row.get("transcript", "")),
        source_url=str(row.get("source_url", "")),
    )


def _load_fallback(ticker: str) -> list[TranscriptRecord]:
    if not _FALLBACK_PATH.is_file():
        return []
    raw = json.loads(_FALLBACK_PATH.read_text(encoding="utf-8"))
    return [_row_to_record(r) for r in raw if str(r.get("ticker", "")).upper() == ticker.upper()]


def _fetch_from_hf(ticker: str, limit: int) -> list[TranscriptRecord]:
    from datasets import load_dataset

    ticker_upper = ticker.upper()
    records: list[TranscriptRecord] = []

    ds = load_dataset(settings.hf_dataset, split="train", streaming=True)
    for row in ds:
        if str(row.get("ticker", "")).upper() != ticker_upper:
            continue
        records.append(_row_to_record(row))
        if len(records) >= limit:
            break

    records.sort(key=lambda r: (r.earnings_year, r.quarter), reverse=True)
    return records


def load_transcripts(ticker: str | None = None, limit: int | None = None) -> list[TranscriptRecord]:
    """Load transcripts for a ticker (cache -> HF streaming -> bundled fallback)."""
    ticker = (ticker or settings.default_ticker).upper()
    limit = limit or settings.max_transcripts
    cache = _cache_path(ticker)

    if cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        records = [_row_to_record(r) for r in raw[:limit]]
        if records:
            return records

    try:
        records = _fetch_from_hf(ticker, limit=max(limit, settings.max_transcripts))
        if records:
            cache.write_text(
                json.dumps([r.model_dump() for r in records], indent=2),
                encoding="utf-8",
            )
            return records[:limit]
    except Exception:
        pass

    fallback = _load_fallback(ticker)
    if fallback:
        return fallback[:limit]

    raise RuntimeError(
        f"No transcripts for {ticker}. Check network/HF access or add data/fallback_{ticker.lower()}_transcripts.json"
    )


def list_available_tickers_from_cache() -> list[str]:
    cache_dir = Path(settings.hf_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = _PACKAGE_ROOT / cache_dir
    if not cache_dir.is_dir():
        return ["AMD"]
    tickers = []
    for path in cache_dir.glob("*_transcripts.json"):
        tickers.append(path.stem.replace("_transcripts", ""))
    return sorted(tickers) or ["AMD"]
