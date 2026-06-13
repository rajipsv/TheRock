from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.models import ParsedDocument
from app.services.pdf_parser import parse_pdf, parse_text


RAW_BASE = "https://raw.githubusercontent.com/{repo}/{branch}/data"

# MIT-licensed: https://github.com/kornosk/GDPR-similarity-comparison
POLICY_PAIRS: dict[str, dict[str, str]] = {
    "europe-brazil": {
        "legacy_label": "EU GDPR (Europe)",
        "modernized_label": "LGPD (Brazil)",
        "legacy_pdf": "GDPR-EN-Europe.pdf",
        "modernized_pdf": "LGPD-ES-Brazil.pdf",
        "legacy_csv": "GDPR-EN-Europe-converted.csv",
        "modernized_csv": "LGPD-ES-Brazil-converted.csv",
        "description": "Compare EU GDPR (legacy baseline) vs Brazil LGPD (modernized national implementation).",
    },
    "europe-india": {
        "legacy_label": "EU GDPR (Europe)",
        "modernized_label": "India PDPB-style law",
        "legacy_pdf": "GDPR-EN-Europe.pdf",
        "modernized_pdf": "GDPR-EN-Indian.pdf",
        "legacy_csv": "GDPR-EN-Europe-converted.csv",
        "modernized_csv": "GDPR-EN-Indian-converted.csv",
        "description": "Compare EU GDPR vs India GDPR-like privacy legislation.",
    },
}


@dataclass
class PolicyPair:
    pair_id: str
    legacy_label: str
    modernized_label: str
    description: str
    legacy_pdf: str
    modernized_pdf: str


@dataclass
class PairSummary:
    pair_id: str
    legacy_label: str
    modernized_label: str
    description: str
    legacy_pdf: str
    modernized_pdf: str
    legacy_cached: bool
    modernized_cached: bool


def _agent_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_cache_dir() -> Path:
    cache = Path(settings.github_cache_dir)
    if not cache.is_absolute():
        cache = _agent_root() / cache
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _raw_url(filename: str) -> str:
    return RAW_BASE.format(repo=settings.github_repo, branch=settings.github_branch) + f"/{filename}"


def download_file(filename: str, force: bool = False) -> Path:
    cache_dir = resolve_cache_dir()
    dest = cache_dir / filename
    if dest.is_file() and not force:
        return dest

    url = _raw_url(filename)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def download_dataset(force: bool = False) -> list[Path]:
    filenames = {
        meta["legacy_pdf"]
        for meta in POLICY_PAIRS.values()
    } | {
        meta["modernized_pdf"]
        for meta in POLICY_PAIRS.values()
    }
    return [download_file(name, force=force) for name in sorted(filenames)]


def list_pairs() -> list[PairSummary]:
    cache_dir = resolve_cache_dir()
    summaries: list[PairSummary] = []
    for pair_id, meta in POLICY_PAIRS.items():
        legacy_pdf = meta["legacy_pdf"]
        modernized_pdf = meta["modernized_pdf"]
        summaries.append(
            PairSummary(
                pair_id=pair_id,
                legacy_label=meta["legacy_label"],
                modernized_label=meta["modernized_label"],
                description=meta["description"],
                legacy_pdf=legacy_pdf,
                modernized_pdf=modernized_pdf,
                legacy_cached=(cache_dir / legacy_pdf).is_file(),
                modernized_cached=(cache_dir / modernized_pdf).is_file(),
            )
        )
    return summaries


def get_pair(pair_id: str) -> PolicyPair:
    meta = POLICY_PAIRS.get(pair_id)
    if meta is None:
        raise KeyError(f"Unknown pair_id: {pair_id}. Available: {list(POLICY_PAIRS.keys())}")
    return PolicyPair(
        pair_id=pair_id,
        legacy_label=meta["legacy_label"],
        modernized_label=meta["modernized_label"],
        description=meta["description"],
        legacy_pdf=meta["legacy_pdf"],
        modernized_pdf=meta["modernized_pdf"],
    )


def pair_to_documents(pair_id: str, use_pdf: bool = True) -> tuple[ParsedDocument, ParsedDocument, PolicyPair]:
    pair = get_pair(pair_id)
    legacy_path = download_file(pair.legacy_pdf)
    modernized_path = download_file(pair.modernized_pdf)

    if use_pdf:
        legacy = parse_pdf(pair.legacy_pdf, legacy_path.read_bytes())
        modernized = parse_pdf(pair.modernized_pdf, modernized_path.read_bytes())
    else:
        legacy = parse_text(pair.legacy_pdf, legacy_path.read_text(encoding="utf-8", errors="replace"))
        modernized = parse_text(pair.modernized_pdf, modernized_path.read_text(encoding="utf-8", errors="replace"))

    return legacy, modernized, pair


def dataset_status() -> dict[str, Any]:
    cache_dir = resolve_cache_dir()
    pairs = list_pairs()
    cached_count = sum(1 for p in pairs if p.legacy_cached and p.modernized_cached)
    return {
        "loaded": cached_count == len(pairs),
        "source": "github",
        "repo": f"https://github.com/{settings.github_repo}",
        "license": "MIT",
        "cache_dir": str(cache_dir),
        "pair_count": len(pairs),
        "pairs_cached": cached_count,
        "pairs": [p.__dict__ for p in pairs],
    }
