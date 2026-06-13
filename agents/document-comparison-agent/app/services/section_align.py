import difflib
import re
from typing import Literal

from pydantic import BaseModel

from app.models import DocumentSection, ParsedDocument


CONTENT_MATCH_THRESHOLD = 0.55


class SectionPair(BaseModel):
    legacy: DocumentSection
    modern: DocumentSection | None = None
    method: Literal["title", "content", "unmatched"] = "unmatched"
    similarity: float = 0.0


class AlignmentResult(BaseModel):
    pairs: list[SectionPair]
    alignment_score: float
    unmatched_legacy: list[DocumentSection]
    unmatched_modern: list[DocumentSection]


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _normalize_body(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _best_title_match(title: str, candidates: dict[str, DocumentSection]) -> str | None:
    norm = _normalize_title(title)
    if norm in candidates:
        return norm

    for key in candidates:
        if norm in key or key in norm:
            return key
    return None


def _content_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return difflib.SequenceMatcher(None, _normalize_body(a), _normalize_body(b)).ratio()


def align_sections(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    *,
    content_threshold: float = CONTENT_MATCH_THRESHOLD,
) -> AlignmentResult:
    legacy_map = {_normalize_title(s.title): s for s in legacy.sections}
    modern_map = {_normalize_title(s.title): s for s in modernized.sections}

    matched_modern_keys: set[str] = set()
    pairs: list[SectionPair] = []

    for leg_key, leg_section in legacy_map.items():
        mod_key = _best_title_match(leg_key, modern_map)
        if mod_key is not None and mod_key not in matched_modern_keys:
            mod_section = modern_map[mod_key]
            matched_modern_keys.add(mod_key)
            sim = _content_similarity(leg_section.text, mod_section.text)
            pairs.append(
                SectionPair(
                    legacy=leg_section,
                    modern=mod_section,
                    method="title",
                    similarity=sim,
                )
            )
            continue

        best_key: str | None = None
        best_sim = 0.0
        for mod_key, mod_section in modern_map.items():
            if mod_key in matched_modern_keys:
                continue
            sim = _content_similarity(leg_section.text, mod_section.text)
            if sim > best_sim:
                best_sim = sim
                best_key = mod_key

        if best_key is not None and best_sim >= content_threshold:
            mod_section = modern_map[best_key]
            matched_modern_keys.add(best_key)
            pairs.append(
                SectionPair(
                    legacy=leg_section,
                    modern=mod_section,
                    method="content",
                    similarity=best_sim,
                )
            )
        else:
            pairs.append(
                SectionPair(
                    legacy=leg_section,
                    modern=None,
                    method="unmatched",
                    similarity=0.0,
                )
            )

    unmatched_modern = [s for k, s in modern_map.items() if k not in matched_modern_keys]
    unmatched_legacy = [p.legacy for p in pairs if p.modern is None]

    matched_count = sum(1 for p in pairs if p.modern is not None)
    alignment_score = matched_count / len(legacy.sections) if legacy.sections else 0.0

    return AlignmentResult(
        pairs=pairs,
        alignment_score=round(alignment_score, 4),
        unmatched_legacy=unmatched_legacy,
        unmatched_modern=unmatched_modern,
    )
