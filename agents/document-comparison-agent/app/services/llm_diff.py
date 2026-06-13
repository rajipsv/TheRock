import re
from uuid import uuid4

from app.llm.client import LLMClient
from app.models import ChangeType, ParsedDocument, StructuralChange

MAX_CHUNK_CHARS = 1200
MAX_CHUNKS_PER_DOC = 8


def _chunk_text(full_text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[str, str]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", full_text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) > max_chars:
            chunks.append((f"Chunk {len(chunks) + 1}", "\n\n".join(current)))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append((f"Chunk {len(chunks) + 1}", "\n\n".join(current)))

    return chunks[:MAX_CHUNKS_PER_DOC]


def _needs_llm_fallback(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    alignment_score: float,
) -> bool:
    single_blob = (
        len(legacy.sections) == 1
        and legacy.sections[0].title == "Document"
        and len(modernized.sections) == 1
        and modernized.sections[0].title == "Document"
    )
    return single_blob or alignment_score < 0.4


def should_use_llm_fallback(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    alignment_score: float,
) -> bool:
    return _needs_llm_fallback(legacy, modernized, alignment_score)


async def llm_chunk_diff(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    llm: LLMClient,
    *,
    alignment_score: float,
) -> tuple[list[StructuralChange], bool]:
    if not _needs_llm_fallback(legacy, modernized, alignment_score):
        return [], False

    legacy_chunks = _chunk_text(legacy.full_text)
    modern_chunks = _chunk_text(modernized.full_text)
    if not legacy_chunks and not modern_chunks:
        return [], False

    payload = {
        "legacy_document": legacy.filename,
        "modernized_document": modernized.filename,
        "legacy_chunks": [{"id": t, "text": body[:800]} for t, body in legacy_chunks],
        "modernized_chunks": [{"id": t, "text": body[:800]} for t, body in modern_chunks],
    }

    system = (
        "You are a regulatory policy analyst. Two policy documents may have different formatting "
        "(plain text vs tables, reordered sections). Compare the provided text chunks and return "
        "JSON array only. Each item: section_title (string), change_type (added|removed|modified), "
        "legacy_excerpt (string), modernized_excerpt (string), summary (string). "
        "Focus on substantive policy differences, not formatting noise."
    )

    import json

    raw = await llm.complete(system, json.dumps(payload, indent=2), max_tokens=2000)
    parsed = LLMClient.parse_json_block(raw or "")

    if not isinstance(parsed, list) or not parsed:
        return [], False

    changes: list[StructuralChange] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        change_type_str = str(item.get("change_type", "modified")).lower()
        try:
            change_type = ChangeType(change_type_str)
        except ValueError:
            change_type = ChangeType.MODIFIED

        title = str(item.get("section_title", "LLM-detected difference"))
        legacy_ex = str(item.get("legacy_excerpt", ""))
        modern_ex = str(item.get("modernized_excerpt", ""))

        changes.append(
            StructuralChange(
                change_type=change_type,
                section_id=f"llm-{uuid4().hex[:8]}",
                section_title=title,
                legacy_excerpt=legacy_ex[:320],
                modernized_excerpt=modern_ex[:320],
                line_changes=1,
                alignment_method="llm",
            )
        )

    return changes, True


def merge_structural_changes(
    primary: list[StructuralChange],
    supplemental: list[StructuralChange],
) -> list[StructuralChange]:
    if not supplemental:
        return primary

    seen_titles = {_normalize_key(c.section_title) for c in primary if c.change_type != ChangeType.UNCHANGED}
    merged = list(primary)

    for change in supplemental:
        key = _normalize_key(change.section_title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        merged.append(change)

    order = {
        ChangeType.REMOVED: 0,
        ChangeType.MODIFIED: 1,
        ChangeType.ADDED: 2,
        ChangeType.UNCHANGED: 3,
    }
    merged.sort(key=lambda c: (order[c.change_type], c.section_title.lower()))
    return merged


def _normalize_key(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())
