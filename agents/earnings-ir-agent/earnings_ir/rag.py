"""Lightweight transcript chunking and keyword retrieval (no embeddings required for demo)."""

from __future__ import annotations

import re

from earnings_ir.models import TranscriptRecord

QA_PATTERN = re.compile(
    r"(?i)(question|q&a|operator|analyst|your line is open|please go ahead)",
)


def chunk_transcript(text: str, max_chars: int = 1500) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text[:max_chars]] if text else []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for para in paragraphs:
        if current and length + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            length = len(para)
        else:
            current.append(para)
            length += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def extract_qa_sections(text: str, max_sections: int = 5) -> list[str]:
    """Heuristic: paragraphs near analyst/question cues."""
    chunks = chunk_transcript(text, max_chars=1200)
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        score = len(QA_PATTERN.findall(chunk))
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [c for _, c in scored[:max_sections]]

    return chunks[:max_sections]


def build_rag_context(
    records: list[TranscriptRecord], query_terms: list[str], max_chars: int = 6000
) -> tuple[str, int]:
    """Select transcript excerpts relevant to simple keyword query."""
    terms = [t.lower() for t in query_terms]
    sections: list[tuple[int, str, TranscriptRecord]] = []

    for record in records:
        for chunk in extract_qa_sections(record.transcript, max_sections=3):
            lower = chunk.lower()
            score = sum(lower.count(t) for t in terms)
            if score > 0:
                header = f"[{record.quarter} {record.earnings_year} - {record.title}]"
                sections.append((score, f"{header}\n{chunk}", record))

    if not sections:
        for record in records[:2]:
            excerpt = record.transcript[:1500]
            header = f"[{record.quarter} {record.earnings_year} - {record.title}]"
            sections.append((1, f"{header}\n{excerpt}", record))

    sections.sort(key=lambda x: x[0], reverse=True)

    out: list[str] = []
    total = 0
    used = 0
    for _, text, _ in sections:
        if total + len(text) > max_chars:
            break
        out.append(text)
        total += len(text)
        used += 1

    return "\n\n---\n\n".join(out), used
