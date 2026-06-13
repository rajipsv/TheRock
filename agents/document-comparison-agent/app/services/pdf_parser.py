import io
import re
from uuid import uuid4

import pdfplumber

from app.models import DocumentSection, ParsedDocument


SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(?:Article|Art\.?|Section|Chapter|Part|Appendix|Schedule)\s+[\dIVXLC]+(?:[.\-][\d]+)?(?:\s*-\s*.+)?"
    r"|(?:\d+\.|\d+\))\s+[A-Z]"
    r"|\d+\.\d+\s+\S"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _serialize_table(table: list[list[str | None]], page_num: int, table_idx: int) -> str:
    rows: list[str] = []
    for row in table:
        cells = [str(cell or "").strip().replace("|", "/") for cell in row]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    return f"[Table page {page_num} #{table_idx + 1}]\n" + "\n".join(rows)


def _split_sections(full_text: str) -> list[tuple[str, str, int | None]]:
    lines = full_text.split("\n")
    sections: list[tuple[str, str, int | None]] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and SECTION_PATTERN.match(stripped):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip(), None))
            current_title = stripped[:120]
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip(), None))

    if not sections:
        sections.append(("Document", full_text, None))

    return sections


def has_detectable_headings(full_text: str) -> bool:
    return bool(SECTION_PATTERN.search(full_text))


def parse_text(
    filename: str,
    text: str,
    page_count: int = 1,
    *,
    table_count: int = 0,
    extraction_mode: str = "text",
) -> ParsedDocument:
    full_text = _normalize(text)
    raw_sections = _split_sections(full_text)

    sections = [
        DocumentSection(
            id=f"sec-{uuid4().hex[:8]}",
            title=title,
            text=body,
            page_start=None,
            page_end=None,
        )
        for title, body, _ in raw_sections
        if body.strip()
    ]

    if not sections and full_text:
        sections = [
            DocumentSection(
                id=f"sec-{uuid4().hex[:8]}",
                title="Document",
                text=full_text,
            )
        ]

    mode: str = extraction_mode if extraction_mode in ("text", "text+tables") else "text"
    return ParsedDocument(
        filename=filename,
        page_count=page_count,
        full_text=full_text,
        sections=sections,
        table_count=table_count,
        extraction_mode=mode,  # type: ignore[arg-type]
    )


def parse_pdf(filename: str, content: bytes, *, extract_tables: bool = True) -> ParsedDocument:
    page_parts: list[str] = []
    table_count = 0

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            parts: list[str] = []
            text = page.extract_text()
            if text:
                parts.append(text)
            if extract_tables:
                for table_idx, table in enumerate(page.extract_tables() or []):
                    serialized = _serialize_table(table, page_num, table_idx)
                    if serialized:
                        table_count += 1
                        parts.append(serialized)
            page_parts.append("\n\n".join(parts))

    extraction_mode = "text+tables" if table_count else "text"
    return parse_text(
        filename,
        "\n\n".join(page_parts),
        page_count=len(page_parts) or 1,
        table_count=table_count,
        extraction_mode=extraction_mode,
    )
