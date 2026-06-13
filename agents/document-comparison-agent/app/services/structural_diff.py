import difflib
import re

from app.models import ChangeType, ParsedDocument, StructuralChange
from app.services.pdf_parser import has_detectable_headings
from app.services.section_align import AlignmentResult, align_sections


def compare_structural(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    *,
    alignment: AlignmentResult | None = None,
) -> tuple[list[StructuralChange], AlignmentResult]:
    if alignment is None:
        alignment = align_sections(legacy, modernized)

    changes: list[StructuralChange] = []

    for pair in alignment.pairs:
        leg_section = pair.legacy
        mod_section = pair.modern

        if mod_section is None:
            changes.append(
                StructuralChange(
                    change_type=ChangeType.REMOVED,
                    section_id=leg_section.id,
                    section_title=leg_section.title,
                    legacy_excerpt=_excerpt(leg_section.text),
                    line_changes=len(leg_section.text.splitlines()),
                    alignment_method="unmatched",
                )
            )
            continue

        display_title = leg_section.title
        if pair.method == "content" and _normalize_title(leg_section.title) != _normalize_title(
            mod_section.title
        ):
            display_title = f"{leg_section.title} (aligned to: {mod_section.title})"

        leg_text = leg_section.text.strip()
        mod_text = mod_section.text.strip()

        if leg_text == mod_text:
            changes.append(
                StructuralChange(
                    change_type=ChangeType.UNCHANGED,
                    section_id=leg_section.id,
                    section_title=display_title,
                    legacy_excerpt=_excerpt(leg_text),
                    modernized_excerpt=_excerpt(mod_text),
                    alignment_method=pair.method,
                )
            )
            continue

        diff_lines = list(
            difflib.unified_diff(
                leg_text.splitlines(),
                mod_text.splitlines(),
                lineterm="",
            )
        )
        changes.append(
            StructuralChange(
                change_type=ChangeType.MODIFIED,
                section_id=leg_section.id,
                section_title=display_title,
                legacy_excerpt=_excerpt(leg_text),
                modernized_excerpt=_excerpt(mod_text),
                line_changes=max(0, len(diff_lines) - 3),
                alignment_method=pair.method,
            )
        )

    for mod_section in alignment.unmatched_modern:
        changes.append(
            StructuralChange(
                change_type=ChangeType.ADDED,
                section_id=mod_section.id,
                section_title=mod_section.title,
                modernized_excerpt=_excerpt(mod_section.text),
                line_changes=len(mod_section.text.splitlines()),
                alignment_method="unmatched",
            )
        )

    order = {
        ChangeType.REMOVED: 0,
        ChangeType.MODIFIED: 1,
        ChangeType.ADDED: 2,
        ChangeType.UNCHANGED: 3,
    }
    changes.sort(key=lambda c: (order[c.change_type], c.section_title.lower()))
    return changes, alignment


def compute_format_warnings(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    alignment_score: float,
    *,
    low_alignment_threshold: float = 0.4,
) -> list[str]:
    warnings: list[str] = []

    leg_sections = len(legacy.sections)
    mod_sections = len(modernized.sections)
    if leg_sections and mod_sections:
        ratio = max(leg_sections, mod_sections) / min(leg_sections, mod_sections)
        if ratio >= 1.5:
            warnings.append(
                f"Legacy has {leg_sections} sections, modernized has {mod_sections} — "
                "structure may differ significantly."
            )

    if alignment_score < low_alignment_threshold:
        pct = int(alignment_score * 100)
        warnings.append(
            f"Low alignment score ({pct}%) — consider reviewing full document text."
        )

    if legacy.table_count != modernized.table_count:
        if legacy.table_count == 0 and modernized.table_count > 0:
            warnings.append(
                f"Tables detected in modernized document ({modernized.table_count}) but not in legacy."
            )
        elif modernized.table_count == 0 and legacy.table_count > 0:
            warnings.append(
                f"Tables detected in legacy document ({legacy.table_count}) but not in modernized."
            )
        else:
            warnings.append(
                f"Table count differs: legacy={legacy.table_count}, modernized={modernized.table_count}."
            )

    leg_headings = has_detectable_headings(legacy.full_text)
    mod_headings = has_detectable_headings(modernized.full_text)
    if not leg_headings or not mod_headings:
        side = []
        if not leg_headings:
            side.append("legacy")
        if not mod_headings:
            side.append("modernized")
        warnings.append(
            f"No detectable section headings in {' and '.join(side)} document(s)."
        )

    single_blob = (
        len(legacy.sections) == 1
        and legacy.sections[0].title == "Document"
        and len(modernized.sections) == 1
        and modernized.sections[0].title == "Document"
    )
    if single_blob:
        warnings.append("Both documents parsed as a single block with no section structure.")

    return warnings


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _excerpt(text: str, limit: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
