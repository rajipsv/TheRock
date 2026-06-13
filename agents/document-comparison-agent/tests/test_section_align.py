import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.models import ChangeType
from app.services.pdf_parser import parse_text
from app.services.section_align import align_sections
from app.services.structural_diff import compare_structural, compute_format_warnings


def _doc(name: str, sections: list[tuple[str, str]]):
    body = "\n\n".join(f"{title}\n{text}" for title, text in sections)
    return parse_text(name, body)


def test_title_alignment_matches_identical_sections():
    legacy = _doc(
        "legacy.pdf",
        [
            ("Section 1. Purpose", "Collect personal data."),
            ("Section 2. Consent", "Opt-in required."),
        ],
    )
    modern = _doc(
        "modern.pdf",
        [
            ("Section 1. Purpose", "Collect personal data."),
            ("Section 2. Consent", "Explicit opt-in required."),
        ],
    )

    alignment = align_sections(legacy, modern)
    assert alignment.alignment_score == 1.0
    changes, _ = compare_structural(legacy, modern, alignment=alignment)
    assert sum(1 for c in changes if c.change_type == ChangeType.MODIFIED) == 1
    assert sum(1 for c in changes if c.change_type == ChangeType.REMOVED) == 0


def test_content_alignment_renamed_headings():
    shared = "Users must provide opt-in consent before marketing communications."
    legacy = _doc("legacy.pdf", [("Section 3. Consent", shared)])
    modern = _doc("modern.pdf", [("Art. 3 - Consent Requirements", shared + " Updated yearly.")])

    alignment = align_sections(legacy, modern)
    assert alignment.pairs[0].method == "content"
    assert alignment.alignment_score == 1.0

    changes, _ = compare_structural(legacy, modern, alignment=alignment)
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].alignment_method == "content"


def test_unmatched_sections_produce_warnings():
    legacy = _doc(
        "legacy.pdf",
        [("Section 1. A", "Alpha"), ("Section 2. B", "Beta"), ("Section 3. C", "Gamma")],
    )
    modern = _doc("modern.pdf", [("Part I", "Completely different content block.")])

    alignment = align_sections(legacy, modern)
    assert alignment.alignment_score < 0.4

    warnings = compute_format_warnings(legacy, modern, alignment.alignment_score)
    assert any("Low alignment score" in w for w in warnings)
    assert any("sections" in w.lower() for w in warnings)


def test_table_metadata_on_parsed_document():
    text = "Section 1. Test\nBody text.\n\n[Table page 1 #1]\n| A | B |\n| 1 | 2 |"
    doc = parse_text("t.pdf", text, table_count=1, extraction_mode="text+tables")
    assert doc.table_count == 1
    assert doc.extraction_mode == "text+tables"
    assert "[Table page 1 #1]" in doc.full_text


def test_format_warning_when_table_counts_differ():
    legacy = parse_text("l.pdf", "Section 1.\nText.", table_count=0)
    modern = parse_text("m.pdf", "Section 1.\nText.", table_count=2, extraction_mode="text+tables")
    warnings = compute_format_warnings(legacy, modern, alignment_score=1.0)
    assert any("Tables detected" in w for w in warnings)
