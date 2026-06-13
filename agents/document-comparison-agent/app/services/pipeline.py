from app.config import settings
from app.llm.client import LLMClient
from app.models import ChangeType, ComparisonResult, ParsedDocument
from app.services.github_regulatory import pair_to_documents
from app.services.impact_analyzer import analyze_regulatory_impact
from app.services.llm_diff import llm_chunk_diff, merge_structural_changes, should_use_llm_fallback
from app.services.pdf_parser import parse_pdf
from app.services.semantic_compare import compare_semantic
from app.services.structural_diff import compare_structural, compute_format_warnings


async def run_comparison_from_documents(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    *,
    source: str | None = "pdf",
    dataset_pair_id: str | None = None,
    dataset_legacy_label: str | None = None,
    dataset_modernized_label: str | None = None,
) -> ComparisonResult:
    llm = LLMClient()

    structural, alignment = compare_structural(legacy, modernized)
    format_warnings = compute_format_warnings(
        legacy,
        modernized,
        alignment.alignment_score,
        low_alignment_threshold=settings.alignment_score_threshold,
    )

    llm_structural_used = False
    if settings.use_llm and settings.use_llm_alignment_fallback and should_use_llm_fallback(
        legacy, modernized, alignment.alignment_score
    ):
        llm_changes, llm_structural_used = await llm_chunk_diff(
            legacy,
            modernized,
            llm,
            alignment_score=alignment.alignment_score,
        )
        structural = merge_structural_changes(structural, llm_changes)
        if llm_structural_used and llm_changes:
            format_warnings.append(
                f"LLM chunk diff supplement applied ({len(llm_changes)} additional item(s))."
            )

    semantic, semantic_llm = await compare_semantic(legacy, modernized, structural, llm)
    impacts, executive, impact_llm = await analyze_regulatory_impact(semantic, llm)

    stats = {
        "legacy_sections": len(legacy.sections),
        "modernized_sections": len(modernized.sections),
        "added": sum(1 for c in structural if c.change_type == ChangeType.ADDED),
        "removed": sum(1 for c in structural if c.change_type == ChangeType.REMOVED),
        "modified": sum(1 for c in structural if c.change_type == ChangeType.MODIFIED),
        "unchanged": sum(1 for c in structural if c.change_type == ChangeType.UNCHANGED),
        "semantic_differences": len(semantic),
        "regulatory_impacts": len(impacts),
        "legacy_tables": legacy.table_count,
        "modernized_tables": modernized.table_count,
    }

    return ComparisonResult(
        legacy=legacy,
        modernized=modernized,
        structural_changes=structural,
        semantic_differences=semantic,
        regulatory_impacts=impacts,
        executive_summary=executive,
        stats=stats,
        llm_used=semantic_llm or impact_llm or llm_structural_used,
        source=source,
        dataset_pair_id=dataset_pair_id,
        dataset_legacy_label=dataset_legacy_label,
        dataset_modernized_label=dataset_modernized_label,
        format_warnings=format_warnings,
        alignment_score=alignment.alignment_score,
    )


async def run_comparison(
    legacy_filename: str,
    legacy_bytes: bytes,
    modernized_filename: str,
    modernized_bytes: bytes,
) -> ComparisonResult:
    legacy = parse_pdf(legacy_filename, legacy_bytes)
    modernized = parse_pdf(modernized_filename, modernized_bytes)
    return await run_comparison_from_documents(legacy, modernized, source="pdf")


async def run_comparison_from_github(pair_id: str) -> ComparisonResult:
    legacy, modernized, pair = pair_to_documents(pair_id, use_pdf=True)
    return await run_comparison_from_documents(
        legacy,
        modernized,
        source="github",
        dataset_pair_id=pair.pair_id,
        dataset_legacy_label=pair.legacy_label,
        dataset_modernized_label=pair.modernized_label,
    )
