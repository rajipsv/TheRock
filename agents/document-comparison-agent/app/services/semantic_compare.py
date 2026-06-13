import re

from app.llm.client import LLMClient
from app.models import ChangeType, ImpactSeverity, ParsedDocument, SemanticDifference, StructuralChange


REGULATORY_KEYWORDS = {
    "consent": "data subject consent requirements",
    "personal data": "personal data processing scope",
    "retention": "data retention obligations",
    "breach": "breach notification duties",
    "processor": "data processor responsibilities",
    "controller": "data controller obligations",
    "penalty": "enforcement and penalties",
    "fine": "financial penalties",
    "audit": "audit and accountability",
    "cross-border": "cross-border transfer rules",
    "transfer": "international data transfers",
    "automated decision": "automated decision-making",
    "profiling": "profiling restrictions",
    "children": "children's data protections",
    "sensitive": "special category / sensitive data",
    "security": "security safeguards",
    "notification": "regulatory notification timelines",
    "compliance": "compliance program requirements",
}


def _keyword_hits(text: str) -> list[str]:
    lower = text.lower()
    return [desc for term, desc in REGULATORY_KEYWORDS.items() if term in lower]


def _rule_based_semantic(change: StructuralChange) -> SemanticDifference | None:
    if change.change_type == ChangeType.UNCHANGED:
        return None

    legacy = change.legacy_excerpt
    modern = change.modernized_excerpt

    if change.change_type == ChangeType.ADDED:
        hits = _keyword_hits(modern)
        return SemanticDifference(
            section_title=change.section_title,
            summary=f"New section added: {change.section_title}",
            legacy_meaning="Not present in legacy policy.",
            modernized_meaning=modern[:500],
            significance="; ".join(hits) if hits else "Introduces new policy obligations or definitions.",
            severity=ImpactSeverity.MEDIUM if hits else ImpactSeverity.LOW,
        )

    if change.change_type == ChangeType.REMOVED:
        hits = _keyword_hits(legacy)
        return SemanticDifference(
            section_title=change.section_title,
            summary=f"Section removed: {change.section_title}",
            legacy_meaning=legacy[:500],
            modernized_meaning="Removed in modernized policy.",
            significance="; ".join(hits) if hits else "Legacy requirement no longer stated explicitly.",
            severity=ImpactSeverity.MEDIUM if hits else ImpactSeverity.LOW,
        )

    legacy_hits = set(_keyword_hits(legacy))
    modern_hits = set(_keyword_hits(modern))
    new_topics = modern_hits - legacy_hits
    removed_topics = legacy_hits - modern_hits

    significance_parts = []
    if new_topics:
        significance_parts.append("New regulatory topics: " + ", ".join(sorted(new_topics)))
    if removed_topics:
        significance_parts.append("Topics no longer emphasized: " + ", ".join(sorted(removed_topics)))
    if not significance_parts:
        significance_parts.append("Wording or procedural detail changed without obvious topic shift.")

    severity = ImpactSeverity.HIGH if new_topics or removed_topics else ImpactSeverity.MEDIUM
    if change.line_changes > 20:
        severity = ImpactSeverity.HIGH

    return SemanticDifference(
        section_title=change.section_title,
        summary=f"Section modified: {change.section_title} ({change.line_changes} diff lines)",
        legacy_meaning=legacy[:500],
        modernized_meaning=modern[:500],
        significance=" ".join(significance_parts),
        severity=severity,
    )


async def compare_semantic(
    legacy: ParsedDocument,
    modernized: ParsedDocument,
    structural_changes: list[StructuralChange],
    llm: LLMClient,
) -> tuple[list[SemanticDifference], bool]:
    material = [c for c in structural_changes if c.change_type != ChangeType.UNCHANGED]
    if not material:
        return [], False

    llm_payload = [
        {
            "section": c.section_title,
            "change_type": c.change_type.value,
            "legacy_excerpt": c.legacy_excerpt,
            "modernized_excerpt": c.modernized_excerpt,
        }
        for c in material[:12]
    ]

    system = (
        "You are a regulatory policy analyst. Compare legacy vs modernized policy excerpts. "
        "Return JSON array only. Each item: section_title, summary, legacy_meaning, "
        "modernized_meaning, significance, severity (high|medium|low|info)."
    )
    user = (
        f"Legacy document: {legacy.filename}\n"
        f"Modernized document: {modernized.filename}\n"
        f"Changes:\n{llm_payload}"
    )

    raw = await llm.complete(system, user, max_tokens=1800)
    parsed = LLMClient.parse_json_block(raw or "")

    if isinstance(parsed, list) and parsed:
        results: list[SemanticDifference] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                results.append(
                    SemanticDifference(
                        section_title=str(item.get("section_title", "Unknown")),
                        summary=str(item.get("summary", "")),
                        legacy_meaning=str(item.get("legacy_meaning", "")),
                        modernized_meaning=str(item.get("modernized_meaning", "")),
                        significance=str(item.get("significance", "")),
                        severity=ImpactSeverity(str(item.get("severity", "medium")).lower()),
                    )
                )
            except ValueError:
                continue
        if results:
            return results, True

    rule_results = [_rule_based_semantic(c) for c in material]
    return [r for r in rule_results if r is not None], False
