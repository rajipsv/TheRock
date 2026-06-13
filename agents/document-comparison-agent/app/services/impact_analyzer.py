from app.llm.client import LLMClient
from app.models import ImpactSeverity, RegulatoryImpact, SemanticDifference


async def analyze_regulatory_impact(
    semantic_differences: list[SemanticDifference],
    llm: LLMClient,
) -> tuple[list[RegulatoryImpact], str, bool]:
    if not semantic_differences:
        summary = "No material differences detected between legacy and modernized policies."
        return [], summary, False

    system = (
        "You are a compliance officer. Given policy semantic differences, produce JSON with:\n"
        "executive_summary (string),\n"
        "impacts (array of {title, severity, summary, affected_areas, recommended_actions, citations}).\n"
        "severity must be high|medium|low|info. citations should reference section titles."
    )
    user = {
        "differences": [d.model_dump() for d in semantic_differences[:15]],
    }

    import json

    raw = await llm.complete(system, json.dumps(user, indent=2), max_tokens=1600)
    parsed = LLMClient.parse_json_block(raw or "")

    if isinstance(parsed, dict) and parsed.get("impacts"):
        impacts: list[RegulatoryImpact] = []
        for item in parsed["impacts"]:
            if not isinstance(item, dict):
                continue
            try:
                impacts.append(
                    RegulatoryImpact(
                        title=str(item.get("title", "Regulatory impact")),
                        severity=ImpactSeverity(str(item.get("severity", "medium")).lower()),
                        summary=str(item.get("summary", "")),
                        affected_areas=[str(x) for x in item.get("affected_areas", [])],
                        recommended_actions=[str(x) for x in item.get("recommended_actions", [])],
                        citations=[str(x) for x in item.get("citations", [])],
                    )
                )
            except ValueError:
                continue
        executive = str(parsed.get("executive_summary", _fallback_summary(semantic_differences)))
        if impacts:
            return impacts, executive, True

    return _rule_based_impacts(semantic_differences)


def _rule_based_impacts(
    semantic_differences: list[SemanticDifference],
) -> tuple[list[RegulatoryImpact], str, bool]:
    high = [d for d in semantic_differences if d.severity == ImpactSeverity.HIGH]
    medium = [d for d in semantic_differences if d.severity == ImpactSeverity.MEDIUM]

    impacts: list[RegulatoryImpact] = []

    if high:
        impacts.append(
            RegulatoryImpact(
                title="High-priority policy shifts",
                severity=ImpactSeverity.HIGH,
                summary=f"{len(high)} section(s) contain substantive regulatory topic changes.",
                affected_areas=sorted({d.significance for d in high if d.significance}),
                recommended_actions=[
                    "Review high-severity sections with legal/compliance stakeholders.",
                    "Update control mappings and staff training for changed obligations.",
                ],
                citations=[d.section_title for d in high],
            )
        )

    if medium:
        impacts.append(
            RegulatoryImpact(
                title="Moderate policy updates",
                severity=ImpactSeverity.MEDIUM,
                summary=f"{len(medium)} section(s) were modified with moderate compliance relevance.",
                affected_areas=sorted({d.significance for d in medium if d.significance}),
                recommended_actions=[
                    "Validate whether procedural or wording changes alter operational controls.",
                ],
                citations=[d.section_title for d in medium],
            )
        )

    low_count = len(semantic_differences) - len(high) - len(medium)
    if low_count > 0:
        impacts.append(
            RegulatoryImpact(
                title="Minor or administrative changes",
                severity=ImpactSeverity.LOW,
                summary=f"{low_count} additional change(s) appear low impact but should be spot-checked.",
                recommended_actions=["Archive diff for audit trail."],
                citations=[],
            )
        )

    executive = _fallback_summary(semantic_differences)
    return impacts, executive, False


def _fallback_summary(differences: list[SemanticDifference]) -> str:
    high = sum(1 for d in differences if d.severity == ImpactSeverity.HIGH)
    medium = sum(1 for d in differences if d.severity == ImpactSeverity.MEDIUM)
    return (
        f"Compared legacy vs modernized policy: {len(differences)} material difference(s) found "
        f"({high} high, {medium} medium severity). Review highlighted sections for compliance impact."
    )
