from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ImpactSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DocumentSection(BaseModel):
    id: str
    title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(BaseModel):
    filename: str
    page_count: int
    full_text: str
    sections: list[DocumentSection]
    table_count: int = 0
    extraction_mode: Literal["text", "text+tables"] = "text"


class StructuralChange(BaseModel):
    change_type: ChangeType
    section_id: str
    section_title: str
    legacy_excerpt: str = ""
    modernized_excerpt: str = ""
    line_changes: int = 0
    alignment_method: Literal["title", "content", "unmatched", "llm"] | None = None


class SemanticDifference(BaseModel):
    section_title: str
    summary: str
    legacy_meaning: str
    modernized_meaning: str
    significance: str
    severity: ImpactSeverity


class RegulatoryImpact(BaseModel):
    title: str
    severity: ImpactSeverity
    summary: str
    affected_areas: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    legacy: ParsedDocument
    modernized: ParsedDocument
    structural_changes: list[StructuralChange]
    semantic_differences: list[SemanticDifference]
    regulatory_impacts: list[RegulatoryImpact]
    executive_summary: str
    stats: dict[str, int]
    llm_used: bool
    source: str | None = None
    dataset_pair_id: str | None = None
    dataset_legacy_label: str | None = None
    dataset_modernized_label: str | None = None
    format_warnings: list[str] = Field(default_factory=list)
    alignment_score: float | None = None


class CompareResponse(BaseModel):
    status: Literal["success", "error"] = "success"
    result: ComparisonResult | None = None
    message: str | None = None
