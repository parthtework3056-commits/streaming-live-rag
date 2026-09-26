"""Frozen Pydantic v2 contracts for Streaming Live RAG (SRD-SLRAG-001 / PRD-SLRAG-001)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


CHUNK_ID_PATTERN = re.compile(r"^DOC_.+_CHUNK_\d{3}$")


class ChunkRecord(BaseModel):
    """Frozen contract representing an indexed corpus chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(
        ...,
        description="Deterministic unique identifier matching DOC_{id}_CHUNK_{idx:03d}",
    )
    doc_id: str = Field(..., description="Document identifier")
    page: int = Field(..., ge=1, description="1-indexed source document page number")
    section: str = Field(..., description="Detected section header or symbol tag")
    text: str = Field(..., min_length=1, description="Cleaned chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible chunk metadata (token count, character offset, etc.)",
    )

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id_format(cls, v: str) -> str:
        if not CHUNK_ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid chunk_id '{v}'. Must strictly follow format 'DOC_{{id}}_CHUNK_{{idx:03d}}'"
            )
        return v


class ControllerDecisionEvent(BaseModel):
    """Frozen contract representing a decision made by the streaming controller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(..., description="Unique streaming session identifier")
    timestamp_ms: int = Field(..., ge=0, description="Epoch timestamp in milliseconds")
    chunk_id: str = Field(..., description="Referenced candidate or stream chunk ID")
    decision: Literal["WAIT", "RETRIEVE", "SUPPRESS"] = Field(
        ..., description="Controller policy action"
    )
    reason: str = Field(..., description="Deterministic policy or heuristic explanation")
    eligibility: bool = Field(
        ..., description="True if evidence threshold criteria are met"
    )


class SubQuery(BaseModel):
    """Frozen contract representing a decomposed sub-query or intent branch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: str = Field(..., description="Unique intent branch identifier")
    label: str = Field(..., description="Categorical or semantic intent label")
    query: str = Field(..., min_length=1, description="Synthesized search query")
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context variables shared across retrieval branches",
    )


class EvidenceCandidate(BaseModel):
    """Frozen contract representing a retrieved evidence item from lexical or vector stores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., description="Source chunk ID matching DOC_{id}_CHUNK_{idx:03d}")
    doc_id: str = Field(..., description="Source document identifier")
    section: str = Field(..., description="Source section title or tag")
    score: float = Field(..., description="Normalized or raw retrieval relevance score")
    rank: int = Field(..., ge=1, description="Rank position in retriever candidate list (1-based)")
    retriever: str = Field(..., description="Retriever identifier (e.g. 'bm25', 'vector')")
    text: str = Field(..., description="Snippet or full chunk text")


class AnswerState(BaseModel):
    """Frozen contract representing an answer version in the streaming response cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(..., description="Active session ID")
    answer_version: int = Field(
        ..., ge=1, description="Monotonically increasing answer version"
    )
    answer: str = Field(..., description="Generated answer text")
    citations: List[str] = Field(
        default_factory=list, description="List of cited chunk IDs"
    )
    uncertainty: str = Field(
        ..., description="Confidence or uncertainty indicator (e.g. 'LOW', 'MEDIUM', 'HIGH')"
    )
