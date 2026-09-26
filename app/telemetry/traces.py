"""OpenTelemetry-compliant Structured Event and Trace Logger (Component C10, SRD-SLRAG-001).

Records deterministic, auditable telemetry events for every streaming turn.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ControllerDecisionTelemetry(BaseModel):
    """Telemetry payload for Controller decision."""

    decision: str = Field(..., description="WAIT, RETRIEVE, or SUPPRESS")
    reason: str = Field(..., description="Deterministic decision reason")
    latency_ms: float = Field(..., ge=0.0, description="Controller evaluation latency in milliseconds")


class IntentDecompositionTelemetry(BaseModel):
    """Telemetry payload for Intent Decomposition."""

    sub_queries: List[str] = Field(default_factory=list, description="List of generated orthogonal sub-query strings")
    latencies: Dict[str, float] = Field(default_factory=dict, description="Latency breakdown per sub-query or total in ms")


class RetrievalTelemetry(BaseModel):
    """Telemetry payload for hybrid retrieval and fusion."""

    candidate_count: int = Field(..., ge=0, description="Total candidates retrieved and fused")
    rrf_scores: Dict[str, float] = Field(default_factory=dict, description="Mapping of chunk_id to RRF fusion score")
    retrieved_chunk_ids: List[str] = Field(default_factory=list, description="Ordered list of retrieved candidate chunk IDs")


class TokenUsageTelemetry(BaseModel):
    """Telemetry payload for token consumption tracking."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class SessionTraceRecord(BaseModel):
    """Canonical OpenTelemetry-compliant structured event schema."""

    trace_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Standard 128-bit hex trace identifier",
    )
    session_id: str = Field(..., description="Session identifier")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Event epoch timestamp in milliseconds",
    )
    controller_decision: ControllerDecisionTelemetry
    intent_decomposition: IntentDecompositionTelemetry
    retrieval: RetrievalTelemetry
    answer_version: int = Field(default=1, ge=1)
    citations: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = Field(default=None)
    token_usage: TokenUsageTelemetry

    def to_json(self) -> str:
        """Serializes record to deterministic JSON string."""
        return self.model_dump_json(indent=2)


class StructuredTraceLogger:
    """Manages emission and local storage of structured session telemetry."""

    def __init__(self, log_path: Optional[Path | str] = None) -> None:
        self.log_path = Path(log_path) if log_path else Path("logs/traces.jsonl")
        self._in_memory_traces: List[SessionTraceRecord] = []

    def record_turn(
        self,
        session_id: str,
        decision: str,
        decision_reason: str,
        decision_latency_ms: float,
        sub_queries: List[str],
        decomp_latencies: Dict[str, float],
        retrieved_chunk_ids: List[str],
        rrf_scores: Dict[str, float],
        answer_version: int,
        citations: List[str],
        uncertainty: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        trace_id: Optional[str] = None,
    ) -> SessionTraceRecord:
        """Constructs, validates, and stores a structured turn trace."""
        record = SessionTraceRecord(
            trace_id=trace_id or uuid.uuid4().hex,
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
            controller_decision=ControllerDecisionTelemetry(
                decision=decision,
                reason=decision_reason,
                latency_ms=decision_latency_ms,
            ),
            intent_decomposition=IntentDecompositionTelemetry(
                sub_queries=sub_queries,
                latencies=decomp_latencies,
            ),
            retrieval=RetrievalTelemetry(
                candidate_count=len(retrieved_chunk_ids),
                rrf_scores=rrf_scores,
                retrieved_chunk_ids=retrieved_chunk_ids,
            ),
            answer_version=answer_version,
            citations=citations,
            uncertainty=uncertainty,
            token_usage=TokenUsageTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        self._in_memory_traces.append(record)

        # Append to jsonl log file
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception:
            pass

        return record

    def get_traces_for_session(self, session_id: str) -> List[SessionTraceRecord]:
        """Returns in-memory traces for a specific session."""
        return [t for t in self._in_memory_traces if t.session_id == session_id]

    def clear(self) -> None:
        """Clears in-memory buffer."""
        self._in_memory_traces.clear()


# Default singleton logger
default_telemetry_logger = StructuredTraceLogger()
