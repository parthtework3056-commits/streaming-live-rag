"""OpenTelemetry-compliant Structured Event and Trace Logger (Component C10, SRD-SLRAG-001).

Records deterministic, auditable telemetry events for every streaming turn.
STT telemetry extension added as additive layer — does NOT change existing
record_turn() contract.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# STT Telemetry — additive extension (does NOT change existing record_turn())
# ---------------------------------------------------------------------------

class STTEventTelemetry(BaseModel):
    """Telemetry payload for a single STT lifecycle event.

    Covered events:
        stt_session_started, stt_connected, stt_partial_transcript,
        stt_final_transcript, stt_error, stt_disconnected

    Security: raw transcript text is NEVER stored here — only metadata.
    """

    event_name: str = Field(..., description="STT lifecycle event name")
    session_id: str = Field(..., description="Active session identifier")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Epoch timestamp in milliseconds",
    )
    sequence_id: int = Field(default=0, ge=0)
    provider: str = Field(default="unknown", description="STT provider identifier")
    text_length: int = Field(
        default=0, ge=0,
        description="Character count of transcript (no raw text logged for privacy)",
    )
    is_final: bool = Field(default=False)
    latency_ms: Optional[float] = Field(default=None, description="Measured latency if available")
    error_detail: Optional[str] = Field(
        default=None,
        description="Human-readable error description (must not contain secrets)",
    )

    def to_json(self) -> str:
        return self.model_dump_json()


class StructuredTraceLogger:
    """Manages emission and local storage of structured session telemetry."""

    def __init__(self, log_path: Optional[Path | str] = None) -> None:
        self.log_path = Path(log_path) if log_path else Path("logs/traces.jsonl")
        self._in_memory_traces: List[SessionTraceRecord] = []
        # STT events go to a separate log file to keep concerns separated
        self._stt_log_path = self.log_path.parent / "stt_events.jsonl"
        self._in_memory_stt_events: List[STTEventTelemetry] = []

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

    def record_stt_event(
        self,
        event_name: str,
        session_id: str,
        sequence_id: int = 0,
        provider: str = "unknown",
        text_length: int = 0,
        is_final: bool = False,
        latency_ms: Optional[float] = None,
        error_detail: Optional[str] = None,
    ) -> STTEventTelemetry:
        """Records a single STT lifecycle event.

        SECURITY: raw transcript text is NEVER logged here.
        error_detail must not contain API keys or secrets.
        """
        record = STTEventTelemetry(
            event_name=event_name,
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
            sequence_id=sequence_id,
            provider=provider,
            text_length=text_length,
            is_final=is_final,
            latency_ms=latency_ms,
            error_detail=error_detail,
        )

        self._in_memory_stt_events.append(record)

        try:
            self._stt_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stt_log_path, "a", encoding="utf-8") as f:
                f.write(record.to_json() + "\n")
        except Exception:
            pass

        return record

    def get_traces_for_session(self, session_id: str) -> List[SessionTraceRecord]:
        """Returns in-memory traces for a specific session."""
        return [t for t in self._in_memory_traces if t.session_id == session_id]

    def get_stt_events_for_session(self, session_id: str) -> List[STTEventTelemetry]:
        """Returns in-memory STT telemetry events for a specific session."""
        return [e for e in self._in_memory_stt_events if e.session_id == session_id]

    def clear(self) -> None:
        """Clears in-memory buffer."""
        self._in_memory_traces.clear()
        self._in_memory_stt_events.clear()


# Default singleton logger
default_telemetry_logger = StructuredTraceLogger()
