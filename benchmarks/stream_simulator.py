"""Streaming Transcript Replay Engine (SRD-SLRAG-001 Benchmarking Harness).

Replays timestamped audio/speech transcript chunks sequentially into the Retrieval Controller.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.common.schemas import ControllerDecisionEvent
from app.controller.stability import RetrievalController, default_controller


@dataclass(frozen=True)
class TranscriptChunk:
    """Represents a timestamped incoming speech transcript slice."""

    offset_seconds: float
    text: str
    chunk_id: Optional[str] = None


# Canonical benchmark sequence specified in Theme 04 / SRD-SLRAG-001
WORKSHOP_PLANNING_TRANSCRIPT = [
    TranscriptChunk(0.0, "I need to plan a customer workshop in..."),
    TranscriptChunk(0.8, "...Pune for 30 people, and I need..."),
    TranscriptChunk(1.6, "...the cancellation policy and the catering options."),
    TranscriptChunk(2.1, "[Utterance End]"),
]


@dataclass
class SimulationStep:
    """Records the controller evaluation at a single transcript slice."""

    chunk: TranscriptChunk
    accumulated_text: str
    decision_event: ControllerDecisionEvent
    latency_ms: float


@dataclass
class SimulationReport:
    """Aggregated performance metrics and event history for a stream simulation."""

    session_id: str
    total_chunks: int
    steps: List[SimulationStep]
    max_latency_ms: float
    avg_latency_ms: float
    retrieval_triggered: bool
    first_retrieval_offset_sec: Optional[float]
    first_retrieval_chunk_id: Optional[str]


def clean_accumulate(previous: str, new_chunk: str) -> str:
    """Intelligently joins streaming spoken speech fragments, stripping ellipses."""
    cleaned_new = re.sub(r"^\.{2,3}\s*", "", new_chunk.strip())
    cleaned_prev = re.sub(r"\s*\.{2,3}$", "", previous.strip())

    if not cleaned_prev:
        return cleaned_new
    if not cleaned_new or cleaned_new == "[Utterance End]":
        return f"{cleaned_prev} {cleaned_new}".strip()

    return f"{cleaned_prev} {cleaned_new}".strip()


class StreamSimulator:
    """Replays streaming audio transcripts into the controller and profiles performance."""

    def __init__(
        self,
        controller: Optional[RetrievalController] = None,
        real_time_replay: bool = False,
    ) -> None:
        self.controller = controller or RetrievalController()
        self.real_time_replay = real_time_replay

    def replay_stream(
        self,
        chunks: List[TranscriptChunk],
        session_id: str = "sim_session_001",
        on_decision: Optional[Callable[[SimulationStep], None]] = None,
    ) -> SimulationReport:
        """Executes sequential replay through the controller."""
        self.controller.reset_session(session_id)
        steps: List[SimulationStep] = []
        accumulated = ""
        prev_offset = 0.0

        retrieval_triggered = False
        first_retrieval_offset_sec = None
        first_retrieval_chunk_id = None

        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = chunk.chunk_id or f"TRANSCRIPT_CHUNK_{idx:03d}"
            timestamp_ms = int(chunk.offset_seconds * 1000)

            # Optional simulated real-time delay
            if self.real_time_replay and chunk.offset_seconds > prev_offset:
                time.sleep(chunk.offset_seconds - prev_offset)
                prev_offset = chunk.offset_seconds

            accumulated = clean_accumulate(accumulated, chunk.text)

            # High precision latency measurement
            t0 = time.perf_counter()
            decision = self.controller.evaluate_chunk(
                chunk_id=chunk_id,
                incoming_chunk=chunk.text,
                accumulated_text=accumulated,
                timestamp_ms=timestamp_ms,
                session_id=session_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            step = SimulationStep(
                chunk=chunk,
                accumulated_text=accumulated,
                decision_event=decision,
                latency_ms=elapsed_ms,
            )
            steps.append(step)

            if on_decision:
                on_decision(step)

            if decision.decision == "RETRIEVE" and not retrieval_triggered:
                retrieval_triggered = True
                first_retrieval_offset_sec = chunk.offset_seconds
                first_retrieval_chunk_id = chunk_id

        latencies = [s.latency_ms for s in steps]
        max_lat = max(latencies) if latencies else 0.0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

        return SimulationReport(
            session_id=session_id,
            total_chunks=len(steps),
            steps=steps,
            max_latency_ms=max_lat,
            avg_latency_ms=avg_lat,
            retrieval_triggered=retrieval_triggered,
            first_retrieval_offset_sec=first_retrieval_offset_sec,
            first_retrieval_chunk_id=first_retrieval_chunk_id,
        )
