"""Unit tests for Component C1 (Retrieval Controller) and streaming transcript replay engine.

Verifies deterministic decisions (WAIT, RETRIEVE, SUPPRESS), < 15ms latency SLA,
and Gate G2 early retrieval before utterance end (SRD-SLRAG-001).
"""

from __future__ import annotations

import time
import pytest

from app.controller.rules import evaluate_presentation_rules
from app.controller.stability import RetrievalController, evaluate_stream_chunk
from benchmarks.stream_simulator import (
    StreamSimulator,
    TranscriptChunk,
    WORKSHOP_PLANNING_TRANSCRIPT,
)


class TestRetrievalController:
    """Validates core decision logic and SLA compliance."""

    @pytest.fixture
    def controller(self):
        return RetrievalController()

    def test_case_a_partial_fragment_returns_wait(self, controller):
        """Test Case A: Partial fragment ('I was wondering...') returns WAIT."""
        fragment = "I was wondering..."
        event = controller.evaluate_chunk(
            chunk_id="CHUNK_001",
            incoming_chunk=fragment,
            accumulated_text=fragment,
            timestamp_ms=0,
            session_id="test_sess_a",
        )

        assert event.decision == "WAIT"
        assert event.reason == "incomplete_intent"
        assert event.eligibility is False

    def test_case_b_actionable_fragment_at_0_8s_returns_retrieve_before_utterance_end(self, controller):
        """Test Case B: Actionable fragment at 0.8s returns RETRIEVE before utterance end (Validating Gate G2)."""
        simulator = StreamSimulator(controller=controller)
        report = simulator.replay_stream(
            WORKSHOP_PLANNING_TRANSCRIPT,
            session_id="test_sess_b",
        )

        # Total 4 chunks in benchmark transcript
        assert report.total_chunks == 4

        # Chunk 0: 0.0s "I need to plan a customer workshop in..." -> WAIT
        step_0 = report.steps[0]
        assert step_0.chunk.offset_seconds == 0.0
        assert step_0.decision_event.decision == "WAIT"
        assert step_0.decision_event.reason == "incomplete_intent"

        # Chunk 1: 0.8s "...Pune for 30 people, and I need..." -> RETRIEVE
        step_1 = report.steps[1]
        assert step_1.chunk.offset_seconds == 0.8
        assert step_1.decision_event.decision == "RETRIEVE"
        assert step_1.decision_event.reason == "stable_actionable_intent"
        assert step_1.decision_event.eligibility is True

        # Validate Gate G2: Earliest RETRIEVE triggered at 0.8s strictly before utterance end at 2.1s
        assert report.retrieval_triggered is True
        assert report.first_retrieval_offset_sec == 0.8
        assert report.first_retrieval_offset_sec < 2.1

        # Validate SLA: decision latency under 15ms per chunk
        for step in report.steps:
            assert step.latency_ms < 15.0, f"Step latency {step.latency_ms}ms exceeded 15ms SLA"

    def test_case_c_presentation_restructure_returns_suppress_zero_searches(self, controller):
        """Test Case C: 'Can you turn the last answer into 3 bullets?' returns SUPPRESS with zero searches triggered."""
        query = "Can you turn the last answer into 3 bullets?"
        event = controller.evaluate_chunk(
            chunk_id="CHUNK_RESTYLE",
            incoming_chunk=query,
            accumulated_text=query,
            timestamp_ms=1000,
            session_id="test_sess_c",
        )

        assert event.decision == "SUPPRESS"
        assert event.reason == "presentation_restructure"
        assert event.eligibility is False

    @pytest.mark.parametrize(
        "prompt",
        [
            "repeat in bullets",
            "summarize in 2 points",
            "format as bullet points",
            "translate into Spanish",
            "turn that into 5 bullets please",
            "can you restate that in bullets?",
        ],
    )
    def test_presentation_restructure_variations(self, controller, prompt):
        """Validates coverage of presentation and formatting suppression heuristics."""
        event = controller.evaluate_chunk(
            chunk_id="CHUNK_VAR",
            incoming_chunk=prompt,
            accumulated_text=prompt,
            timestamp_ms=500,
            session_id="test_sess_var",
        )
        assert event.decision == "SUPPRESS"
        assert event.reason == "presentation_restructure"
        assert event.eligibility is False

    def test_decision_latency_sla_under_15ms(self, controller):
        """Microbenchmark ensuring deterministic execution well below 15ms."""
        incoming = "...Pune for 30 people, and I need..."
        accumulated = "I need to plan a customer workshop in Pune for 30 people, and I need..."

        # Warm up
        controller.evaluate_chunk("CH_WARM", incoming, accumulated, 800)

        # 100 iterations timing
        latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            controller.evaluate_chunk(f"CH_{i}", incoming, accumulated, 800)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)

        assert max_latency < 15.0, f"Max latency was {max_latency:.3f}ms (threshold 15ms)"
        # Typically executes in < 0.2ms
        assert avg_latency < 2.0, f"Average latency was {avg_latency:.3f}ms"
