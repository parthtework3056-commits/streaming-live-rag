"""Unit tests for Grounding Verifier (C8) and Telemetry Trace Logger (C10).

Validates explicit citation marker verification, hallucination rejection,
uncertainty handling without guessing, and OpenTelemetry-compliant trace schemas (SRD-SLRAG-001).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.common.schemas import ChunkRecord, EvidenceCandidate
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chunker import chunk_document
from app.retrieval.fusion import HybridFusionRetriever
from app.retrieval.vector_index import VectorIndex
from app.synthesis.verifier import GroundingVerifier, VerificationResult
from app.telemetry.traces import SessionTraceRecord, StructuredTraceLogger


SAMPLE_DOC_PATH = Path("corpus/raw/doc_sample.txt")


@pytest.fixture(scope="module")
def sample_evidence():
    chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_CORPUS")
    return chunks


class TestGroundingVerifier:
    """Validates Component C8 Grounding Verifier constraints."""

    def test_case_1_sufficient_evidence_citation_support_above_85_percent(self, sample_evidence):
        """Test Case 1: Synthesize with sufficient evidence -> Assert citation support >= 85%."""
        verifier = GroundingVerifier()

        # Well-grounded synthetic response with explicit [Doc_ID §Section] markers
        answer_text = (
            "The maximum daily reimbursement limits for domestic travel shall not exceed $150 per day "
            "[SAMPLE_CORPUS §2.1 Reimbursement Limits & Expense Allowances]. "
            "Lodging reimbursement limits are capped at $250 per night excluding state occupancy taxes "
            "[SAMPLE_CORPUS §2.1 Reimbursement Limits & Expense Allowances]. "
            "Any individual expense exceeding $50 requires a verified itemized receipt within fifteen days "
            "[SAMPLE_CORPUS §2.1 Reimbursement Limits & Expense Allowances]."
        )

        result: VerificationResult = verifier.verify_answer(
            answer_text=answer_text,
            retrieved_evidence=sample_evidence,
        )

        # Assert citation support >= 85%
        assert result.citation_support_ratio >= 0.85, (
            f"Expected citation support >= 85%, got {result.citation_support_ratio * 100:.1f}%"
        )
        assert result.is_grounded is True
        assert result.hallucination_detected is False
        assert len(result.valid_citations) == 3
        assert len(result.hallucinated_citations) == 0

    def test_hallucinated_document_id_rejection(self, sample_evidence):
        """Validates that citing a non-existent document ID flags HALLUCINATION and rejects."""
        verifier = GroundingVerifier()

        # Response citing an invented document ID
        hallucinated_answer = (
            "Domestic travel allowances are unlimited if approved by the CEO "
            "[FICTITIOUS_DOC_99 §99.9 Secret Executive Policy]."
        )

        result: VerificationResult = verifier.verify_answer(
            answer_text=hallucinated_answer,
            retrieved_evidence=sample_evidence,
        )

        assert result.is_grounded is False
        assert result.hallucination_detected is True
        assert "HALLUCINATION" in result.rejection_reason
        assert len(result.hallucinated_citations) == 1
        assert "[FICTITIOUS_DOC_99 §99.9 Secret Executive Policy]" in result.hallucinated_citations

    def test_case_2_unindexed_topic_outputs_explicit_uncertainty(self, sample_evidence):
        """Test Case 2: Provide query with unindexed topic -> Assert system outputs explicit uncertainty string without inventing Doc IDs."""
        verifier = GroundingVerifier()

        # Unindexed query topic: "interplanetary spaceflight travel per diem"
        unindexed_topic = "interplanetary spaceflight travel per diem"

        # Answer acknowledging absence of evidence without guessing
        honest_answer = (
            "Corporate guidelines on standard travel reimbursement apply to ground and commercial flights. "
            "No interplanetary or astronautical travel guidelines could be found."
        )

        result: VerificationResult = verifier.verify_answer(
            answer_text=honest_answer,
            retrieved_evidence=sample_evidence,
            unanswered_intents=[unindexed_topic],
        )

        # Assert system outputs explicit uncertainty string
        assert result.uncertainty is not None
        assert "could not be verified from the retrieved corpus" in result.uncertainty
        assert unindexed_topic in result.uncertainty
        # Zero invented Doc IDs
        assert len(result.hallucinated_citations) == 0


class TestTelemetryTraces:
    """Validates Component C10 OpenTelemetry-compliant structured event logging."""

    def test_case_3_verify_trace_output_schema_against_required_json_fields(self, tmp_path):
        """Test Case 3: Verify trace output schema against required JSON fields."""
        log_file = tmp_path / "traces.jsonl"
        logger = StructuredTraceLogger(log_path=log_file)

        session_id = "prism_session_trace_001"
        record: SessionTraceRecord = logger.record_turn(
            session_id=session_id,
            decision="RETRIEVE",
            decision_reason="stable_actionable_intent",
            decision_latency_ms=0.125,
            sub_queries=["Pune venue capacity 30", "cancellation refund policy"],
            decomp_latencies={"total_ms": 1.45},
            retrieved_chunk_ids=["DOC_SAMPLE_CORPUS_CHUNK_003", "DOC_SAMPLE_CORPUS_CHUNK_005"],
            rrf_scores={
                "DOC_SAMPLE_CORPUS_CHUNK_003": 0.0327,
                "DOC_SAMPLE_CORPUS_CHUNK_005": 0.0312,
            },
            answer_version=1,
            citations=["DOC_SAMPLE_CORPUS_CHUNK_003"],
            uncertainty=None,
            input_tokens=142,
            output_tokens=85,
        )

        # 1. Parse serialized JSON string
        trace_json = json.loads(record.to_json())

        # 2. Verify all required OpenTelemetry-compliant schema fields
        assert "trace_id" in trace_json and isinstance(trace_json["trace_id"], str)
        assert len(trace_json["trace_id"]) == 32, "Trace ID must be 32-char hex string"
        assert trace_json["session_id"] == session_id
        assert "timestamp_ms" in trace_json and isinstance(trace_json["timestamp_ms"], int)

        # Controller decision block
        ctrl = trace_json["controller_decision"]
        assert ctrl["decision"] == "RETRIEVE"
        assert ctrl["reason"] == "stable_actionable_intent"
        assert ctrl["latency_ms"] == 0.125

        # Intent decomposition block
        decomp = trace_json["intent_decomposition"]
        assert len(decomp["sub_queries"]) == 2
        assert "Pune venue capacity 30" in decomp["sub_queries"]
        assert "latencies" in decomp

        # Retrieval block
        ret = trace_json["retrieval"]
        assert ret["candidate_count"] == 2
        assert len(ret["retrieved_chunk_ids"]) == 2
        assert "DOC_SAMPLE_CORPUS_CHUNK_003" in ret["rrf_scores"]

        # Synthesis block
        assert trace_json["answer_version"] == 1
        assert trace_json["citations"] == ["DOC_SAMPLE_CORPUS_CHUNK_003"]
        assert trace_json["uncertainty"] is None

        # Token usage block
        tokens = trace_json["token_usage"]
        assert tokens["input_tokens"] == 142
        assert tokens["output_tokens"] == 85

        # 3. Confirm file persistence in JSON Lines format
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        persisted_json = json.loads(lines[0])
        assert persisted_json["trace_id"] == trace_json["trace_id"]
