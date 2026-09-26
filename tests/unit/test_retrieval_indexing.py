"""Unit tests for retrieval indexing pipeline, frozen contracts, and determinism.

Adheres strictly to SRD-SLRAG-001 and PRD-SLRAG-001.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.common.schemas import (
    ChunkRecord,
    ControllerDecisionEvent,
    SubQuery,
    EvidenceCandidate,
    AnswerState,
)
from app.retrieval.chunker import (
    SectionAwareChunker,
    chunk_document,
    generate_corpus_manifest,
)
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vector_index import VectorIndex


SAMPLE_DOC_PATH = Path("corpus/raw/doc_sample.txt")


class TestFrozenContracts:
    """Validates frozen Pydantic v2 schemas and validation constraints."""

    def test_chunk_record_valid_and_frozen(self):
        record = ChunkRecord(
            chunk_id="DOC_SAMPLE_CHUNK_001",
            doc_id="SAMPLE",
            page=1,
            section="§ 1.0 General",
            text="Valid test chunk text.",
            metadata={"token_count": 4},
        )
        assert record.chunk_id == "DOC_SAMPLE_CHUNK_001"
        assert record.page == 1

        # Must be immutable (frozen)
        with pytest.raises(ValidationError):
            record.chunk_id = "DOC_SAMPLE_CHUNK_002"

    def test_chunk_record_invalid_id_format(self):
        # Invalid format should fail validation
        with pytest.raises(ValidationError):
            ChunkRecord(
                chunk_id="INVALID_CHUNK_FORMAT",
                doc_id="SAMPLE",
                page=1,
                section="Section 1",
                text="Some text",
            )

    def test_controller_decision_event_contract(self):
        event = ControllerDecisionEvent(
            session_id="sess_1234",
            timestamp_ms=1720000000000,
            chunk_id="DOC_SAMPLE_CHUNK_001",
            decision="RETRIEVE",
            reason="High semantic relevance threshold met",
            eligibility=True,
        )
        assert event.decision == "RETRIEVE"
        assert event.eligibility is True

        with pytest.raises(ValidationError):
            event.decision = "WAIT"

        # Invalid decision literal
        with pytest.raises(ValidationError):
            ControllerDecisionEvent(
                session_id="sess_1234",
                timestamp_ms=1720000000000,
                chunk_id="DOC_SAMPLE_CHUNK_001",
                decision="INVALID_ACTION",  # type: ignore
                reason="Invalid",
                eligibility=False,
            )

    def test_subquery_and_evidence_and_answer_state_frozen(self):
        sub_q = SubQuery(
            intent_id="intent_01",
            label="policy_lookup",
            query="reimbursement limits for meals",
            shared_context={"user_tier": "gold"},
        )
        with pytest.raises(ValidationError):
            sub_q.label = "changed"

        evidence = EvidenceCandidate(
            chunk_id="DOC_SAMPLE_CHUNK_002",
            doc_id="SAMPLE",
            section="§ 2.1 Reimbursement Limits",
            score=0.92,
            rank=1,
            retriever="bm25",
            text="Meal limit text snippet",
        )
        with pytest.raises(ValidationError):
            evidence.score = 0.99

        state = AnswerState(
            session_id="sess_1234",
            answer_version=1,
            answer="Domestic meal limits are capped at $150 per day.",
            citations=["DOC_SAMPLE_CHUNK_002"],
            uncertainty="LOW",
        )
        with pytest.raises(ValidationError):
            state.answer_version = 2


class TestChunkingAndDeterminism:
    """Verifies section-aware chunking, header detection, and manifest stability."""

    def test_section_tag_detection(self):
        chunker = SectionAwareChunker()
        chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_DOC", chunker=chunker)
        assert len(chunks) >= 3

        # Verify chunk IDs follow DOC_{id}_CHUNK_{idx:03d}
        for idx, chunk in enumerate(chunks, start=1):
            expected_id = f"DOC_SAMPLE_DOC_CHUNK_{idx:03d}"
            assert chunk.chunk_id == expected_id

        # Verify section symbols (§) are captured in section field
        sections = [c.section for c in chunks]
        assert any("§ 2.1" in s for s in sections)
        assert any("§ 3.4" in s for s in sections)

    def test_chunking_determinism_across_multiple_runs(self):
        """Chunk IDs and contents must be strictly identical across repeated runs."""
        run1 = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_DOC")
        run2 = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_DOC")

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.section == c2.section
            assert c1.text == c2.text
            assert c1.metadata == c2.metadata

    def test_corpus_manifest_emission(self, tmp_path):
        chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_DOC")
        manifest_file = tmp_path / "manifest.json"
        emitted_path = generate_corpus_manifest(chunks, manifest_path=manifest_file)

        assert emitted_path.exists()
        data = json.loads(emitted_path.read_text(encoding="utf-8"))

        assert data["schema_version"] == "1.0.0"
        assert data["total_chunks"] == len(chunks)
        assert "SAMPLE_DOC" in data["documents"]
        assert len(data["chunks"]) == len(chunks)


class TestRetrievalIndexing:
    """Verifies BM25 and Vector index search candidate retrieval and isolation."""

    @pytest.fixture(scope="module")
    def indexed_corpus(self):
        chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="POL_01")
        bm25_idx = BM25Index(chunks)
        vec_idx = VectorIndex(chunks, use_chroma=True)
        return chunks, bm25_idx, vec_idx

    def test_bm25_reimbursement_limits_query(self, indexed_corpus):
        _, bm25_idx, _ = indexed_corpus
        results = bm25_idx.search("reimbursement limits", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.retriever == "bm25"
        assert top.rank == 1
        assert "reimbursement" in top.text.lower()
        assert "§ 2.1" in top.section

    def test_bm25_cancellation_terms_query(self, indexed_corpus):
        _, bm25_idx, _ = indexed_corpus
        results = bm25_idx.search("cancellation terms", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.retriever == "bm25"
        assert "cancellation" in top.text.lower()
        assert "§ 3.4" in top.section

    def test_vector_reimbursement_limits_query(self, indexed_corpus):
        _, _, vec_idx = indexed_corpus
        results = vec_idx.search("reimbursement limits", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.retriever == "vector"
        assert top.rank == 1
        # Top retrieved chunk should correspond to reimbursement section
        assert "reimbursement" in top.text.lower()
        assert "DOC_POL_01_CHUNK_" in top.chunk_id

    def test_vector_cancellation_terms_query(self, indexed_corpus):
        _, _, vec_idx = indexed_corpus
        results = vec_idx.search("cancellation terms", top_k=3)

        assert len(results) > 0
        top = results[0]
        assert top.retriever == "vector"
        assert top.rank == 1
        assert "cancellation" in top.text.lower()
        assert "DOC_POL_01_CHUNK_" in top.chunk_id

    def test_both_retrievers_return_same_target_chunk_id(self, indexed_corpus):
        """Cross-retriever validation confirming semantic and lexical consensus."""
        _, bm25_idx, vec_idx = indexed_corpus

        bm25_res = bm25_idx.search("reimbursement limits", top_k=1)
        vec_res = vec_idx.search("reimbursement limits", top_k=1)

        assert len(bm25_res) == 1
        assert len(vec_res) == 1
        # Both should prioritize the reimbursement chunk
        assert bm25_res[0].chunk_id == vec_res[0].chunk_id

        bm25_cancel = bm25_idx.search("cancellation terms", top_k=1)
        vec_cancel = vec_idx.search("cancellation terms", top_k=1)

        assert len(bm25_cancel) == 1
        assert len(vec_cancel) == 1
        assert bm25_cancel[0].chunk_id == vec_cancel[0].chunk_id
