"""Unit tests for Session Store (C5) and Delta Refinement Engine (C6).

Validates ephemeral session memory, claim-level recheck gating, Gate G5 targeted delta search,
and citation continuity across answer versions (SRD-SLRAG-001).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from app.common.schemas import AnswerState, EvidenceCandidate, SubQuery
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chunker import chunk_document
from app.retrieval.fusion import HybridFusionRetriever
from app.retrieval.vector_index import VectorIndex
from app.session.delta_engine import DeltaRefinementEngine
from app.session.state_store import ClaimItem, SessionState, SessionStateStore


SAMPLE_DOC_PATH = Path("corpus/raw/doc_sample.txt")


@pytest.fixture(scope="module")
def retrieval_pipeline():
    chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_CORPUS")
    bm25 = BM25Index(chunks)
    vector = VectorIndex(chunks, use_chroma=True)
    fusion = HybridFusionRetriever(bm25, vector)
    return chunks, bm25, vector, fusion


class TestSessionStore:
    """Validates Component C5 ephemeral session store and boundary isolation."""

    def test_session_state_creation_and_isolation(self):
        store = SessionStateStore()
        s1 = store.get_or_create("sess_alpha")
        s2 = store.get_or_create("sess_beta")

        s1.transcript_version = 2
        s1.constraints.append("pune")
        store.save(s1)

        # Confirm strict isolation: sess_beta unaffected
        loaded_beta = store.get("sess_beta")
        assert loaded_beta.transcript_version == 1
        assert "pune" not in loaded_beta.constraints

        # Confirm retrieval of mutated session
        loaded_alpha = store.get("sess_alpha")
        assert loaded_alpha.transcript_version == 2
        assert "pune" in loaded_alpha.constraints


class TestDeltaRefinementEngine:
    """Validates Component C6 delta extraction, claim recheck, and Gate G5 compliance."""

    def test_multi_turn_delta_refinement_and_citation_continuity(self, retrieval_pipeline):
        async def _run():
            _, _, _, fusion = retrieval_pipeline
            store = SessionStateStore()
            engine = DeltaRefinementEngine(store=store)
            session_id = "test_prism_session_001"

            # -------------------------------------------------------------
            # Simulate Turn 1: Retrieve and synthesize Answer Version 1
            # -------------------------------------------------------------
            t1_query = SubQuery(
                intent_id="intent_01",
                label="reimbursement_limits",
                query="standard domestic employee travel meal and lodging reimbursement limits",
            )
            v1_candidates = await fusion.retrieve_and_fuse([t1_query], top_k_per_query=3)
            assert len(v1_candidates) > 0

            # V1 citation for standard reimbursement (e.g. § 2.1)
            v1_primary_citation = v1_candidates[0].chunk_id

            v1_claims = [
                ClaimItem(
                    claim_id="claim_01",
                    statement="Maximum daily reimbursement limits for domestic travel shall not exceed $150 per day.",
                    status="VERIFIED",
                    citations=[v1_primary_citation],
                ),
                ClaimItem(
                    claim_id="claim_02",
                    statement="Lodging reimbursement limits are capped at $250 per night.",
                    status="VERIFIED",
                    citations=[v1_primary_citation],
                ),
                ClaimItem(
                    claim_id="claim_03",
                    statement="Receipts for any individual expense exceeding $50 must be submitted within fifteen business days.",
                    status="VERIFIED",
                    citations=[v1_primary_citation],
                ),
            ]

            session = store.get_or_create(session_id)
            session.transcript_version = 1
            session.answer_version = 1
            session.claims = v1_claims
            session.evidence_chunk_ids = [v1_primary_citation]
            session.constraints = ["domestic travel"]
            session.last_answer = " ".join([c.statement for c in v1_claims])
            store.save(session)

            assert session.answer_version == 1

            # -------------------------------------------------------------
            # Simulate Turn 2: User provides late conversational detail
            # -------------------------------------------------------------
            late_detail = "The trip was international and the booking was made after travel."

            v2_answer_state: AnswerState = await engine.refine_session_turn(
                session_id=session_id,
                user_input=late_detail,
                fusion_retriever=fusion,
            )

            # Assert 1: Answer version incremented to 2
            assert v2_answer_state.answer_version == 2
            assert store.get(session_id).answer_version == 2

            # Assert 2: Prior valid citations from V1 are retained alongside delta citations
            assert v1_primary_citation in v2_answer_state.citations
            assert len(v2_answer_state.citations) > 1

            # Verify delta citation corresponding to § 2.4 was appended
            delta_chunk_id = next(
                (cid for cid in v2_answer_state.citations if cid != v1_primary_citation),
                None,
            )
            assert delta_chunk_id is not None, "Delta citations must be appended to V2"

            # Assert 3: Gate G5 Pass Condition - Full-corpus search was NOT executed for unchanged claims
            dispatched = engine.last_dispatched_queries
            assert len(dispatched) >= 1
            # Dispatched queries must strictly be targeted delta queries
            for q in dispatched:
                q_lower = q.lower()
                assert "international" in q_lower or "post-travel" in q_lower or "after travel" in q_lower
                # Standard receipt submission deadline search must NOT be re-executed
                assert "fifteen business days" not in q_lower
                assert "receipt submitted within" not in q_lower

            # Assert 4: Content synthesis checks
            v2_text = v2_answer_state.answer
            # Domestic meal claim mutated to international rate ($220)
            assert "$220" in v2_text or "international" in v2_text.lower()
            # Post-travel booking exception synthesized
            assert "post-travel" in v2_text.lower() or "non-reimbursable" in v2_text.lower()
            # Unchanged fact preserved
            assert "fifteen business days" in v2_text

        asyncio.run(_run())
