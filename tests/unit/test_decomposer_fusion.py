"""Unit tests for Multi-Intent Decomposer (C2) and Parallel Hybrid Fusion (C3/C4).

Validates Gate G3 decomposition, shared context preservation, RRF score fusion,
and multi-intent evidence candidate retrieval.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from app.common.schemas import SubQuery
from app.intent.decomposer import MultiIntentDecomposer
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chunker import chunk_document
from app.retrieval.fusion import HybridFusionRetriever, RRF_K
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_index import VectorIndex


SAMPLE_DOC_PATH = Path("corpus/raw/doc_sample.txt")


@pytest.fixture(scope="module")
def corpus_indexes():
    chunks = chunk_document(SAMPLE_DOC_PATH, doc_id="SAMPLE_CORPUS")
    bm25 = BM25Index(chunks)
    vector = VectorIndex(chunks, use_chroma=True)
    fusion = HybridFusionRetriever(bm25, vector)
    return chunks, bm25, vector, fusion


class TestMultiIntentDecomposer:
    """Validates Component C2 Multi-Intent Decomposition and Anti-Overfragmentation."""

    def test_decomposer_gate_g3_and_shared_context(self):
        transcript = "Pune for 30 people, cancellation policy and catering options"
        decomposer = MultiIntentDecomposer()

        sub_queries = decomposer.decompose(transcript)

        # Gate G3 verification: Must decompose compound input into at least 2 orthogonal sub-queries
        assert len(sub_queries) >= 2, f"Gate G3 failed: expected >= 2 sub-queries, got {len(sub_queries)}"
        assert len(sub_queries) <= 3, f"Anti-overfragmentation violated: got {len(sub_queries)} sub-queries"

        # Verify shared context preservation across sub-queries
        for sq in sub_queries:
            assert isinstance(sq, SubQuery)
            assert sq.shared_context.get("location") == "Pune"
            assert sq.shared_context.get("headcount") == 30

        # Verify orthogonal intents represented
        labels = [sq.label for sq in sub_queries]
        assert "venue_capacity" in labels or any("venue" in l or "capacity" in l for l in labels)
        assert "cancellation_policy" in labels or any("cancellation" in l for l in labels)

    def test_anti_overfragmentation_single_intent(self):
        """Single-intent query should not be arbitrarily split."""
        decomposer = MultiIntentDecomposer()
        single_query = "What is the reimbursement limit for dinner?"
        sub_queries = decomposer.decompose(single_query)

        assert len(sub_queries) == 1
        assert "reimbursement" in sub_queries[0].query.lower()


class TestHybridFusionAndDeduplication:
    """Validates Component C3/C4 Parallel Retrieval, RRF Fusion, and Candidate Selection."""

    def test_parallel_retrieval_and_rrf_multi_intent_coverage(self, corpus_indexes):
        async def _run():
            _, _, _, fusion = corpus_indexes
            transcript = "Pune for 30 people, cancellation policy and catering options"
            decomposer = MultiIntentDecomposer()
            sub_queries = decomposer.decompose(transcript)

            # 1. Parallel retrieval and RRF fusion
            candidates = await fusion.retrieve_and_fuse(sub_queries, top_k_per_query=5)

            assert len(candidates) > 0

            # Verify RRF scoring properties
            for cand in candidates:
                assert cand.retriever == "rrf_hybrid"
                assert cand.score > 0.0
                assert cand.score <= (len(sub_queries) * 2) * (1.0 / (RRF_K + 1))

            # 2. Deduplication verification: Unique chunk IDs and monotonically increasing ranks
            chunk_ids = [c.chunk_id for c in candidates]
            assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in fused output"
            ranks = [c.rank for c in candidates]
            assert ranks == list(range(1, len(candidates) + 1)), "Ranks must be contiguous 1-indexed"

            # 3. Multi-intent coverage verification:
            # Candidate list must contain evidence answering BOTH venue and cancellation policies
            sections_retrieved = [c.section.lower() for c in candidates]
            texts_retrieved = [c.text.lower() for c in candidates]

            has_venue_evidence = any(
                ("venue" in s or "capacity" in s or "pune" in s) or
                ("pune" in t and ("30 people" in t or "capacity" in t or "workshop" in t))
                for s, t in zip(sections_retrieved, texts_retrieved)
            )
            has_cancellation_evidence = any(
                ("cancellation" in s or "refund" in s) or
                ("cancellation" in t or "refund" in t)
                for s, t in zip(sections_retrieved, texts_retrieved)
            )

            assert has_venue_evidence, "Failed to retrieve venue capacity evidence for Pune"
            assert has_cancellation_evidence, "Failed to retrieve cancellation policy evidence"

        asyncio.run(_run())

    def test_cross_encoder_reranker(self, corpus_indexes):
        """Validates that CrossEncoder reranker scores and returns top-k pack."""
        async def _run():
            _, _, _, fusion = corpus_indexes
            reranker = CrossEncoderReranker()

            sub_query = SubQuery(
                intent_id="intent_01",
                label="cancellation_policy",
                query="cancellation terms and refund conditions",
            )
            candidates = await fusion.retrieve_and_fuse([sub_query], top_k_per_query=3)
            reranked = reranker.rerank("cancellation terms and refund conditions", candidates, top_k=3)

            assert len(reranked) > 0
            assert len(reranked) <= 3
            # Top reranked result should be cancellation policy
            top = reranked[0]
            assert "cancellation" in top.text.lower()
            assert top.rank == 1

        asyncio.run(_run())
