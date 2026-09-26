"""Parallel asynchronous retrieval dispatch, Reciprocal Rank Fusion (RRF), and deduplication.

Adheres to Component C3/C4 (SRD-SLRAG-001).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple

from app.common.schemas import EvidenceCandidate, SubQuery
from app.config import settings
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vector_index import VectorIndex


RRF_K = 60  # Standard smoothing constant for Reciprocal Rank Fusion


def compute_text_hash(text: str) -> str:
    """Computes a normalized SHA-256 fingerprint for text deduplication."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """Computes token Jaccard similarity between two text snippets."""
    tokens_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class HybridFusionRetriever:
    """Coordinates parallel sparse+dense retrieval and Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        rrf_k: int = RRF_K,
        near_duplicate_threshold: float = 0.88,
    ) -> None:
        self.bm25 = bm25_index
        self.vector = vector_index
        self.rrf_k = rrf_k
        self.near_duplicate_threshold = near_duplicate_threshold

    def _retrieve_single_query_sync(
        self,
        sub_query: SubQuery,
        top_k: int,
    ) -> Tuple[SubQuery, List[EvidenceCandidate], List[EvidenceCandidate]]:
        """Synchronously queries BM25 and Vector indexes."""
        bm25_results = self.bm25.search(sub_query.query, top_k=top_k)
        vector_results = self.vector.search(sub_query.query, top_k=top_k)
        return sub_query, bm25_results, vector_results

    async def _retrieve_single_query_async(
        self,
        sub_query: SubQuery,
        top_k: int,
    ) -> Tuple[SubQuery, List[EvidenceCandidate], List[EvidenceCandidate]]:
        """Asynchronously dispatches BM25 and Vector searches concurrently."""
        loop = asyncio.get_running_loop()
        bm25_task = loop.run_in_executor(None, self.bm25.search, sub_query.query, top_k)
        vector_task = loop.run_in_executor(None, self.vector.search, sub_query.query, top_k)
        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)
        return sub_query, bm25_results, vector_results

    async def parallel_retrieve(
        self,
        sub_queries: List[SubQuery],
        top_k_per_query: int = settings.default_top_k,
    ) -> List[Tuple[SubQuery, List[EvidenceCandidate], List[EvidenceCandidate]]]:
        """Asynchronously dispatches retrieval across all sub-queries using asyncio.gather."""
        tasks = [
            self._retrieve_single_query_async(sq, top_k=top_k_per_query)
            for sq in sub_queries
        ]
        return await asyncio.gather(*tasks)

    def reciprocal_rank_fusion(
        self,
        retrieval_batches: List[Tuple[SubQuery, List[EvidenceCandidate], List[EvidenceCandidate]]],
    ) -> List[EvidenceCandidate]:
        """Merges multiple ranking lists via RRF: RRF_Score(d) = sum(1.0 / (60 + rank_i(d)))."""
        # Map: chunk_id -> accumulated RRF score
        rrf_scores: Dict[str, float] = {}
        # Map: chunk_id -> representative EvidenceCandidate
        chunk_candidates: Dict[str, EvidenceCandidate] = {}

        for _, bm25_list, vec_list in retrieval_batches:
            # Score BM25 candidates
            for cand in bm25_list:
                score_increment = 1.0 / (self.rrf_k + cand.rank)
                rrf_scores[cand.chunk_id] = rrf_scores.get(cand.chunk_id, 0.0) + score_increment
                if cand.chunk_id not in chunk_candidates:
                    chunk_candidates[cand.chunk_id] = cand

            # Score Vector candidates
            for cand in vec_list:
                score_increment = 1.0 / (self.rrf_k + cand.rank)
                rrf_scores[cand.chunk_id] = rrf_scores.get(cand.chunk_id, 0.0) + score_increment
                if cand.chunk_id not in chunk_candidates:
                    chunk_candidates[cand.chunk_id] = cand

        # Sort chunk IDs by descending RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_candidates: List[EvidenceCandidate] = []
        for rank, cid in enumerate(sorted_chunk_ids, start=1):
            base = chunk_candidates[cid]
            fused_candidates.append(
                EvidenceCandidate(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    section=base.section,
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    retriever="rrf_hybrid",
                    text=base.text,
                )
            )

        return fused_candidates

    def deduplicate(
        self,
        candidates: List[EvidenceCandidate],
    ) -> List[EvidenceCandidate]:
        """Eliminates exact text-hash duplicates and near-duplicate passages."""
        deduped: List[EvidenceCandidate] = []
        seen_hashes: Set[str] = set()
        seen_chunk_ids: Set[str] = set()

        for cand in candidates:
            # 1. Exact chunk_id check
            if cand.chunk_id in seen_chunk_ids:
                continue

            # 2. Text hash fingerprinting
            thash = compute_text_hash(cand.text)
            if thash in seen_hashes:
                continue

            # 3. Near-duplicate cosine/Jaccard suppression against accepted candidates
            is_near_dup = False
            for accepted in deduped:
                sim = compute_token_jaccard(cand.text, accepted.text)
                if sim >= self.near_duplicate_threshold:
                    is_near_dup = True
                    break

            if is_near_dup:
                continue

            seen_chunk_ids.add(cand.chunk_id)
            seen_hashes.add(thash)
            deduped.append(cand)

        # Re-index ranks monotonically
        return [
            EvidenceCandidate(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                section=c.section,
                score=c.score,
                rank=new_rank,
                retriever=c.retriever,
                text=c.text,
            )
            for new_rank, c in enumerate(deduped, start=1)
        ]

    async def retrieve_and_fuse(
        self,
        sub_queries: List[SubQuery],
        top_k_per_query: int = settings.default_top_k,
    ) -> List[EvidenceCandidate]:
        """Full pipeline: Parallel dispatch -> Reciprocal Rank Fusion -> Deduplication."""
        batches = await self.parallel_retrieve(sub_queries, top_k_per_query=top_k_per_query)
        fused = self.reciprocal_rank_fusion(batches)
        return self.deduplicate(fused)
