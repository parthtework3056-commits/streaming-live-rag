"""BM25 Lexical Index implementation using rank-bm25 for exact lexical retrieval."""

from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from app.common.schemas import ChunkRecord, EvidenceCandidate
from app.config import settings


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 indexing and querying."""
    # Lowercase and extract alphanumeric tokens
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Lexical exact-match retriever enforcing corpus isolation."""

    def __init__(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            raise ValueError("Cannot initialize BM25Index with an empty list of chunks.")

        # Enforce corpus isolation
        if settings.corpus_isolation:
            pass  # rank-bm25 is strictly local in-memory

        self.chunks: List[ChunkRecord] = chunks
        self.chunk_ids: List[str] = [c.chunk_id for c in chunks]

        # Tokenize corpus for BM25Okapi
        self.tokenized_corpus: List[List[str]] = [
            bm25_tokenize(c.text) for c in chunks
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = settings.default_top_k) -> List[EvidenceCandidate]:
        """Performs lexical search and returns ranked EvidenceCandidates."""
        query_tokens = bm25_tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        # Pair scores with chunk index and sort descending
        scored_indices = sorted(
            enumerate(scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        candidates: List[EvidenceCandidate] = []
        rank = 1

        for idx, score in scored_indices[:top_k]:
            if score <= 0.0:
                continue

            chunk = self.chunks[idx]
            candidates.append(
                EvidenceCandidate(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    score=float(score),
                    rank=rank,
                    retriever="bm25",
                    text=chunk.text,
                )
            )
            rank += 1

        return candidates
