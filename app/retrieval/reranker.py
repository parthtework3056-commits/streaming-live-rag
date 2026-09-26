"""CrossEncoder neural reranking module (SRD-SLRAG-001).

Applies lightweight CrossEncoder scoring to refine the top fused evidence candidates.
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.common.schemas import EvidenceCandidate
from app.config import settings


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks top fused candidates using a CrossEncoder or deterministic semantic alignment."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempts to load local CrossEncoder."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except Exception:
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[EvidenceCandidate],
        top_k: int = settings.default_top_k,
    ) -> List[EvidenceCandidate]:
        """Scores top-10 candidates and returns top-k refined evidence pack."""
        if not candidates:
            return []

        pool = candidates[:10]

        # Use neural CrossEncoder if loaded
        if self.model is not None:
            try:
                pairs = [(query, c.text) for c in pool]
                scores = self.model.predict(pairs)

                scored = sorted(
                    zip(pool, scores),
                    key=lambda pair: pair[1],
                    reverse=True,
                )

                return [
                    EvidenceCandidate(
                        chunk_id=cand.chunk_id,
                        doc_id=cand.doc_id,
                        section=cand.section,
                        score=float(score),
                        rank=rank,
                        retriever="cross_encoder",
                        text=cand.text,
                    )
                    for rank, (cand, score) in enumerate(scored[:top_k], start=1)
                ]
            except Exception:
                pass

        # Fallback deterministic lexical-semantic overlap alignment
        q_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        scored_fallback = []
        for cand in pool:
            c_tokens = set(re.findall(r"\b\w{3,}\b", cand.text.lower()))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            composite_score = cand.score + (0.5 * overlap)
            scored_fallback.append((cand, composite_score))

        scored_fallback.sort(key=lambda pair: pair[1], reverse=True)

        return [
            EvidenceCandidate(
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                section=cand.section,
                score=float(score),
                rank=rank,
                retriever=cand.retriever,
                text=cand.text,
            )
            for rank, (cand, score) in enumerate(scored_fallback[:top_k], start=1)
        ]
