"""Dense Vector Index implementation enforcing Corpus Isolation (SRD-SLRAG-001).

Supports in-memory ChromaDB ephemeral client or local SentenceTransformers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from app.common.schemas import ChunkRecord, EvidenceCandidate
from app.config import settings


class VectorIndex:
    """In-memory Vector Index adhering to strict corpus isolation."""

    def __init__(
        self,
        chunks: List[ChunkRecord],
        model_name: str = settings.embedding_model_name,
        use_chroma: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot initialize VectorIndex with empty chunk list.")

        self.chunks: List[ChunkRecord] = chunks
        self.chunk_by_id: Dict[str, ChunkRecord] = {c.chunk_id: c for c in chunks}
        self.model_name = model_name
        self.encoder = None
        self._init_encoder(model_name)

        # Pre-compute dense embeddings
        texts = [c.text for c in chunks]
        self.embeddings = self._encode(texts)

        self.chroma_collection = None
        if use_chroma:
            try:
                import chromadb
                client = chromadb.EphemeralClient()
                # Create an isolated collection with cosine distance
                self.chroma_collection = client.create_collection(
                    name="streaming_live_rag",
                    metadata={"hnsw:space": "cosine"},
                )
                self.chroma_collection.add(
                    ids=[c.chunk_id for c in chunks],
                    embeddings=self.embeddings.tolist(),
                    metadatas=[
                        {"doc_id": c.doc_id, "section": c.section, "page": c.page}
                        for c in chunks
                    ],
                    documents=texts,
                )
            except Exception:
                # Fallback to local numpy cosine search if chromadb has issues
                self.chroma_collection = None

    def _init_encoder(self, model_name: str) -> None:
        """Loads local embedding model or falls back to deterministic local dense representations."""
        try:
            from sentence_transformers import SentenceTransformer
            # Load local transformer model
            self.encoder = SentenceTransformer(model_name)
        except Exception:
            self.encoder = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized dense vectors."""
        if self.encoder is not None:
            embeddings = self.encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype("float32")

        # Fallback local deterministic embedding (useful in test/isolated environments)
        vectors = []
        for text in texts:
            # Deterministic hash-based feature representation (384 dimensions matching MiniLM)
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vectors.append(vec)
                continue
            for w in words:
                h = hash(w) % 384
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = settings.default_top_k,
    ) -> List[EvidenceCandidate]:
        """Search query against vector index and return ranked EvidenceCandidates."""
        if not query.strip():
            return []

        # If ChromaDB collection is available, use it
        if self.chroma_collection is not None:
            query_embedding = self._encode([query]).tolist()
            results = self.chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, len(self.chunks)),
            )

            candidates: List[EvidenceCandidate] = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

                for rank, (chunk_id, dist) in enumerate(zip(ids, distances), start=1):
                    chunk = self.chunk_by_id[chunk_id]
                    # Cosine distance to similarity: similarity = 1 - distance
                    score = float(1.0 - dist)
                    candidates.append(
                        EvidenceCandidate(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            section=chunk.section,
                            score=score,
                            rank=rank,
                            retriever="vector",
                            text=chunk.text,
                        )
                    )
            return candidates

        # Direct numpy cosine similarity
        query_vec = self._encode([query])[0]
        scores = np.dot(self.embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        candidates = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            chunk = self.chunks[idx]
            candidates.append(
                EvidenceCandidate(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    score=score,
                    rank=rank,
                    retriever="vector",
                    text=chunk.text,
                )
            )

        return candidates
