import faiss

from sentence_transformers import SentenceTransformer

from app.retrieval.evidence import EvidenceItem, EvidencePack


class Retriever:

    def __init__(self, chunks):
        self.chunks = chunks

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Creating embeddings...")

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.embeddings = embeddings.astype("float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(self.embeddings)

        print(f"FAISS index created with {len(chunks)} chunks.")

    def search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "score": float(score),
                "text": chunk.text
            })

        return EvidencePack(
    query=query,
    items=evidence_items
)