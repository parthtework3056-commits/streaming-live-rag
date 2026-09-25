from app.corpus.loader import load_pdf
from app.corpus.chunker import create_chunks
from app.retrieval.retriever import Retriever


# Load PDF
pdf_path = "corpus/Streaming Live RAG_ Documentation Pack.pdf"

pages = load_pdf(pdf_path)


# Create chunks
chunks = create_chunks(
    pages,
    doc_id="DOC_01",
    chunk_size=1000,
    overlap=200
)


# Create retriever
retriever = Retriever(chunks)


# Query
query = "What are the four required capabilities of Streaming Live RAG?"


# Retrieve evidence
evidence_pack = retriever.search(
    query,
    top_k=5
)


# Display EvidencePack
print("\n========================================")
print("EVIDENCE PACK")
print("========================================")

print(f"Query: {evidence_pack.query}")
print(f"Evidence items: {len(evidence_pack.items)}")


for i, item in enumerate(evidence_pack.items, start=1):

    print(f"\n--- Evidence {i} ---")
    print(f"Chunk ID : {item.chunk_id}")
    print(f"Document : {item.doc_id}")
    print(f"Page    : {item.page}")
    print(f"Score   : {item.score:.4f}")

    print("\nText:")
    print(item.text[:500])