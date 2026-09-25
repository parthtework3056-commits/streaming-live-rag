from app.corpus.loader import load_pdf
from app.corpus.chunker import create_chunks


pdf_path = "corpus/Streaming Live RAG_ Documentation Pack.pdf"

pages = load_pdf(pdf_path)

chunks = create_chunks(
    pages,
    doc_id="DOC_01",
    chunk_size=1000,
    overlap=200
)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:5]:
    print("\n==============================")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Document: {chunk.doc_id}")
    print(f"Page: {chunk.page}")
    print("==============================")
    print(chunk.text[:300])