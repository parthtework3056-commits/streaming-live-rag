from .models import Chunk


def create_chunks(pages, doc_id, chunk_size=1000, overlap=200):
    chunks = []
    chunk_number = 1

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_CHUNK_{chunk_number:03d}",
                    doc_id=doc_id,
                    page=page["page"],
                    section="",
                    text=chunk_text,
                    metadata={}
                )

                chunks.append(chunk)
                chunk_number += 1

            start += chunk_size - overlap

    return chunks