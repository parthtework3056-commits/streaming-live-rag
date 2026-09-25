from app.corpus.loader import load_pdf


pdf_path = "corpus/Streaming Live RAG_ Documentation Pack.pdf"

pages = load_pdf(pdf_path)

print(f"Total pages: {len(pages)}")

for page in pages[:3]:
    print("\n==============================")
    print(f"PAGE {page['page']}")
    print("==============================")
    print(page["text"][:1000])