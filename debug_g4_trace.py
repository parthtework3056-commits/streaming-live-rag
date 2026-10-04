import asyncio
from app.synthesis.verifier import GroundingVerifier
from app.common.schemas import EvidenceCandidate

async def debug_trace():
    cand1 = EvidenceCandidate(chunk_id="chunk_1", doc_id="DOC_A", section="1.0", text="A", score=1.0, rank=1, retriever="rrf")
    cand2 = EvidenceCandidate(chunk_id="chunk_2", doc_id="DOC_B", section="2.0", text="B", score=1.0, rank=2, retriever="rrf")
    cand3 = EvidenceCandidate(chunk_id="chunk_3", doc_id="DOC_C", section="3.0", text="C", score=1.0, rank=3, retriever="rrf")
    
    candidates = [cand1, cand2, cand3]
    synthesized_sentences = []
    for cand in candidates:
        citation_tag = f"[{cand.doc_id} \u00a7{cand.section}]"
        sentence = f"Regarding this topic, the policy states something {citation_tag}."
        synthesized_sentences.append(sentence)
        
    full_answer = " ".join(synthesized_sentences)
    print("FULL ANSWER:", full_answer)
    
    import re
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", full_answer.strip())
    ]
    print("ALL SPLIT SENTENCES:", sentences)
    
    sentences_filtered = [
        s for s in sentences if len(s.split()) >= 4
    ]
    print("FILTERED SENTENCES:", sentences_filtered)
    
    verifier = GroundingVerifier()
    res = verifier.verify_answer(full_answer, candidates)
    print("VERIFICATION RESULT:", res)

if __name__ == "__main__":
    asyncio.run(debug_trace())
