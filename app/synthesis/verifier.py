"""Grounding Verifier (Component C8, SRD-SLRAG-001).

Validates explicit citation markers [Doc_ID §Section], flags hallucinated document IDs,
computes citation support ratio, and enforces honest uncertainty without guessing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.common.schemas import ChunkRecord, EvidenceCandidate, SubQuery


# Regex matching [Doc_ID §Section]
# Examples: [SAMPLE_CORPUS §2.1 Reimbursement Limits], [DOC_01 §3.4 Cancellation Terms], [DOC_SAMPLE_CHUNK_001 §1.0]
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_\-]+)\s+§\s*([^\]]+)\]")


class VerificationResult(BaseModel):
    """Result emitted by the Grounding Verifier."""

    is_grounded: bool = Field(..., description="True if no hallucinations detected and citations are valid")
    hallucination_detected: bool = Field(..., description="True if non-existent document IDs are cited")
    rejection_reason: Optional[str] = Field(default=None, description="Explanation if rejected")
    citation_support_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of assertions supported by evidence")
    verification_status: str = Field(..., description="Explicit status: VERIFIED, PARTIALLY GROUNDED, or INSUFFICIENT EVIDENCE")
    total_claims: int = Field(default=0)
    supported_claims: int = Field(default=0)
    valid_citations: List[str] = Field(default_factory=list)
    hallucinated_citations: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = Field(default=None, description="Explicit uncertainty declaration if intent unanswerable")


class GroundingVerifier:
    """Verifies citation accuracy, cross-checks retrieved evidence, and detects hallucinations."""

    @staticmethod
    def extract_citations(text: str) -> List[Tuple[str, str, str]]:
        """Extracts all [Doc_ID §Section] citation markers.
        
        Returns:
            List of (full_marker, doc_id, section)
        """
        matches = []
        for m in CITATION_PATTERN.finditer(text):
            full_marker = m.group(0)
            doc_id = m.group(1).strip()
            section = m.group(2).strip()
            matches.append((full_marker, doc_id, section))
        return matches

    @staticmethod
    def _is_doc_id_matching(cited_doc: str, cand: EvidenceCandidate | ChunkRecord) -> bool:
        """Determines if cited doc_id matches candidate doc_id or chunk_id."""
        cited = cited_doc.upper().strip()
        cand_doc = cand.doc_id.upper().strip()
        cand_chunk = cand.chunk_id.upper().strip()

        if cited == cand_doc or cited == cand_chunk:
            return True
        if cited in cand_chunk:
            return True
        # Match clean prefix / suffix (e.g. DOC_SAMPLE_CORPUS vs SAMPLE_CORPUS)
        clean_cited = re.sub(r"^DOC_", "", cited)
        clean_cand = re.sub(r"^DOC_", "", cand_doc)
        if clean_cited == clean_cand:
            return True

        return False

    @staticmethod
    def _is_section_matching(cited_sec: str, cand_section: str) -> bool:
        """Determines if cited section matches candidate section."""
        c_sec = cited_sec.lower().strip()
        cand_s = cand_section.lower().strip()

        # Check section number or tag match (e.g. "2.1" in "§ 2.1 Reimbursement Limits")
        sec_num_match = re.search(r"\d+(?:\.\d+)*", c_sec)
        if sec_num_match:
            sec_num = sec_num_match.group(0)
            if sec_num in cand_s:
                return True

        # Check text overlap
        c_words = set(re.findall(r"\b\w{3,}\b", c_sec))
        cand_words = set(re.findall(r"\b\w{3,}\b", cand_s))
        if c_words and len(c_words & cand_words) > 0:
            return True

        return False

    def verify_answer(
        self,
        answer_text: str,
        retrieved_evidence: List[EvidenceCandidate | ChunkRecord],
        unanswered_intents: Optional[List[str]] = None,
    ) -> VerificationResult:
        """Cross-checks all factual assertions and citations against retrieved evidence.
        
        Rules:
        - If non-existent document ID cited -> Flag as HALLUCINATION and reject.
        - Computes citation_support_ratio (must be >= 0.85 for full support).
        - If an intent cannot be answered -> Populates uncertainty field explicitly.
        """
        citations = self.extract_citations(answer_text)

        valid_citations: List[str] = []
        hallucinated_citations: List[str] = []

        # If no evidence retrieved, any cited doc is hallucinated
        valid_doc_ids = {c.doc_id.upper() for c in retrieved_evidence}
        valid_chunk_ids = {c.chunk_id.upper() for c in retrieved_evidence}

        for full_marker, doc_id, section in citations:
            # Check if doc_id exists in retrieved evidence
            matching_cands = [
                c for c in retrieved_evidence if self._is_doc_id_matching(doc_id, c)
            ]

            if not matching_cands:
                # LLM cited a non-existent document ID!
                hallucinated_citations.append(full_marker)
            else:
                # Check if any candidate in the retrieved evidence matches the cited section
                matched_cand = next(
                    (c for c in matching_cands if self._is_section_matching(section, c.section)),
                    None,
                )
                if matched_cand is not None:
                    valid_citations.append(full_marker)
                else:
                    # Section hallucination or mismatch
                    hallucinated_citations.append(full_marker)

        # Flag hallucination and reject if any non-existent doc cited
        if hallucinated_citations:
            return VerificationResult(
                is_grounded=False,
                hallucination_detected=True,
                rejection_reason=f"HALLUCINATION: Non-existent or mismatched citation(s) detected: {hallucinated_citations}",
                citation_support_ratio=0.0,
                verification_status="INSUFFICIENT EVIDENCE",
                total_claims=len(citations),
                supported_claims=len(valid_citations),
                valid_citations=valid_citations,
                hallucinated_citations=hallucinated_citations,
                uncertainty=None,
            )

        # Compute citation support ratio across factual sentences
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text.strip())
            if len(s.strip().split()) >= 4
        ]

        if not sentences:
            support_ratio = 1.0 if citations else 0.0
        else:
            supported_sentences = 0
            for sent in sentences:
                if any(marker in sent for marker in valid_citations):
                    supported_sentences += 1
            support_ratio = float(supported_sentences / len(sentences))

        # Handle unanswered intents
        uncertainty = None
        if unanswered_intents:
            unanswered_str = ", ".join(unanswered_intents)
            uncertainty = f"'{unanswered_str}' could not be verified from the retrieved corpus."

        is_grounded = (support_ratio >= 0.85) if sentences else True
        
        if support_ratio >= 1.0:
            status = "VERIFIED"
        elif support_ratio > 0.0:
            status = "PARTIALLY GROUNDED"
        else:
            status = "INSUFFICIENT EVIDENCE"

        return VerificationResult(
            is_grounded=is_grounded,
            hallucination_detected=False,
            rejection_reason=None if is_grounded else f"Insufficient citation support: {support_ratio * 100:.1f}% < 85%",
            citation_support_ratio=support_ratio,
            verification_status=status,
            total_claims=len(sentences),
            supported_claims=supported_sentences if sentences else 0,
            valid_citations=valid_citations,
            hallucinated_citations=[],
            uncertainty=uncertainty,
        )


# Global verifier instance
default_verifier = GroundingVerifier()
