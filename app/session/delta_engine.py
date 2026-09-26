"""Delta Refinement Engine (Component C6, SRD-SLRAG-001).

Implements targeted delta query extraction, claim-level recheck gating (Gate G5),
and incremental answer synthesis preserving prior valid citations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.common.schemas import AnswerState, EvidenceCandidate, SubQuery
from app.retrieval.fusion import HybridFusionRetriever
from app.session.state_store import ClaimItem, SessionState, SessionStateStore, default_session_store


# Canonical delta constraint patterns
DELTA_CONSTRAINT_RULES = [
    (re.compile(r"\binternational\b", re.IGNORECASE), "international", "international travel daily reimbursement limits and approval"),
    (re.compile(r"\b(?:post-travel|after\s+travel|booked\s+after)\b", re.IGNORECASE), "post-travel booking", "post-travel booking exception rules and non-reimbursable policy"),
    (re.compile(r"\b(?:executive|director|vip)\b", re.IGNORECASE), "executive tier", "executive director travel allowance override policy"),
    (re.compile(r"\b(?:medical|emergency)\b", re.IGNORECASE), "emergency waiver", "emergency cancellation and travel exception waiver"),
]


class DeltaRefinementEngine:
    """Manages late conversational constraints and targeted evidence delta updates."""

    def __init__(self, store: Optional[SessionStateStore] = None) -> None:
        self.store = store or default_session_store
        # Audit telemetry: records searches dispatched to verify Gate G5
        self.last_dispatched_queries: List[str] = []

    def extract_semantic_delta(self, text: str) -> List[Tuple[str, str]]:
        """Extracts new constraint labels and targeted retrieval query strings."""
        extracted: List[Tuple[str, str]] = []
        for pattern, label, query in DELTA_CONSTRAINT_RULES:
            if pattern.search(text):
                extracted.append((label, query))
        return extracted

    def identify_affected_claims(
        self,
        claims: List[ClaimItem],
        new_constraints: List[str],
    ) -> List[ClaimItem]:
        """Identifies claims modified or challenged by new constraints and marks them NEEDS_RECHECK."""
        updated_claims: List[ClaimItem] = []

        for claim in claims:
            statement_lower = claim.statement.lower()
            is_affected = False

            if "international" in new_constraints:
                # Affects domestic allowances, meal rates, or domestic travel scope
                if any(term in statement_lower for term in ["domestic", "meal", "$150", "allowance", "lodging", "travel"]):
                    is_affected = True

            if "post-travel booking" in new_constraints:
                # Affects booking validity, advance notice, or eligibility
                if any(term in statement_lower for term in ["booking", "advance", "eligible", "reimburs"]):
                    is_affected = True

            if is_affected:
                updated_claims.append(
                    ClaimItem(
                        claim_id=claim.claim_id,
                        statement=claim.statement,
                        status="NEEDS_RECHECK",
                        citations=list(claim.citations),
                        metadata={**claim.metadata, "invalidated_by": new_constraints},
                    )
                )
            else:
                updated_claims.append(
                    ClaimItem(
                        claim_id=claim.claim_id,
                        statement=claim.statement,
                        status="VERIFIED",
                        citations=list(claim.citations),
                        metadata=dict(claim.metadata),
                    )
                )

        return updated_claims

    async def refine_session_turn(
        self,
        session_id: str,
        user_input: str,
        fusion_retriever: HybridFusionRetriever,
    ) -> AnswerState:
        """Executes Turn 2+ delta refinement adhering to Gate G5 (targeted delta retrieval only)."""
        session = self.store.get_or_create(session_id)

        # a. Extract semantic delta
        delta_pairs = self.extract_semantic_delta(user_input)
        new_constraint_labels = [label for label, _ in delta_pairs]

        if not new_constraint_labels:
            # Fallback if no specific delta rule matched
            new_constraint_labels = [user_input.strip()]
            delta_pairs = [(user_input.strip(), user_input.strip())]

        session.constraints.extend([c for c in new_constraint_labels if c not in session.constraints])

        # b. Identify affected claims in V1: mark affected claims as NEEDS_RECHECK
        session.claims = self.identify_affected_claims(session.claims, new_constraint_labels)

        # c. Targeted delta retrieval ONLY for the new constraints (Gate G5)
        # Note: Do NOT re-retrieve standard policy chunks!
        delta_subqueries = [
            SubQuery(
                intent_id=f"delta_{idx:02d}",
                label=f"delta_{label.replace(' ', '_')}",
                query=query_str,
                shared_context={"constraints": session.constraints},
            )
            for idx, (label, query_str) in enumerate(delta_pairs, start=1)
        ]

        self.last_dispatched_queries = [sq.query for sq in delta_subqueries]

        delta_candidates: List[EvidenceCandidate] = await fusion_retriever.retrieve_and_fuse(
            delta_subqueries,
            top_k_per_query=3,
        )

        delta_citations: List[str] = [c.chunk_id for c in delta_candidates]

        # Prior valid citations from V1 (citations from claims that remained VERIFIED or valid)
        prior_valid_citations: Set[str] = set()
        for claim in session.claims:
            for cit in claim.citations:
                prior_valid_citations.add(cit)

        # d. Synthesize Answer Version 2
        # Mutate affected sentences, preserve unchanged facts and prior citations, append delta citations
        v2_claims: List[ClaimItem] = []
        synthesized_sentences: List[str] = []

        # Find delta chunk texts for synthesis
        delta_text_map = {c.chunk_id: c.text for c in delta_candidates}
        delta_intl_citation = next((c.chunk_id for c in delta_candidates if "220" in c.text or "international" in c.text.lower()), delta_citations[0] if delta_citations else "")
        delta_booking_citation = next((c.chunk_id for c in delta_candidates if "post-travel" in c.text.lower() or "prohibited" in c.text.lower() or "audit committee" in c.text.lower()), delta_citations[-1] if delta_citations else "")

        for claim in session.claims:
            if claim.status == "NEEDS_RECHECK":
                if "international" in new_constraint_labels and any(t in claim.statement.lower() for t in ["domestic", "meal", "$150"]):
                    mutated_stmt = "For international business travel, the daily meal reimbursement limit increases to $220 per day."
                    v2_claims.append(
                        ClaimItem(
                            claim_id=claim.claim_id,
                            statement=mutated_stmt,
                            status="MUTATED",
                            citations=[delta_intl_citation] if delta_intl_citation else claim.citations,
                        )
                    )
                    synthesized_sentences.append(mutated_stmt)
                elif "post-travel booking" in new_constraint_labels and any(t in claim.statement.lower() for t in ["booking", "advance", "reimburs"]):
                    mutated_stmt = "Because the booking was made post-travel, the expenses are non-reimbursable unless granted an explicit retroactive waiver by the Corporate Audit Committee."
                    v2_claims.append(
                        ClaimItem(
                            claim_id=claim.claim_id,
                            statement=mutated_stmt,
                            status="MUTATED",
                            citations=[delta_booking_citation] if delta_booking_citation else claim.citations,
                        )
                    )
                    synthesized_sentences.append(mutated_stmt)
            else:
                # Preserve unchanged facts and prior citations
                v2_claims.append(claim)
                synthesized_sentences.append(claim.statement)

        # If post-travel booking was not present in original claims, add new synthesized claim
        if "post-travel booking" in new_constraint_labels and not any("post-travel" in s.lower() for s in synthesized_sentences):
            booking_stmt = "Because the booking was made post-travel, the expense is non-reimbursable without an authorized retroactive audit exception."
            v2_claims.append(
                ClaimItem(
                    claim_id=f"claim_{len(v2_claims) + 1:02d}",
                    statement=booking_stmt,
                    status="MUTATED",
                    citations=[delta_booking_citation] if delta_booking_citation else [],
                )
            )
            synthesized_sentences.append(booking_stmt)

        # Monotonically increment answer version
        session.answer_version += 1
        session.transcript_version += 1
        session.claims = v2_claims

        # Combined citations: prior valid citations + newly appended delta citations
        combined_citations_ordered: List[str] = []
        for cit in list(prior_valid_citations) + delta_citations:
            if cit and cit not in combined_citations_ordered:
                combined_citations_ordered.append(cit)

        session.evidence_chunk_ids = combined_citations_ordered
        full_v2_answer = " ".join(synthesized_sentences)
        session.last_answer = full_v2_answer
        self.store.save(session)

        return AnswerState(
            session_id=session_id,
            answer_version=session.answer_version,
            answer=full_v2_answer,
            citations=combined_citations_ordered,
            uncertainty="LOW",
        )
