"""Multi-Intent Decomposer (Component C2, SRD-SLRAG-001).

Decomposes compound conversational transcripts into 1-3 orthogonal search sub-queries
while enforcing anti-overfragmentation rules and preserving shared contextual bindings.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.common.schemas import SubQuery


class DecomposedQueryList(BaseModel):
    """Structured output contract for LLM query decomposition."""

    sub_queries: List[SubQuery] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="1 to 3 orthogonal search queries with preserved shared context",
    )


# Context extraction patterns
LOCATION_EXTRACTOR = re.compile(
    r"\b(?:Pune|Mumbai|Delhi|Bangalore|Bengaluru|Hyderabad|Chennai|Kolkata|London|Paris|Tokyo|New\s+York|Singapore)\b",
    re.IGNORECASE,
)
HEADCOUNT_EXTRACTOR = re.compile(
    r"\b(?:for\s+)?(\d+)\s*(?:people|attendees|participants|guests|pax|persons)\b|"
    r"\bcapacity\s*(?:of|for)?\s*(\d+)\b",
    re.IGNORECASE,
)
EVENT_EXTRACTOR = re.compile(
    r"\b(?:customer\s+workshop|workshop|conference|seminar|meeting|training|event)\b",
    re.IGNORECASE,
)


class MultiIntentDecomposer:
    """Decomposes compound conversational transcripts into orthogonal sub-queries."""

    def __init__(self, llm_callable: Optional[Any] = None) -> None:
        self.llm_callable = llm_callable

    @staticmethod
    def extract_shared_context(transcript: str) -> Dict[str, Any]:
        """Extracts shared environmental bindings (location, headcount, event type)."""
        context: Dict[str, Any] = {}

        loc_match = LOCATION_EXTRACTOR.search(transcript)
        if loc_match:
            context["location"] = loc_match.group(0).strip().capitalize()

        hc_match = HEADCOUNT_EXTRACTOR.search(transcript)
        if hc_match:
            number = hc_match.group(1) or hc_match.group(2)
            if number:
                context["headcount"] = int(number)

        event_match = EVENT_EXTRACTOR.search(transcript)
        if event_match:
            context["event_type"] = event_match.group(0).strip().lower()

        return context

    def _heuristic_decompose(
        self,
        transcript: str,
        shared_context: Dict[str, Any],
    ) -> List[SubQuery]:
        """Deterministic semantic decomposition adhering to anti-overfragmentation rules."""
        cleaned = transcript.replace("[Utterance End]", "").strip()
        loc = shared_context.get("location", "")
        hc = shared_context.get("headcount", "")

        # Candidate intent categories
        intents_found: Dict[str, str] = {}

        # 1. Venue & Capacity intent
        venue_indicators = [
            "venue", "capacity", "room", "facility", "workshop in", "seats", "space", "location"
        ]
        if any(ind in cleaned.lower() for ind in venue_indicators) or (loc and hc):
            q_parts = [p for p in [loc, "venue capacity", f"{hc} people" if hc else ""] if p]
            intents_found["venue_capacity"] = " ".join(q_parts) or "venue capacity and booking"

        # 2. Cancellation & Refund intent
        cancel_indicators = ["cancellation", "cancel", "refund", "penalty", "cancellation policy"]
        if any(ind in cleaned.lower() for ind in cancel_indicators):
            intents_found["cancellation_policy"] = "cancellation refund policy terms and penalties"

        # 3. Catering & Meals intent
        catering_indicators = ["catering", "cater", "meal", "food", "lunch", "dietary", "buffet"]
        if any(ind in cleaned.lower() for ind in catering_indicators):
            q_parts = [p for p in ["catering services food options", f"for {hc} people" if hc else ""] if p]
            intents_found["catering_services"] = " ".join(q_parts)

        # 4. Reimbursement & Expense intent
        expense_indicators = ["reimbursement", "expense", "per diem", "allowance", "lodging"]
        if any(ind in cleaned.lower() for ind in expense_indicators):
            intents_found["reimbursement_limits"] = "travel reimbursement limits and expense claims"

        # Anti-overfragmentation rule:
        # If no specific orthogonal intents matched, treat whole transcript as single intent
        if not intents_found:
            single_q = re.sub(r"[^\w\s]", " ", cleaned).strip()
            return [
                SubQuery(
                    intent_id="intent_01",
                    label="general_query",
                    query=single_q or cleaned,
                    shared_context=shared_context,
                )
            ]

        # Enforce maximum 3 orthogonal queries
        sub_queries: List[SubQuery] = []
        for idx, (label, query_str) in enumerate(list(intents_found.items())[:3], start=1):
            sub_queries.append(
                SubQuery(
                    intent_id=f"intent_{idx:02d}",
                    label=label,
                    query=query_str,
                    shared_context=shared_context,
                )
            )

        return sub_queries

    def decompose(
        self,
        accumulated_transcript: str,
        session_id: str = "default_session",
    ) -> List[SubQuery]:
        """Decomposes transcript into 1-3 orthogonal SubQueries with preserved context."""
        shared_context = self.extract_shared_context(accumulated_transcript)

        # If external structured LLM callable is provided, invoke it
        if self.llm_callable is not None:
            try:
                system_prompt = (
                    "You are the Multi-Intent Decomposer for Streaming Live RAG. "
                    "Decompose compound spoken queries into 1 to 3 orthogonal search queries. "
                    "Anti-overfragmentation rule: Do NOT split queries with overlapping intent. "
                    "Preserve shared context (e.g. location, headcount) across all sub-queries."
                )
                raw_result = self.llm_callable(
                    system_prompt=system_prompt,
                    user_prompt=accumulated_transcript,
                    response_model=DecomposedQueryList,
                )
                if isinstance(raw_result, DecomposedQueryList):
                    # Guarantee max 3 and shared context persistence
                    queries = raw_result.sub_queries[:3]
                    return [
                        SubQuery(
                            intent_id=q.intent_id,
                            label=q.label,
                            query=q.query,
                            shared_context={**shared_context, **q.shared_context},
                        )
                        for q in queries
                    ]
            except Exception:
                # Graceful fallback to deterministic decomposition
                pass

        # Deterministic semantic decomposition
        return self._heuristic_decompose(accumulated_transcript, shared_context)


# Global default decomposer
default_decomposer = MultiIntentDecomposer()


def decompose_accumulated_transcript(
    accumulated_transcript: str,
    session_id: str = "default_session",
) -> List[SubQuery]:
    """Convenience helper to decompose accumulated transcript."""
    return default_decomposer.decompose(accumulated_transcript, session_id=session_id)
