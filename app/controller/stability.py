"""Retrieval Controller Stability and Intent Analyzer (Component C1, SRD-SLRAG-001).

Determines whether incoming streaming speech fragments warrant WAIT, RETRIEVE, or SUPPRESS.
Execution target: < 15ms deterministic decision latency.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set, Tuple

from app.common.schemas import ControllerDecisionEvent
from app.controller.rules import evaluate_presentation_rules


# Regex patterns for actionable entities and terms
LOCATION_PATTERN = re.compile(
    r"\b(?:Pune|Mumbai|Delhi|Bangalore|Bengaluru|Hyderabad|Chennai|Kolkata|London|Paris|Tokyo|New\s+York|Singapore|Chicago|San\s+Francisco|Boston)\b",
    re.IGNORECASE,
)
CAPACITY_PATTERN = re.compile(
    r"\b(?:\d+|a\s+dozen|hundred)\s*(?:people|attendees|participants|guests|pax|persons|seats)\b|"
    r"\bcapacity\s*(?:of|for)?\s*\d+\b",
    re.IGNORECASE,
)
DATES_PATTERN = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|\d{4}|"
    r"(?:next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"tomorrow|today)\b",
    re.IGNORECASE,
)
POLICY_TERMS_PATTERN = re.compile(
    r"\b(?:cancellation\s+(?:policy|terms?|clause)|catering\s+options?|reimbursement\s+(?:limits?|policy|claims?)|"
    r"refund\s+(?:policy|terms?|conditions?)|travel\s+policy|expense\s+(?:policy|limits?)|"
    r"lodging\s+allowance|conference\s+room|flight\s+booking|hotel\s+reservation)\b",
    re.IGNORECASE,
)

# Trailing prepositions or conjunctions indicating unfinished thought
TRAILING_HANGERS = re.compile(
    r"\b(?:in|at|and|or|with|to|for|the|a|an|of|that|if|as)\s*(?:\.{2,3})?$",
    re.IGNORECASE,
)


class RetrievalController:
    """Stateful streaming controller evaluating incoming transcript tokens for retrieval gates."""

    def __init__(self, value_threshold: float = 0.65) -> None:
        self.value_threshold = value_threshold
        # Tracks previously retrieved query terms per session to compute novelty
        self.session_retrieved_terms: Dict[str, Set[str]] = {}

    def reset_session(self, session_id: str) -> None:
        """Clears state for a given session."""
        self.session_retrieved_terms.pop(session_id, None)

    def _extract_actionable_entities(self, text: str) -> Dict[str, list]:
        """Extracts recognizable semantic entities and policy terms from text."""
        return {
            "locations": LOCATION_PATTERN.findall(text),
            "capacities": CAPACITY_PATTERN.findall(text),
            "dates": DATES_PATTERN.findall(text),
            "policy_terms": POLICY_TERMS_PATTERN.findall(text),
        }

    def compute_actionability_score(self, text: str) -> Tuple[float, bool]:
        """Computes actionability score in [0.0, 1.0] and returns whether actionable entities exist."""
        entities = self._extract_actionable_entities(text)
        detected_types = sum(1 for items in entities.values() if items)

        if detected_types == 0:
            return 0.0, False

        # Specific high-value policy term + location/capacity is strongly actionable
        has_policy = bool(entities["policy_terms"])
        has_specs = bool(entities["locations"] or entities["capacities"] or entities["dates"])

        if has_policy and has_specs:
            return 1.0, True
        if detected_types >= 2:
            return 0.90, True
        if has_policy:
            return 0.85, True
        if has_specs:
            return 0.75, True

        return 0.50, True

    def compute_stability_score(self, text: str) -> float:
        """Computes structural stability of the utterance."""
        cleaned = text.strip()
        words = cleaned.split()

        # Utterance marked complete
        if cleaned.endswith(("[Utterance End]", ".", "?", "!")):
            return 1.0

        # Trailing incomplete connectors reduce stability
        if TRAILING_HANGERS.search(cleaned):
            base = 0.65
        else:
            base = 0.85

        # Longer word context increases stability
        if len(words) >= 8:
            base = min(1.0, base + 0.15)
        elif len(words) >= 5:
            base = min(1.0, base + 0.05)
        else:
            base = 0.30

        return base

    def compute_novelty_score(self, text: str, session_id: str) -> float:
        """Computes novelty score relative to already retrieved items in this session."""
        words = set(re.findall(r"\b\w{3,}\b", text.lower()))
        previous = self.session_retrieved_terms.get(session_id, set())

        if not previous:
            return 1.0

        new_words = words - previous
        if not words:
            return 0.0

        novelty = len(new_words) / len(words)
        return max(0.1, min(1.0, novelty))

    def evaluate_chunk(
        self,
        chunk_id: str,
        incoming_chunk: str,
        accumulated_text: str,
        timestamp_ms: int,
        session_id: str = "default_session",
    ) -> ControllerDecisionEvent:
        """Evaluates whether to WAIT, RETRIEVE, or SUPPRESS in < 15ms.
        
        Formula: value = 0.4 * actionability + 0.3 * stability + 0.3 * novelty
        Trigger RETRIEVE when value >= threshold and actionable entity identified.
        """
        # 1. Gate G0: Presentation / formatting suppression check
        pres_match = evaluate_presentation_rules(accumulated_text) or evaluate_presentation_rules(incoming_chunk)
        if pres_match is not None:
            decision, reason, eligibility = pres_match
            return ControllerDecisionEvent(
                session_id=session_id,
                timestamp_ms=timestamp_ms,
                chunk_id=chunk_id,
                decision=decision,
                reason=reason,
                eligibility=eligibility,
            )

        words = accumulated_text.strip().split()

        # 2. Gate G1: Minimal semantic length check
        # Words < 5 or fragment lacks semantic noun/verb structure -> "WAIT", reason: "incomplete_intent"
        if len(words) < 5:
            return ControllerDecisionEvent(
                session_id=session_id,
                timestamp_ms=timestamp_ms,
                chunk_id=chunk_id,
                decision="WAIT",
                reason="incomplete_intent",
                eligibility=False,
            )

        # 3. Actionability, Stability, and Novelty computation
        actionability, has_entities = self.compute_actionability_score(accumulated_text)
        stability = self.compute_stability_score(accumulated_text)
        novelty = self.compute_novelty_score(accumulated_text, session_id)

        # Retrieval value score: 0.4*actionability + 0.3*stability + 0.3*novelty
        retrieval_value = (0.4 * actionability) + (0.3 * stability) + (0.3 * novelty)

        # If lacks actionable entities or semantic noun/verb core -> WAIT
        if not has_entities or actionability < 0.4:
            return ControllerDecisionEvent(
                session_id=session_id,
                timestamp_ms=timestamp_ms,
                chunk_id=chunk_id,
                decision="WAIT",
                reason="incomplete_intent",
                eligibility=False,
            )

        # Trigger RETRIEVE when value >= threshold
        if retrieval_value >= self.value_threshold:
            # Register tokens in session history
            if session_id not in self.session_retrieved_terms:
                self.session_retrieved_terms[session_id] = set()
            self.session_retrieved_terms[session_id].update(
                re.findall(r"\b\w{3,}\b", accumulated_text.lower())
            )

            return ControllerDecisionEvent(
                session_id=session_id,
                timestamp_ms=timestamp_ms,
                chunk_id=chunk_id,
                decision="RETRIEVE",
                reason="stable_actionable_intent",
                eligibility=True,
            )

        # Insufficient value score -> WAIT
        return ControllerDecisionEvent(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            chunk_id=chunk_id,
            decision="WAIT",
            reason="incomplete_intent",
            eligibility=False,
        )


# Global default controller instance
default_controller = RetrievalController()


def evaluate_stream_chunk(
    chunk_id: str,
    incoming_chunk: str,
    accumulated_text: str,
    timestamp_ms: int,
    session_id: str = "default_session",
) -> ControllerDecisionEvent:
    """Functional convenience entry point for evaluating a streaming transcript chunk."""
    return default_controller.evaluate_chunk(
        chunk_id=chunk_id,
        incoming_chunk=incoming_chunk,
        accumulated_text=accumulated_text,
        timestamp_ms=timestamp_ms,
        session_id=session_id,
    )
