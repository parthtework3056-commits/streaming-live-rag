"""Deterministic rule engine for presentation, formatting, and restructuring suppression.

Adheres to SRD-SLRAG-001 (Component C1 - Retrieval Controller).
Lightweight heuristic evaluation executing in < 1ms.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


# Regex patterns matching presentation, restructuring, and repetition requests
PRESENTATION_PATTERNS = [
    # E.g. "turn the last answer into 3 bullets", "format in bullet points", "repeat in bullets"
    re.compile(
        r"\b(?:turn|convert|put|format|present|restructure|make|transform|give|rewrite|rephrase|repeat)\b.*?"
        r"\b(?:bullets?|bullet\s+points?|points?|numbered\s+list|table|summary|concise|shorter)\b",
        re.IGNORECASE,
    ),
    # E.g. "summarize in 2 points", "summarize into 3 bullets"
    re.compile(
        r"\b(?:summarize|summary|condense|abstract)\b.*?\b(?:\d+\s+(?:points?|bullets?|lines?)|briefly|in\s+bullets?)\b",
        re.IGNORECASE,
    ),
    # E.g. "translate to Spanish", "translate into French"
    re.compile(
        r"\b(?:translate|translation)\b.*?\b(?:into|to|in)\b\s+[a-zA-Z]+",
        re.IGNORECASE,
    ),
    # Direct bullet point transformations
    re.compile(r"\b(?:in|into|as)\s+(?:\d+\s+)?(?:bullets?|bullet\s+points?)\b", re.IGNORECASE),
    # Simple repeat / restate queries without new factual information
    re.compile(
        r"^(?:can\s+you\s+)?(?:repeat|restate|say\s+that\s+again|read\s+back)(?:\s+that|\s+the\s+last\s+answer|\s+again)?[\?\.]?$",
        re.IGNORECASE,
    ),
]


def evaluate_presentation_rules(text: str) -> Optional[Tuple[str, str, bool]]:
    """Evaluates whether an incoming utterance matches presentation or formatting restructuring.
    
    Returns:
        (decision, reason, eligibility) if matched, else None.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    for pattern in PRESENTATION_PATTERNS:
        if pattern.search(cleaned):
            return ("SUPPRESS", "presentation_restructure", False)

    return None
