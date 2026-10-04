"""Ephemeral Session State Store (Component C5, SRD-SLRAG-001).

Maintains strict session-bound ephemeral state with zero cross-session user profiling.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimItem(BaseModel):
    """Atomic factual statement synthesized in an answer version."""

    claim_id: str = Field(..., description="Unique claim identifier within session")
    statement: str = Field(..., description="Synthesized claim sentence or assertion")
    status: str = Field(
        default="VERIFIED",
        description="Claim verification state: VERIFIED, NEEDS_RECHECK, or MUTATED",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Chunk IDs supporting this atomic claim",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Strictly session-bound conversational state adhering to SRD-SLRAG-001."""

    session_id: str = Field(..., description="Unique ephemeral session identifier")
    transcript_version: int = Field(default=1, ge=1)
    answer_version: int = Field(default=1, ge=1)
    active_intents: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    claims: List[ClaimItem] = Field(default_factory=list)
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    last_answer: Optional[str] = Field(default=None)
    g2_eligible: bool = Field(default=False)
    g2_retrieval_start: Optional[float] = Field(default=None)
    g2_utterance_end: Optional[float] = Field(default=None)
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


class SessionStateStore:
    """Ephemeral, thread-safe session memory store indexed by session_id.
    
    Guarantees strict session boundary isolation without cross-session profiling.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        """Retrieves session state if existing."""
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> SessionState:
        """Retrieves or initializes a new ephemeral session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def save(self, state: SessionState) -> None:
        """Persists session state in ephemeral memory."""
        state.updated_at_ms = int(time.time() * 1000)
        self._sessions[state.session_id] = state

    def clear(self, session_id: str) -> None:
        """Wipes all state for a given session."""
        self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        """Wipes entire memory store across all sessions."""
        self._sessions.clear()


# Default singleton instance
default_session_store = SessionStateStore()
