"""Base abstraction for STT (Speech-to-Text) providers.

Defines the normalized TranscriptEvent schema and the abstract STTProvider
interface.  The rest of the RAG system consumes only TranscriptEvent objects
and never depends on a concrete provider implementation.

Design note:
  TranscriptEvent is intentionally aligned with the existing ChunkIngestRequest
  schema (text, offset_seconds, is_last) so it can be forwarded directly to
  POST /api/session/{session_id}/chunk without any further translation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel, Field


class TranscriptEvent(BaseModel):
    """Normalized transcript event emitted by any STT provider.

    is_final=False  → partial / incremental transcript (still evolving).
    is_final=True   → utterance finalized; downstream pipeline should treat
                       this as the definitive text for the chunk.

    Fields align with the existing ChunkIngestRequest so that the adapter
    can forward events directly to the /chunk endpoint.
    """

    session_id: str = Field(..., description="Active session identifier")
    event_type: str = Field(
        ...,
        description="'partial_transcript' | 'final_transcript' | 'stt_error' | 'stt_status'",
    )
    text: str = Field(default="", description="Current transcript text (cumulative from provider)")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Epoch timestamp in milliseconds",
    )
    sequence_id: int = Field(default=0, ge=0, description="Monotonically increasing sequence counter")
    is_final: bool = Field(
        default=False,
        description="True when the provider has finalized this utterance segment",
    )
    # offset_seconds mirrors ChunkIngestRequest.offset_seconds
    offset_seconds: float = Field(
        default=0.0,
        description="Approximate speech offset in seconds from session start",
    )
    # Optional provider-level detail
    provider: str = Field(default="unknown", description="STT provider identifier")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Provider confidence score (0.0–1.0) when available",
    )
    error_detail: Optional[str] = Field(
        default=None,
        description="Human-readable error description when event_type is 'stt_error'",
    )


class STTProvider(ABC):
    """Abstract interface that every STT backend must implement.

    Concrete implementations:
      - MockSTTProvider  (INPUT_MODE=mock — deterministic, no API required)
      - AssemblyAISTTProvider  (INPUT_MODE=stt, STT_PROVIDER=assemblyai)
      - GoogleSTTProvider      (INPUT_MODE=stt, STT_PROVIDER=google)
      - DeepgramSTTProvider    (INPUT_MODE=stt, STT_PROVIDER=deepgram)

    The RAG pipeline must NEVER import a concrete provider class directly.
    Use create_stt_provider() from app.stt.factory instead.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns a stable provider identifier string (e.g. 'assemblyai')."""

    @abstractmethod
    async def stream_audio(
        self,
        audio_source,  # file-like / AsyncIterator[bytes] / None for mic
        session_id: str,
        language: str = "en",
    ) -> AsyncIterator[TranscriptEvent]:
        """Streams audio and yields normalized TranscriptEvent objects.

        Must yield partial events as they arrive and a final event when the
        utterance is complete.  Must handle all provider-level errors and
        yield a TranscriptEvent(event_type='stt_error', ...) instead of
        raising — callers rely on this contract for graceful degradation.
        """
        # Make this an async generator so ABC subclasses don't need to repeat
        # the yield statement in the abstract method body.
        if False:  # pragma: no cover
            yield  # type: ignore[misc]

    async def close(self) -> None:
        """Optional cleanup / connection teardown.  Subclasses override as needed."""
