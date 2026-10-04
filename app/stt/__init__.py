"""STT (Speech-to-Text) adapter package for Streaming Live RAG.

Provides a clean abstraction layer between any STT provider and the existing
transcript event interface used by the Retrieval Controller.

Input layer only — never touches retrieval or synthesis logic directly.
"""

from app.stt.base import STTProvider, TranscriptEvent
from app.stt.factory import create_stt_provider

__all__ = ["STTProvider", "TranscriptEvent", "create_stt_provider"]
