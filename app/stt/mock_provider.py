"""Mock STT provider for benchmark, testing, and demo modes.

INPUT_MODE=mock (or when no API credentials are configured).

Replays the canonical WORKSHOP_PLANNING_TRANSCRIPT sequence from the existing
benchmark harness, yielding partial → final transcript events deterministically.
This preserves 100% of the existing benchmark reproducibility contract.

No external API calls, no microphone access, no credentials required.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, List, Optional

from app.stt.base import STTProvider, TranscriptEvent


# Canonical benchmark phrases — mirrors WORKSHOP_PLANNING_TRANSCRIPT in
# benchmarks/stream_simulator.py so mock mode is deterministic.
_DEFAULT_MOCK_PHRASES = [
    (0.0, "I need to plan a customer workshop in..."),
    (0.8, "...Pune for 30 people, and I need..."),
    (1.6, "...the cancellation policy and the catering options."),
    (2.1, "[Utterance End]"),
]


class MockSTTProvider(STTProvider):
    """Deterministic STT provider for testing and demo use (no API required).

    When audio_source is None the provider replays _DEFAULT_MOCK_PHRASES.
    When audio_source is a list of (offset_seconds, text) tuples it replays
    those instead — useful for scenario-specific test fixtures.

    Emits one partial_transcript per phrase, then a final_transcript for the
    completed utterance.
    """

    def __init__(self, replay_delay_ms: float = 100.0) -> None:
        """
        Args:
            replay_delay_ms: Artificial delay between emitted events (ms).
                             Set to 0 for synchronous/unit-test mode.
        """
        self._replay_delay_ms = replay_delay_ms

    @property
    def provider_name(self) -> str:
        return "mock"

    async def stream_audio(
        self,
        audio_source,
        session_id: str,
        language: str = "en",
    ) -> AsyncIterator[TranscriptEvent]:
        """Replays mock transcript as partial events followed by a final event."""
        phrases: List[tuple] = (
            audio_source if isinstance(audio_source, list) else _DEFAULT_MOCK_PHRASES
        )

        accumulated = ""
        for seq_id, (offset_sec, text) in enumerate(phrases):
            # Accumulate exactly as the existing stream accumulator does
            is_utterance_end = text.strip() == "[Utterance End]"

            if not is_utterance_end:
                # Strip leading ellipsis (mirrors clean_accumulate in stream_simulator)
                import re
                stripped = re.sub(r"^\.{2,3}\s*", "", text.strip())
                prev_clean = re.sub(r"\s*\.{2,3}$", "", accumulated.strip())
                accumulated = f"{prev_clean} {stripped}".strip() if prev_clean else stripped

            # Partial transcript for every phrase except the final marker
            event_type = "final_transcript" if is_utterance_end else "partial_transcript"
            is_final = is_utterance_end

            yield TranscriptEvent(
                session_id=session_id,
                event_type=event_type,
                text=accumulated if not is_utterance_end else accumulated,
                timestamp_ms=int(time.time() * 1000),
                sequence_id=seq_id,
                is_final=is_final,
                offset_seconds=offset_sec,
                provider=self.provider_name,
                confidence=1.0,
            )

            if self._replay_delay_ms > 0:
                await asyncio.sleep(self._replay_delay_ms / 1000.0)
