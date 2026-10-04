"""AssemblyAI real-time streaming STT provider for Streaming Live RAG.

Activated when:
    STT_PROVIDER=assemblyai
    STT_API_KEY=<your_key>
    INPUT_MODE=stt

AssemblyAI's real-time WebSocket API emits partial transcripts continuously
while the user speaks, then emits a final transcript when silence is detected.

This adapter normalizes AssemblyAI events into TranscriptEvent objects so the
rest of the RAG pipeline never depends on AssemblyAI-specific schemas.

Reference: https://www.assemblyai.com/docs/speech-to-text/streaming

Partial / final handling:
  AssemblyAI sends:
    { "message_type": "PartialTranscript", "text": "...", "audio_start": ..., ... }
    { "message_type": "FinalTranscript",   "text": "...", "audio_start": ..., ... }

  PartialTranscript text is CUMULATIVE within a session turn — we emit it
  directly without appending to avoid double-accumulation.
  FinalTranscript signals utterance completion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

from app.stt.base import STTProvider, TranscriptEvent

logger = logging.getLogger(__name__)

# AssemblyAI real-time streaming endpoint
_ASSEMBLYAI_WS_ENDPOINT = "wss://api.assemblyai.com/v2/realtime/ws"
_SAMPLE_RATE = 16000  # Hz — standard for STT APIs


class AssemblyAISTTProvider(STTProvider):
    """Real-time streaming STT via AssemblyAI WebSocket API.

    Requires:
        pip install websockets  (already a transitive dep of many async libs)

    Audio format expected by AssemblyAI: PCM 16-bit LE, 16 kHz mono.
    The browser's MediaRecorder / AudioWorklet should capture in this format.
    Audio chunks are forwarded as base64-encoded JSON payloads over WebSocket.
    """

    def __init__(self, api_key: str, sample_rate: int = _SAMPLE_RATE, endpoint: Optional[str] = None) -> None:
        if not api_key:
            raise ValueError(
                "AssemblyAI STT_API_KEY is required. "
                "Set STT_API_KEY in your .env file or environment."
            )
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._endpoint = endpoint or _ASSEMBLYAI_WS_ENDPOINT

    @property
    def provider_name(self) -> str:
        return "assemblyai"

    async def stream_audio(
        self,
        audio_source: AsyncIterator[bytes],
        session_id: str,
        language: str = "en",
    ) -> AsyncIterator[TranscriptEvent]:
        """Streams raw PCM audio chunks to AssemblyAI and yields transcript events.

        audio_source: async iterator producing raw PCM bytes (e.g. from microphone
                      captured by browser and forwarded over WebSocket to backend).
        """
        try:
            import websockets  # type: ignore
        except ImportError:
            yield TranscriptEvent(
                session_id=session_id,
                event_type="stt_error",
                text="",
                timestamp_ms=int(time.time() * 1000),
                sequence_id=0,
                is_final=True,
                provider=self.provider_name,
                error_detail=(
                    "websockets package not installed. "
                    "Run: pip install websockets"
                ),
            )
            return

        import base64
        ws_url = (
            f"{self._endpoint}"
            f"?sample_rate={self._sample_rate}"
        )
        headers = {"Authorization": self._api_key}

        sequence_id = 0
        start_time = time.time()

        # Track the last partial text to detect meaningful changes and avoid duplicates
        last_partial_text: str = ""

        try:
            async with websockets.connect(ws_url, additional_headers=headers) as ws:
                logger.info(
                    "stt_connected provider=assemblyai session=%s",
                    session_id,
                )

                # Yield a status event so the frontend can show "listening"
                yield TranscriptEvent(
                    session_id=session_id,
                    event_type="stt_status",
                    text="",
                    timestamp_ms=int(time.time() * 1000),
                    sequence_id=sequence_id,
                    is_final=False,
                    provider=self.provider_name,
                )

                async def _send_audio():
                    """Coroutine that reads audio chunks and sends them to AssemblyAI."""
                    try:
                        async for audio_chunk in audio_source:
                            if not audio_chunk:
                                continue
                            payload = json.dumps(
                                {"audio_data": base64.b64encode(audio_chunk).decode("utf-8")}
                            )
                            await ws.send(payload)
                    except Exception as exc:
                        logger.warning("stt_audio_send_error session=%s error=%s", session_id, exc)
                    finally:
                        # Signal end-of-stream to AssemblyAI
                        try:
                            await ws.send(json.dumps({"terminate_session": True}))
                        except Exception:
                            pass

                # Run audio sender concurrently with response reader
                send_task = asyncio.create_task(_send_audio())

                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("message_type", "")
                    text = msg.get("text", "").strip()
                    audio_start_ms = msg.get("audio_start", 0)
                    offset_sec = audio_start_ms / 1000.0

                    if msg_type == "SessionBegins":
                        logger.info(
                            "stt_session_started provider=assemblyai session=%s",
                            session_id,
                        )
                        continue

                    if msg_type == "PartialTranscript":
                        # AssemblyAI partial transcripts are cumulative within a segment.
                        # Only emit when text has meaningfully changed to avoid flooding
                        # the controller with no-op updates.
                        if text and text != last_partial_text:
                            last_partial_text = text
                            sequence_id += 1
                            yield TranscriptEvent(
                                session_id=session_id,
                                event_type="partial_transcript",
                                text=text,
                                timestamp_ms=int(time.time() * 1000),
                                sequence_id=sequence_id,
                                is_final=False,
                                offset_seconds=offset_sec,
                                provider=self.provider_name,
                                confidence=msg.get("confidence"),
                            )

                    elif msg_type == "FinalTranscript":
                        if text:
                            sequence_id += 1
                            last_partial_text = ""  # Reset for next utterance
                            yield TranscriptEvent(
                                session_id=session_id,
                                event_type="final_transcript",
                                text=text,
                                timestamp_ms=int(time.time() * 1000),
                                sequence_id=sequence_id,
                                is_final=True,
                                offset_seconds=offset_sec,
                                provider=self.provider_name,
                                confidence=msg.get("confidence"),
                            )

                    elif msg_type == "SessionTerminated":
                        logger.info(
                            "stt_disconnected provider=assemblyai session=%s duration_s=%.1f",
                            session_id,
                            time.time() - start_time,
                        )
                        break

                    elif msg_type == "Error":
                        error_msg = msg.get("error", "Unknown AssemblyAI error")
                        logger.error(
                            "stt_error provider=assemblyai session=%s error=%s",
                            session_id,
                            error_msg,
                        )
                        yield TranscriptEvent(
                            session_id=session_id,
                            event_type="stt_error",
                            text="",
                            timestamp_ms=int(time.time() * 1000),
                            sequence_id=sequence_id,
                            is_final=True,
                            provider=self.provider_name,
                            error_detail=error_msg,
                        )
                        break

                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        except OSError as exc:
            logger.error(
                "stt_connection_failed provider=assemblyai session=%s error=%s",
                session_id,
                exc,
            )
            yield TranscriptEvent(
                session_id=session_id,
                event_type="stt_error",
                text="",
                timestamp_ms=int(time.time() * 1000),
                sequence_id=sequence_id,
                is_final=True,
                provider=self.provider_name,
                error_detail=f"Connection failed: {exc}",
            )
        except Exception as exc:
            logger.error(
                "stt_unexpected_error provider=assemblyai session=%s error=%s",
                session_id,
                exc,
            )
            yield TranscriptEvent(
                session_id=session_id,
                event_type="stt_error",
                text="",
                timestamp_ms=int(time.time() * 1000),
                sequence_id=sequence_id,
                is_final=True,
                provider=self.provider_name,
                error_detail=f"Unexpected error: {exc}",
            )

    async def close(self) -> None:
        """No persistent connections to tear down for AssemblyAI (per-stream WS)."""
        logger.debug("AssemblyAISTTProvider.close() called")
