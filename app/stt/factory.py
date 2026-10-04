"""STT provider factory — constructs the correct provider from configuration.

Usage (in app.main or any endpoint):

    from app.stt.factory import create_stt_provider
    provider = create_stt_provider()

The factory reads settings from app.config (which reads from environment /
.env file).  The rest of the application never touches provider constructors
directly, making it trivial to swap providers or add new ones.

Supported INPUT_MODE values:
    mock        → MockSTTProvider (no API key, fully deterministic)
    stt         → Real provider selected by STT_PROVIDER setting

Supported STT_PROVIDER values (when INPUT_MODE=stt):
    assemblyai  → AssemblyAISTTProvider
    (future: google, deepgram, whisper_local, azure)

If INPUT_MODE=stt but credentials are missing or the provider import fails,
the factory logs a warning and falls back to MockSTTProvider so the rest of
the application continues to work.
"""

from __future__ import annotations

import logging

from app.stt.base import STTProvider

logger = logging.getLogger(__name__)


def create_stt_provider() -> STTProvider:
    """Returns an STTProvider instance based on current configuration."""
    # Import here to avoid circular imports at module load time
    from app.config import settings  # noqa: PLC0415

    input_mode = settings.input_mode.lower()

    if input_mode == "mock":
        return _build_mock_provider(settings)

    if input_mode == "stt":
        provider_name = settings.stt_provider.lower()
        try:
            return _build_real_provider(provider_name, settings)
        except Exception as exc:
            logger.warning(
                "stt_provider_init_failed provider=%s error=%s — falling back to mock",
                provider_name,
                exc,
            )
            return _build_mock_provider(settings)

    # Unknown mode — default to mock to preserve existing demo behaviour
    logger.warning(
        "Unknown INPUT_MODE='%s'. Defaulting to mock STT provider.",
        input_mode,
    )
    return _build_mock_provider(settings)


def _build_mock_provider(settings) -> STTProvider:
    from app.stt.mock_provider import MockSTTProvider  # noqa: PLC0415
    logger.info("stt_provider=mock (INPUT_MODE=%s)", settings.input_mode)
    return MockSTTProvider(replay_delay_ms=settings.stt_mock_replay_delay_ms)


def _build_real_provider(provider_name: str, settings) -> STTProvider:
    if provider_name == "assemblyai":
        from app.stt.assemblyai_provider import AssemblyAISTTProvider  # noqa: PLC0415
        if not settings.stt_api_key:
            raise ValueError(
                "STT_PROVIDER=assemblyai requires STT_API_KEY to be set. "
                "Add STT_API_KEY=<your_key> to your .env file."
            )
        logger.info(
            "stt_provider=assemblyai endpoint=%s model=%s lang=%s",
            settings.stt_endpoint or _ASSEMBLYAI_DEFAULT_ENDPOINT,
            settings.stt_model or "default",
            settings.stt_language,
        )
        return AssemblyAISTTProvider(
            api_key=settings.stt_api_key,
            endpoint=settings.stt_endpoint,
        )

    raise ValueError(
        f"Unsupported STT_PROVIDER='{provider_name}'. "
        "Supported: assemblyai. "
        "Set INPUT_MODE=mock to use without an API key."
    )


_ASSEMBLYAI_DEFAULT_ENDPOINT = "wss://api.assemblyai.com/v2/realtime/ws"
