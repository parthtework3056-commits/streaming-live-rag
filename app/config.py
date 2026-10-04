"""Configuration module for Streaming Live RAG engine (Theme 04, Samsung PRISM Hackathon)."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
CORPUS_RAW_DIR = CORPUS_DIR / "raw"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


class Settings(BaseSettings):
    """Core runtime configuration settings adhering to SRD-SLRAG-001."""

    model_config = SettingsConfigDict(env_prefix="SLRAG_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    # Directory Paths
    base_dir: Path = BASE_DIR
    corpus_dir: Path = CORPUS_DIR
    corpus_raw_dir: Path = CORPUS_RAW_DIR
    manifest_path: Path = MANIFEST_PATH

    # Chunking Hyperparameters
    chunk_min_tokens: int = Field(default=300, description="Minimum tokens per chunk")
    chunk_max_tokens: int = Field(default=500, description="Maximum tokens per chunk")
    chunk_overlap_tokens: int = Field(default=50, description="Token overlap between adjacent chunks")

    # Retrieval Models & Isolated Embeddings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local dense embedding model",
    )
    fallback_embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Secondary supported local dense embedding model",
    )

    # Retrieval Defaults
    default_top_k: int = Field(default=5, description="Default retrieval top-k count")

    # Security & Corpus Isolation (Strict requirement: no external web fetching)
    corpus_isolation: bool = Field(
        default=True,
        description="Corpus isolation flag strictly disallowing external web requests",
    )

    # -------------------------------------------------------------------------
    # STT (Speech-to-Text) Input Layer Configuration
    # These settings control the upstream audio→transcript pipeline.
    # They do NOT affect the retrieval / synthesis pipeline.
    # -------------------------------------------------------------------------

    input_mode: str = Field(
        default="mock",
        description=(
            "Transcript input mode. "
            "'mock' = deterministic benchmark replay (no API key needed). "
            "'stt'  = real microphone + STT API (requires STT_API_KEY)."
        ),
    )

    stt_provider: str = Field(
        default="assemblyai",
        description=(
            "STT backend to use when input_mode='stt'. "
            "Supported: 'assemblyai'. "
            "Future: 'google', 'deepgram', 'whisper_local'."
        ),
    )

    stt_api_key: Optional[str] = Field(
        default=None,
        description="API key for the configured STT provider. Keep server-side only.",
    )

    stt_endpoint: Optional[str] = Field(
        default=None,
        description="Custom STT WebSocket/REST endpoint URL (leave blank for provider default).",
    )

    stt_model: Optional[str] = Field(
        default=None,
        description="STT model identifier (provider-specific, e.g. 'nano' for AssemblyAI).",
    )

    stt_language: str = Field(
        default="en",
        description="BCP-47 language code for STT transcription (e.g. 'en', 'hi', 'en-IN').",
    )

    stt_mock_replay_delay_ms: float = Field(
        default=100.0,
        description="Artificial inter-event delay for mock STT provider (ms). Set 0 for synchronous tests.",
    )


settings = Settings()
