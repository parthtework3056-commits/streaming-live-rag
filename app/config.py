"""Configuration module for Streaming Live RAG engine (Theme 04, Samsung PRISM Hackathon)."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
CORPUS_RAW_DIR = CORPUS_DIR / "raw"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


class Settings(BaseSettings):
    """Core runtime configuration settings adhering to SRD-SLRAG-001."""

    model_config = SettingsConfigDict(env_prefix="SLRAG_", extra="ignore")

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


settings = Settings()
