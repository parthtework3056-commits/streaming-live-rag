"""Multi-Intent Decomposer module (Component C2, SRD-SLRAG-001)."""

from app.intent.decomposer import (
    MultiIntentDecomposer,
    decompose_accumulated_transcript,
)

__all__ = [
    "MultiIntentDecomposer",
    "decompose_accumulated_transcript",
]
