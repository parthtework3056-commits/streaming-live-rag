"""Synthesis and Grounding Verification module (Component C8, SRD-SLRAG-001)."""

from app.synthesis.verifier import (
    GroundingVerifier,
    VerificationResult,
    default_verifier,
)

__all__ = [
    "GroundingVerifier",
    "VerificationResult",
    "default_verifier",
]
