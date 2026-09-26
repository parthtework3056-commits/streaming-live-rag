"""Section-aware chunker producing deterministic ChunkRecords and stable manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.common.schemas import ChunkRecord
from app.config import settings


# Regex patterns for detecting section headers and section symbol tags (§)
SECTION_PATTERNS = [
    # Explicit section symbol (e.g. "§ 1.2 Reimbursement Limits", "§10.4 Cancellation Terms")
    re.compile(r"^(§\s*\d+(?:\.\d+)*\s*[-:—]?\s*[^\n]+)", re.MULTILINE),
    # Markdown headers (e.g. "# Section 1", "## Reimbursement Policy")
    re.compile(r"^(#{1,6}\s+[^\n]+)", re.MULTILINE),
    # Explicit "Section X", "Article X", "Policy X"
    re.compile(r"^((?:SECTION|Section|ARTICLE|Article|POLICY|Policy)\s+\d+(?:[\.:]\d+)*\s*[-:—]?\s*[^\n]+)", re.MULTILINE),
    # Capitalized headings followed by newline (standalone header lines)
    re.compile(r"^([A-Z0-9][A-Z0-9\s,\-:]{3,60})$", re.MULTILINE),
]


class SectionAwareChunker:
    """Splits documents into deterministic, section-aware ChunkRecords."""

    def __init__(
        self,
        min_tokens: int = settings.chunk_min_tokens,
        max_tokens: int = settings.chunk_max_tokens,
        overlap_tokens: int = settings.chunk_overlap_tokens,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        """Deterministic word-level tokenization."""
        return text.split()

    @classmethod
    def _detect_sections(cls, text: str) -> List[Tuple[str, str]]:
        """Splits document text into a sequence of (section_title, section_body) tuples.
        
        Preserves section titles and symbol tags (§).
        """
        # Find all header match positions across the text
        matches: List[Tuple[int, int, str]] = []
        for pattern in SECTION_PATTERNS:
            for match in pattern.finditer(text):
                header = match.group(1).strip()
                matches.append((match.start(), match.end(), header))

        # Sort matches by start position in text
        matches.sort(key=lambda m: m[0])

        # Filter overlapping or nested matches
        filtered_matches: List[Tuple[int, int, str]] = []
        last_end = -1
        for start, end, header in matches:
            if start >= last_end:
                filtered_matches.append((start, end, header))
                last_end = end

        # If no explicit sections detected, treat entire text as one section
        if not filtered_matches:
            return [("General", text.strip())]

        sections: List[Tuple[str, str]] = []

        # Content before the first header, if any
        if filtered_matches[0][0] > 0:
            preamble = text[: filtered_matches[0][0]].strip()
            if preamble:
                sections.append(("Preamble", preamble))

        for i, (start, end, header) in enumerate(filtered_matches):
            next_start = filtered_matches[i + 1][0] if i + 1 < len(filtered_matches) else len(text)
            body = text[end:next_start].strip()
            # Include the header text in the body context if needed
            full_section_text = f"{header}\n{body}".strip() if body else header
            sections.append((header, full_section_text))

        return sections

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        page: int = 1,
        start_chunk_idx: int = 1,
    ) -> List[ChunkRecord]:
        """Produces deterministic ChunkRecords for a document string."""
        sections = self._detect_sections(text)
        chunk_records: List[ChunkRecord] = []
        current_chunk_idx = start_chunk_idx

        for section_title, section_text in sections:
            words = self._tokenize_words(section_text)
            if not words:
                continue

            # If section fits within max_tokens, emit single chunk
            if len(words) <= self.max_tokens:
                chunk_id = f"DOC_{doc_id}_CHUNK_{current_chunk_idx:03d}"
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        page=page,
                        section=section_title,
                        text=" ".join(words),
                        metadata={
                            "token_count": len(words),
                            "char_count": len(section_text),
                            "chunk_index": current_chunk_idx,
                            "is_split": False,
                        },
                    )
                )
                current_chunk_idx += 1
            else:
                # Sliding window chunking with specified overlap
                step = max(1, self.max_tokens - self.overlap_tokens)
                for start_idx in range(0, len(words), step):
                    window = words[start_idx : start_idx + self.max_tokens]
                    if not window:
                        continue

                    # If remaining tail is tiny and not the first chunk, append to previous
                    if len(window) < self.overlap_tokens and chunk_records:
                        continue

                    chunk_id = f"DOC_{doc_id}_CHUNK_{current_chunk_idx:03d}"
                    chunk_records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            page=page,
                            section=section_title,
                            text=" ".join(window),
                            metadata={
                                "token_count": len(window),
                                "char_count": len(" ".join(window)),
                                "chunk_index": current_chunk_idx,
                                "is_split": True,
                                "window_start": start_idx,
                            },
                        )
                    )
                    current_chunk_idx += 1

                    if start_idx + self.max_tokens >= len(words):
                        break

        return chunk_records


def chunk_document(
    doc_path: Path | str,
    doc_id: Optional[str] = None,
    chunker: Optional[SectionAwareChunker] = None,
) -> List[ChunkRecord]:
    """Loads and chunks a raw text document deterministically."""
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    assigned_doc_id = doc_id or path.stem.upper()
    text = path.read_text(encoding="utf-8")

    active_chunker = chunker or SectionAwareChunker()
    return active_chunker.chunk_text(text=text, doc_id=assigned_doc_id)


def generate_corpus_manifest(
    chunks: List[ChunkRecord],
    manifest_path: Path | str = settings.manifest_path,
) -> Path:
    """Emits a stable, sorted manifest JSON file for all indexed chunks."""
    output_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group chunk IDs by document
    docs_summary: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        if c.doc_id not in docs_summary:
            docs_summary[c.doc_id] = {
                "doc_id": c.doc_id,
                "total_chunks": 0,
                "pages": set(),
                "sections": set(),
            }
        docs_summary[c.doc_id]["total_chunks"] += 1
        docs_summary[c.doc_id]["pages"].add(c.page)
        docs_summary[c.doc_id]["sections"].add(c.section)

    serializable_summary = {
        doc_id: {
            "doc_id": data["doc_id"],
            "total_chunks": data["total_chunks"],
            "pages": sorted(list(data["pages"])),
            "sections": sorted(list(data["sections"])),
        }
        for doc_id, data in sorted(docs_summary.items())
    }

    manifest_data = {
        "schema_version": "1.0.0",
        "total_documents": len(serializable_summary),
        "total_chunks": len(chunks),
        "documents": serializable_summary,
        "chunks": [chunk.model_dump() for chunk in sorted(chunks, key=lambda c: c.chunk_id)],
    }

    # Deterministic formatted JSON with sorted keys
    output_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return output_path
