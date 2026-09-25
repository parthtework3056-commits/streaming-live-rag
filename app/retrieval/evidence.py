from dataclasses import dataclass
from typing import List


@dataclass
class EvidenceItem:
    chunk_id: str
    doc_id: str
    page: int
    score: float
    text: str


@dataclass
class EvidencePack:
    query: str
    items: List[EvidenceItem]