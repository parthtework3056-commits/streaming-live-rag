from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    section: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)