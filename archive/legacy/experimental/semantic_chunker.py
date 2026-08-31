import re
from typing import List


class SemanticChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        chunks = []
        index = 0
        while index < len(cleaned):
            end = min(index + self.chunk_size, len(cleaned))
            chunk = cleaned[index:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(cleaned):
                break
            index = max(index + self.chunk_size - self.overlap, end - self.chunk_size)
        return chunks


def get_semantic_chunker() -> SemanticChunker:
    return SemanticChunker()


__all__ = ["SemanticChunker", "get_semantic_chunker"]
