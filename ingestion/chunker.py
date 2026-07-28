import re
import logging
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger("crossmind.chunker")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE or DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP or DEFAULT_CHUNK_OVERLAP

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return []
        chunks = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        chunk_index = 0
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "word_count": len(chunk_text.split()),
                    "metadata": dict(metadata or {}),
                })
                chunk_index += 1
                overlap_sentences = self._overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = list(overlap_sentences)
                current_length = sum(len(s.split()) for s in current_chunk_sentences)
            current_chunk_sentences.append(sentence)
            current_length += sentence_length
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "word_count": len(chunk_text.split()),
                "metadata": dict(metadata or {}),
            })
        logger.debug(f"Chunked document into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _overlap_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences or self.chunk_overlap <= 0:
            return []
        total_words = sum(len(s.split()) for s in sentences)
        if total_words <= self.chunk_overlap:
            return sentences
        target = total_words - self.chunk_overlap
        running = 0
        cut_index = 0
        for i, s in enumerate(sentences):
            running += len(s.split())
            if running >= target:
                cut_index = i
                break
        return sentences[max(0, cut_index):]

_chunker_instance = None

def get_chunker() -> DocumentChunker:
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = DocumentChunker()
    return _chunker_instance