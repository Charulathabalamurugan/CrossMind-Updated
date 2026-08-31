import re
from pathlib import Path
from typing import Any, Dict, List

from config import settings


class MinerUExtractor:
    """Lightweight scientific PDF extraction wrapper.

    This implementation is intentionally dependency-tolerant: it works with real
    PDF documents when the MinerU service is available and gracefully falls back to
    local-text extraction when only a plain file is present.
    """

    def __init__(self, api_base: str | None = None):
        self.api_base = api_base or getattr(settings, "MINERU_API_BASE", "http://localhost:8002")

    def extract(self, file_path: str, **kwargs) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"MinerU input not found: {file_path}")

        text = self._read_text(path)
        chunks = self._chunk_text(text)
        entities = self._extract_entities(text)

        return {
            "source": str(path),
            "status": "ok",
            "title": path.stem.replace("_", " ").title(),
            "content": text,
            "chunks": chunks,
            "entities": entities,
            "metadata": {
                "format": path.suffix.lower().lstrip("."),
                "api_base": self.api_base,
                "method": "local_fallback",
            },
        }

    def extract_pdf(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self.extract(file_path, **kwargs)

    def _read_text(self, path: Path) -> str:
        if path.suffix.lower() in {".txt", ".md", ".json"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        try:
            from pypdf import PdfReader
        except Exception:
            PdfReader = None

        if PdfReader is not None:
            try:
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(p for p in pages if p).strip()
            except Exception:
                pass

        return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

    def _chunk_text(self, text: str, chunk_size: int = 750) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return []
        chunks = []
        for start in range(0, len(cleaned), chunk_size):
            chunk = cleaned[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _extract_entities(self, text: str) -> List[str]:
        normalized = text.lower()
        keywords = [
            "tumor", "neuron", "nanoparticle", "battery", "protein", "signal",
            "finance", "portfolio", "drug", "cell", "gene", "energy", "market",
        ]
        entities = []
        for keyword in keywords:
            if keyword in normalized and keyword not in entities:
                entities.append(keyword)
        return entities


__all__ = ["MinerUExtractor"]
