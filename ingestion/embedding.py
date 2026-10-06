import numpy as np
import logging
from typing import Any, Dict, List, Optional
from config import settings
from vector_store.vector_adapter import get_vector_adapter

logger = logging.getLogger("crossmind.embedding")


class BGE3NeuralEmbedder:
    """Neural BGE-M3 (BAAI/bge-m3) dense embedder.

    Loaded with ``local_files_only=True`` so it never performs a network fetch
    at import/instantiation time: if the weights are not present in the local
    HuggingFace cache the instance is not created and the caller transparently
    falls back to the deterministic DSKE :class:`Embedder`.
    """

    def __init__(self, model: "Any") -> None:
        self.model_name = settings.BGE_M3_MODEL_NAME
        self.model = model
        self.adapter = get_vector_adapter()
        max_len = getattr(settings, "BGE_M3_MAX_LENGTH", 8192) or 8192
        try:
            self.model.max_seq_length = int(max_len)
        except Exception:
            pass
        if settings.BGE_M3_MATRYOSHKA_ENABLED:
            self.dim = settings.BGE_M3_RETRIEVAL_DIM
        else:
            self.dim = settings.BGE_M3_DIM
        logger.info(
            f"Initialized neural BGE-M3 embedder ({self.model_name}) "
            f"with dimension {self.dim}"
        )

    def embed_texts(
        self, texts: List[str], dim: int = None
    ) -> List[List[float]]:
        effective_dim = dim if dim is not None else self.dim
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        results: List[List[float]] = []
        for vec in embs:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] > effective_dim:
                arr = arr[:effective_dim]
            elif arr.shape[0] < effective_dim:
                padded = np.zeros(effective_dim, dtype=np.float32)
                padded[: arr.shape[0]] = arr
                arr = padded
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            results.append(arr.tolist())
        return results

    def embed_text(self, text: str, dim: int = None) -> List[float]:
        return self.embed_texts([text], dim=dim)[0]

    def embed_and_normalize(
        self, text: str, dim: int = None
    ) -> Dict[str, Any]:
        effective_dim = dim if dim is not None else self.dim
        raw = self.embed_text(text, dim=effective_dim)
        return self.adapter.normalize(raw, force_dim=effective_dim)

    def normalize_vector(
        self, vector: Any, force_dim: int = None
    ) -> Dict[str, Any]:
        return self.adapter.normalize(vector, force_dim=force_dim or self.dim)

    def reshape_vector(self, flat_vector: List[float], meta: Dict[str, Any]):
        return self.adapter.reshape(flat_vector, meta)


class Embedder:
    """
    DSKE (Document-Symbolic Knowledge Embedding) Engine.

    Generates high-performance, deterministic symbolic-hashing vectors as a
    fast, zero-dependency fallback used when BGE-M3 is unavailable or disabled.
    """

    def __init__(
        self, model_name: str = "DSKE", dim: int = settings.EMBEDDING_DIM
    ):
        self.model_name = model_name
        self.dim = dim
        self.adapter = get_vector_adapter()
        logger.info(
            f"Initialized DSKE Embedding Engine ({self.model_name}) "
            f"with dimension {self.dim}"
        )

    def embed_texts(
        self, texts: List[str], dim: int = None
    ) -> List[List[float]]:
        return [
            self._deterministic_vector(text, dim if dim is not None else self.dim)
            for text in texts
        ]

    def embed_text(self, text: str, dim: int = None) -> List[float]:
        return self.embed_texts([text], dim=dim)[0]

    def embed_and_normalize(
        self, text: str, dim: int = None
    ) -> Dict[str, Any]:
        effective_dim = dim if dim is not None else self.dim
        raw = self._deterministic_vector(text, effective_dim)
        return self.adapter.normalize(raw, force_dim=effective_dim)

    def _deterministic_vector(self, text: str, dim: int) -> List[float]:
        """Generates a stable, reproducible normalized vector based on text feature hashing (DSKE)."""
        vec = np.zeros(dim, dtype=np.float32)
        text_lower = text.lower()
        words = text_lower.split()

        for i, word in enumerate(words):
            h = hash(word)
            idx = abs(h) % dim
            val = (h % 100) / 100.0
            vec[idx] += val * (1.0 / (i + 1) ** 0.5)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Seed uniform vector if text is empty
            vec = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

        return vec.tolist()

    def normalize_vector(
        self, vector: Any, force_dim: int = None
    ) -> Dict[str, Any]:
        return self.adapter.normalize(vector, force_dim=force_dim or self.dim)

    def reshape_vector(self, flat_vector: List[float], meta: Dict[str, Any]):
        return self.adapter.reshape(flat_vector, meta)


def _load_bge3() -> Optional[BGE3NeuralEmbedder]:
    """Attempt to load the BGE-M3 neural model from the local HF cache only.

    Returns ``None`` on any failure (model not cached, optional dependency
    missing, etc.) so callers seamlessly fall back to DSKE.
    """
    if not (settings.BGE_M3_ENABLED and settings.BGE_M3_RETRIEVAL_ENABLED):
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            settings.BGE_M3_MODEL_NAME, local_files_only=True
        )
        return BGE3NeuralEmbedder(model)
    except Exception as exc:
        logger.warning(
            f"BGE-M3 neural embedder unavailable ({exc}); "
            f"using DSKE deterministic fallback"
        )
        return None


_embedder_instance: Optional[Any] = None


def get_embedder() -> Embedder:
    """Return the active embedder.

    Prefers the neural BGE-M3 embedder when ``BGE_M3_ENABLED`` and
    ``BGE_M3_RETRIEVAL_ENABLED`` are set and the model is resolvable from the
    local HuggingFace cache; otherwise falls back to the DSKE Embedder.
    """
    global _embedder_instance
    if _embedder_instance is None:
        neural = _load_bge3()
        if neural is not None:
            _embedder_instance = neural
        else:
            _embedder_instance = Embedder()
    return _embedder_instance


__all__ = ["Embedder", "BGE3NeuralEmbedder", "get_embedder"]
