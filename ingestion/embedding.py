import numpy as np
import logging
from typing import List
from config import settings

logger = logging.getLogger("crossmind.embedding")

class Embedder:
    """
    Embedding adapter supporting 256-dim Matryoshka truncated embeddings.
    Designed for nomic-embed-text compatibility.
    """
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, dim: int = settings.EMBEDDING_DIM):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to fast deterministic embedding generator.")
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            try:
                embeddings = self._model.encode(texts, normalize_embeddings=True)
                # Truncate to desired Matryoshka dimension (e.g. 256) and re-normalize
                truncated = []
                for vec in embeddings:
                    vec = vec[:self.dim]
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    truncated.append(vec.tolist())
                return truncated
            except Exception as e:
                logger.error(f"Error during SentenceTransformer encoding: {e}. Using deterministic generator fallback.")

        # Fallback deterministic embedder generating normalized 256-dim vectors
        results = []
        for text in texts:
            vec = self._deterministic_vector(text, self.dim)
            results.append(vec)
        return results

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def _deterministic_vector(self, text: str, dim: int) -> List[float]:
        """Generates a stable, reproducible normalized vector based on text feature hashing."""
        vec = np.zeros(dim, dtype=np.float32)
        text_lower = text.lower()
        words = text_lower.split()
        
        for i, word in enumerate(words):
            h = hash(word)
            idx = abs(h) % dim
            val = (h % 100) / 100.0
            vec[idx] += val * (1.0 / (i + 1)**0.5)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Seed uniform vector if text is empty
            vec = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
            
        return vec.tolist()

_embedder_instance = None

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
