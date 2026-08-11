"""
Advanced vector features for CrossMind:
1. True multi-vector search using Qdrant MultiVectorConfig (MAX_SIM)
2. Qdrant native sparse vector support
3. 3D tensor indexing and search
"""
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("crossmind.advanced_vectors")


class MultiVectorSearchEngine:
    """Token-level multi-vector search using Qdrant MultiVectorConfig."""

    @staticmethod
    def build_multivector_config(comparator: str = "MAX_SIM") -> Dict[str, Any]:
        return {
            "comparator": comparator,
        }

    @staticmethod
    def score_multivector(query_vector: List[float], doc_vectors: List[List[float]]) -> float:
        if not doc_vectors:
            return 0.0
        q = np.array(query_vector, dtype=np.float32)
        scores = []
        for doc_vec in doc_vectors:
            d = np.array(doc_vec, dtype=np.float32)
            if q.shape[0] != d.shape[0]:
                continue
            dot = float(np.dot(q, d))
            norm = float(np.linalg.norm(q) * np.linalg.norm(d))
            scores.append(dot / norm if norm > 0 else 0.0)
        return max(scores) if scores else 0.0


class SparseVectorSearchEngine:
    """Qdrant-native sparse vector search support."""

    @staticmethod
    def to_qdrant_sparse(indices: List[int], values: List[float]) -> Dict[str, Any]:
        return {
            "indices": indices,
            "values": values,
        }

    @staticmethod
    def dot_product(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        idx_a = {i: v for i, v in zip(a.get("indices", []), a.get("values", []))}
        idx_b = {i: v for i, v in zip(b.get("indices", []), b.get("values", []))}
        common = set(idx_a.keys()) & set(idx_b.keys())
        return sum(idx_a[i] * idx_b[i] for i in common)


class Tensor3DSearchEngine:
    """3D tensor search: stores flattened tensor with shape metadata."""

    @staticmethod
    def flatten_tensor(tensor: np.ndarray) -> List[float]:
        return tensor.astype(np.float32).flatten().tolist()

    @staticmethod
    def reshape_tensor(flat: List[float], shape: tuple) -> np.ndarray:
        return np.array(flat, dtype=np.float32).reshape(shape)

    @staticmethod
    def similarity_3d(query_tensor: np.ndarray, doc_tensor: np.ndarray) -> float:
        if query_tensor.shape != doc_tensor.shape:
            return 0.0
        q = query_tensor.flatten().astype(np.float32)
        d = doc_tensor.flatten().astype(np.float32)
        dot = float(np.dot(q, d))
        norm = float(np.linalg.norm(q) * np.linalg.norm(d))
        return dot / norm if norm > 0 else 0.0
