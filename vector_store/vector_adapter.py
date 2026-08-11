"""
Universal vector adapter that normalizes any vector representation
into a flat dense vector for Qdrant storage, while preserving original
shape metadata for reshaping on retrieval.

Supported input types:
- Flat dense: List[float] / 1-D np.ndarray
- Multi-vector: List[List[float]] / 2-D np.ndarray (e.g., token embeddings)
- 2-D/3-D tensors: np.ndarray with shape (H, W) or (D, H, W)
- Sparse dict: Dict[int, float] (term_id -> weight)
- Already normalized vectors (pass through)

All outputs are:
- flat_vector: List[float] for Qdrant
- vector_meta: Dict with original shape, type, and norm for later reconstruction
"""
import math
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("crossmind.vector_adapter")

VectorLike = Union[List[float], List[List[float]], np.ndarray, Dict[int, float]]


class VectorAdapter:
    @staticmethod
    def normalize(vector: VectorLike, force_dim: Optional[int] = None) -> Dict[str, Any]:
        if vector is None:
            return VectorAdapter._empty()

        # 1) Sparse dict input
        if isinstance(vector, dict):
            return VectorAdapter._from_sparse_dict(vector, force_dim)

        # 2) numpy array input
        if isinstance(vector, np.ndarray):
            return VectorAdapter._from_ndarray(vector, force_dim)

        # 3) list input
        if isinstance(vector, list):
            if not vector:
                return VectorAdapter._empty()
            if isinstance(vector[0], (int, float)):
                return VectorAdapter._from_flat_list(vector, force_dim)
            if isinstance(vector[0], list):
                return VectorAdapter._from_nested_list(vector, force_dim)
            raise ValueError(f"Unsupported list element type: {type(vector[0])}")

        raise ValueError(f"Unsupported vector type: {type(vector)}")

    @staticmethod
    def reshape(flat_vector: List[float], meta: Dict[str, Any]) -> VectorLike:
        vtype = meta.get("type", "dense")
        if vtype == "dense":
            shape = meta.get("shape")
            if shape is not None:
                try:
                    arr = np.array(flat_vector, dtype=np.float32)
                    return arr.reshape(shape).tolist()
                except Exception:
                    pass
            return flat_vector
        if vtype == "multi_vector":
            try:
                arr = np.array(flat_vector, dtype=np.float32)
                return arr.reshape(meta["shape"]).tolist()
            except Exception:
                return flat_vector
        if vtype == "sparse":
            return dict(zip(meta.get("indices", []), flat_vector))
        return flat_vector

    @staticmethod
    def cosine_similarity(a: VectorLike, b: VectorLike, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> float:
        try:
            vec_a = VectorAdapter._to_1d_for_similarity(a, meta_a)
            vec_b = VectorAdapter._to_1d_for_similarity(b, meta_b)
            dot = float(np.dot(vec_a, vec_b))
            norm = float(np.linalg.norm(vec_a)) * float(np.linalg.norm(vec_b))
            return dot / norm if norm > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "flat_vector": [],
            "vector_meta": {
                "type": "dense",
                "shape": None,
                "original_dim": 0,
                "norm": 0.0,
                "element_count": 0,
            },
        }

    @staticmethod
    def _from_flat_list(vector: List[float], force_dim: Optional[int]) -> Dict[str, Any]:
        arr = np.array(vector, dtype=np.float32)
        if force_dim is not None and arr.shape[0] != force_dim:
            if arr.shape[0] > force_dim:
                arr = arr[:force_dim]
            else:
                pad = np.zeros(force_dim, dtype=np.float32)
                pad[: arr.shape[0]] = arr
                arr = pad
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return {
            "flat_vector": arr.tolist(),
            "vector_meta": {
                "type": "dense",
                "shape": [arr.shape[0]],
                "original_dim": arr.shape[0],
                "norm": round(norm, 6),
                "element_count": int(arr.shape[0]),
            },
        }

    @staticmethod
    def _from_nested_list(vector: List[List[float]], force_dim: Optional[int]) -> Dict[str, Any]:
        arr = np.array(vector, dtype=np.float32)
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1)
        if force_dim is not None and arr.shape[-1] != force_dim:
            if arr.shape[-1] > force_dim:
                arr = arr[..., :force_dim]
            else:
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, force_dim - arr.shape[-1])]
                arr = np.pad(arr, pad_width, mode="constant")
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        arr = arr / norms
        flat = arr.flatten().tolist()
        return {
            "flat_vector": flat,
            "vector_meta": {
                "type": "multi_vector",
                "shape": list(arr.shape),
                "original_dim": int(arr.shape[-1]),
                "norm": float(np.mean(norms)),
                "element_count": int(arr.size),
                "num_vectors": int(arr.shape[0]),
            },
        }

    @staticmethod
    def _from_ndarray(vector: np.ndarray, force_dim: Optional[int]) -> Dict[str, Any]:
        if vector.ndim == 1:
            return VectorAdapter._from_flat_list(vector.tolist(), force_dim)
        if vector.ndim == 2:
            return VectorAdapter._from_nested_list(vector.tolist(), force_dim)
        flat = vector.astype(np.float32).flatten()
        if force_dim is not None and flat.shape[0] != force_dim:
            if flat.shape[0] > force_dim:
                flat = flat[:force_dim]
            else:
                pad = np.zeros(force_dim, dtype=np.float32)
                pad[: flat.shape[0]] = flat
                flat = pad
        norm = float(np.linalg.norm(flat))
        if norm > 0:
            flat = flat / norm
        return {
            "flat_vector": flat.tolist(),
            "vector_meta": {
                "type": "dense",
                "shape": [flat.shape[0]],
                "original_dim": int(flat.shape[0]),
                "norm": round(norm, 6),
                "element_count": int(flat.shape[0]),
            },
        }

    @staticmethod
    def _from_sparse_dict(vector: Dict[int, float], force_dim: Optional[int]) -> Dict[str, Any]:
        if force_dim is None:
            force_dim = max(vector.keys()) + 1 if vector else 1
        indices = sorted(vector.keys())
        values = [float(vector[i]) for i in indices]
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
        flat = [0.0] * force_dim
        for idx, val in zip(indices, values):
            if 0 <= idx < force_dim:
                flat[idx] = val
        return {
            "flat_vector": flat,
            "vector_meta": {
                "type": "sparse",
                "shape": [force_dim],
                "original_dim": force_dim,
                "norm": round(norm, 6),
                "element_count": len(indices),
                "indices": indices,
                "values": values,
            },
        }

    @staticmethod
    def _to_1d_for_similarity(vector: VectorLike, meta: Dict[str, Any]) -> np.ndarray:
        vtype = meta.get("type", "dense")
        if vtype == "sparse":
            if isinstance(vector, dict):
                size = meta.get("shape", [max(vector.keys()) + 1])[0]
                arr = np.zeros(size, dtype=np.float32)
                for idx, val in vector.items():
                    if 0 <= idx < size:
                        arr[idx] = float(val)
                return arr
            if isinstance(vector, list):
                return np.array(vector, dtype=np.float32)
        if isinstance(vector, np.ndarray):
            if vector.ndim > 1:
                return vector.flatten().astype(np.float32)
            return vector.astype(np.float32)
        if isinstance(vector, list):
            if vector and isinstance(vector[0], list):
                flat = []
                for sub in vector:
                    flat.extend(sub)
                return np.array(flat, dtype=np.float32)
            return np.array(vector, dtype=np.float32)
        return np.array([], dtype=np.float32)


_adapter_instance: Optional[VectorAdapter] = None


def get_vector_adapter() -> VectorAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = VectorAdapter()
    return _adapter_instance
