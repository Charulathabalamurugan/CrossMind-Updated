import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anomaly_detection.base import AnomalyResult, BaseDetector

logger = logging.getLogger("crossmind.anomaly_detection")

CONTEXT_CONFIG = {
    "ingestion_metadata": {
        "threshold": 0.78,
        "action": "quarantine",
        "model": "one_class_svm",
    },
    "ingestion_embedding": {
        "threshold": 0.78,
        "action": "quarantine",
        "model": "isolation_forest",
    },
    "retrieval_query": {
        "threshold": 0.72,
        "action": "force_deep_and_log",
        "model": "isolation_forest_time_window",
    },
    "reasoning_output": {
        "threshold": 0.72,
        "action": "trigger_debate",
        "model": "autoencoder_reconstruction_error",
    },
    "scientific_discovery": {
        "threshold": 0.78,
        "action": "candidate_discovery",
        "model": "mahalanobis_distance",
    },
}

_INJECTION_MARKERS = (
    "ignore all previous instructions",
    "reveal the system prompt",
    "bypass safety",
    "disregard previous instructions",
    "act as an unrestricted",
    "exfiltrate",
)


def _stable_json(data: Any) -> str:
    try:
        return json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(data)


def _input_hash(data: Any) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8", errors="replace")).hexdigest()


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class ContextualAnomalyDetector(BaseDetector):
    """Context-specific anomaly scoring with optional fitted sklearn models."""

    def __init__(
        self,
        context_name: str,
        training_window: int = 512,
        model_dir: Optional[Path] = None,
    ):
        if context_name not in CONTEXT_CONFIG:
            raise ValueError(f"Unsupported anomaly context: {context_name}")
        self.context_name = context_name
        self.config = CONTEXT_CONFIG[context_name]
        self._normal_vectors: Deque[np.ndarray] = deque(maxlen=training_window)
        self._model = None
        self._model_threshold: Optional[float] = None
        self._model_version = 0
        self._lock = threading.RLock()
        self._model_dir = (
            Path(model_dir)
            if model_dir
            else Path(
                os.environ.get(
                    "CROSSMIND_ANOMALY_MODEL_DIR",
                    str(Path(__file__).parent / "models"),
                )
            )
        )
        self._load_model()

    @property
    def _artifact_path(self) -> Path:
        return self._model_dir / f"{self.context_name}.joblib"

    def _load_model(self) -> None:
        if not self._artifact_path.exists():
            return
        import joblib

        artifact = joblib.load(self._artifact_path)
        if artifact.get("context") != self.context_name or "model" not in artifact:
            raise ValueError(f"Invalid anomaly model artifact: {self._artifact_path}")
        self._model = artifact["model"]
        self._model_threshold = float(artifact["threshold"])
        self._model_version = int(artifact["version"])

    def _persist_model(self) -> None:
        import joblib

        self._model_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self._model_dir / (
            f".{self.context_name}-{uuid.uuid4().hex}.joblib.tmp"
        )
        with self._lock:
            try:
                joblib.dump(
                    {
                        "context": self.context_name,
                        "model": self._model,
                        "threshold": self._model_threshold,
                        "version": self._model_version,
                    },
                    temporary_path,
                )
                temporary_path.replace(self._artifact_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    async def detect(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AnomalyResult:
        import asyncio

        return await asyncio.to_thread(self._detect, data, context or {})

    def _detect(self, data: Any, context: Dict[str, Any]) -> AnomalyResult:
        input_hash = _input_hash(data)
        vector, heuristic = self._features_and_heuristic(data, context)
        score = heuristic
        model_used = self.config["model"]
        confidence = 0.65

        with self._lock:
            if self._model is not None and heuristic < 1.0:
                model_score = self._score_with_model(vector)
                score = max(heuristic * 0.65, model_score)
                model_used = f"{self.config['model']}:v{self._model_version}"
                confidence = 0.85
            threshold = float(context.get("threshold", self.config["threshold"]))
            score = _clamp(score)
            flagged = score >= threshold

        return AnomalyResult(
            context=self.context_name,
            label="anomaly" if flagged else "normal",
            score=round(score, 6),
            confidence=round(confidence, 4),
            threshold=threshold,
            action=self.config["action"] if flagged else "allow",
            model=model_used,
            input_hash=input_hash,
            details={"model_trained": self._model is not None, "model_version": self._model_version},
        )

    def _features_and_heuristic(
        self, data: Any, context: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        if self.context_name == "ingestion_embedding":
            return self._embedding_features(data)
        if self.context_name == "ingestion_metadata":
            return self._metadata_features(data)
        if self.context_name == "retrieval_query":
            return self._query_features(data, context)
        if self.context_name == "reasoning_output":
            return self._reasoning_features(data)
        return self._discovery_features(data)

    @staticmethod
    def _embedding_features(data: Any) -> Tuple[np.ndarray, float]:
        try:
            values = np.asarray(data, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=np.float64)
        if values.size == 0:
            return np.zeros(16, dtype=np.float64), 1.0
        if not np.all(np.isfinite(values)):
            return np.zeros(16, dtype=np.float64), 1.0
        norm = float(np.linalg.norm(values))
        features = np.zeros(16, dtype=np.float64)
        features[:8] = [
            math.log1p(norm),
            float(np.mean(values)),
            float(np.std(values)),
            float(np.max(np.abs(values))),
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 90)),
            float(np.percentile(values, 10)),
        ]
        features[8 : 8 + min(8, values.size)] = values[:8]
        score = 0.0
        if norm < 1e-8:
            score = 0.95
        elif norm > 1000:
            score = 0.9
        elif float(np.max(np.abs(values))) > 100:
            score = 0.85
        return features, score

    @staticmethod
    def _metadata_features(data: Any) -> Tuple[np.ndarray, float]:
        item = data if isinstance(data, dict) else {}
        content = item.get("content", item.get("text", ""))
        if not isinstance(content, str):
            content = str(content or "")
        words = re.findall(r"\b[\w'-]+\b", content)
        title = str(item.get("title", "") or "").strip()
        authors = item.get("authors") or []
        required_present = sum(bool(item.get(key)) for key in ("title", "domain", "year", "content"))
        duplicate_ratio = float(item.get("duplicate_ratio", 0.0) or 0.0)
        quality = item.get("quality_score")
        quality_value = float(quality) if isinstance(quality, (int, float)) else 0.5
        vector = np.asarray(
            [
                math.log1p(len(content)),
                math.log1p(len(words)),
                len(set(words)) / max(1, len(words)),
                len(title),
                len(authors) if isinstance(authors, (list, tuple)) else bool(authors),
                required_present / 4,
                quality_value,
                duplicate_ratio,
                len(content) / max(1, len(words)),
                sum(ch.isprintable() for ch in content) / max(1, len(content)),
                len(re.findall(r"[^\w\s]", content)) / max(1, len(content)),
                len(re.findall(r"(.)\1{7,}", content)),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        score = 0.0
        if not content.strip():
            score = 0.99
        elif len(words) < 2:
            score = 0.82
        elif duplicate_ratio >= 0.8:
            score = 0.9
        elif quality_value < 0.2:
            score = 0.85
        elif len(set(words)) / len(words) < 0.12 and len(words) > 20:
            score = 0.8
        return vector, score

    @staticmethod
    def _query_features(data: Any, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        query = data if isinstance(data, str) else str(data or "")
        lower = query.lower()
        tokens = re.findall(r"\b[\w'-]+\b", lower)
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        marker_hits = sum(marker in lower for marker in _INJECTION_MARKERS)
        punctuation_ratio = sum(not char.isalnum() and not char.isspace() for char in query) / max(1, len(query))
        token_counts = Counter(token for token in tokens if len(token) > 3)
        repeated = sum(count >= 5 for count in token_counts.values())
        vector = np.asarray(
            [
                math.log1p(len(query)),
                math.log1p(len(tokens)),
                unique_ratio,
                punctuation_ratio,
                marker_hits,
                repeated,
                len(re.findall(r"https?://", lower)),
                len(re.findall(r"\b(?:select|drop|union|insert)\b", lower)),
                len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", query)),
                float(context.get("queries_in_window", 0)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        score = 0.0
        if marker_hits:
            score = min(0.99, 0.82 + marker_hits * 0.08)
        elif len(query) > 12000:
            score = 0.9
        elif punctuation_ratio > 0.6 and len(query) > 100:
            score = 0.78
        elif repeated >= 5:
            score = 0.76
        return vector, score

    @staticmethod
    def _reasoning_features(data: Any) -> Tuple[np.ndarray, float]:
        item = data if isinstance(data, dict) else {"output_text": data}
        text = str(item.get("output_text", item.get("summary", item.get("hypothesis", ""))) or "")
        tokens = re.findall(r"\b[\w'-]+\b", text.lower())
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        citations = item.get("cited_evidence_ids", [])
        confidence = item.get("confidence_score", item.get("confidence", 0.5))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        vector = np.asarray(
            [
                math.log1p(len(text)),
                math.log1p(len(tokens)),
                unique_ratio,
                confidence,
                len(citations) if isinstance(citations, (list, tuple)) else 0,
                len(re.findall(r"\b(?:therefore|however|because|evidence|indicates)\b", text.lower())),
                len(re.findall(r"(.)\1{7,}", text)),
                len(re.findall(r"\b(?:certainly|undoubtedly|proves)\b", text.lower())),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        score = 0.0
        if not text.strip():
            score = 0.99
        elif len(tokens) < 2:
            score = 0.82
        elif confidence < 0.1 and len(tokens) > 30:
            score = 0.8
        elif len(tokens) > 30 and not citations and item.get("cited_evidence_ids") is not None:
            score = 0.75
        elif unique_ratio < 0.15 and len(tokens) > 20:
            score = 0.84
        return vector, score

    @staticmethod
    def _discovery_features(data: Any) -> Tuple[np.ndarray, float]:
        item = data if isinstance(data, dict) else {}
        values: List[float] = []
        for key in (
            "novelty_score",
            "feasibility_score",
            "evidence_score",
            "combined_score",
            "cross_domain_score",
            "overall_score",
        ):
            try:
                values.append(float(item.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                values.append(0.0)
        values.extend(
            [
                float(item.get("evidence_count", 0) or 0),
                float(item.get("domain_count", 0) or 0),
                float(item.get("confidence", 0.0) or 0.0),
            ]
        )
        features = np.asarray(values + [0.0] * (16 - len(values)), dtype=np.float64)
        novelty = max(values[0], values[4] / 100.0, values[5] / 100.0)
        evidence = max(values[2] / 100.0, min(1.0, values[6] / 5.0))
        score = 0.0
        if novelty >= 0.85 and evidence > 0:
            score = min(0.99, 0.8 + (novelty - 0.85) * 0.6)
        return features, score

    def add_normal_sample(self, data: Any, context: Optional[Dict[str, Any]] = None) -> None:
        vector, _ = self._features_and_heuristic(data, context or {})
        with self._lock:
            self._normal_vectors.append(vector)

    def fit(self, normal_samples: Sequence[Any]) -> int:
        """Train the context-specific model from known-normal historical samples."""
        vectors = [self._features_and_heuristic(sample, {})[0] for sample in normal_samples]
        minimum = 12 if self.context_name == "reasoning_output" else 8
        if len(vectors) < minimum:
            raise ValueError(
                f"{self.context_name} needs at least {minimum} known-normal samples; received {len(vectors)}"
            )
        matrix = np.vstack(vectors)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Training data contains non-finite features")

        model_type = self.config["model"]
        if model_type.startswith("isolation_forest"):
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
            ).fit(matrix)
            raw_scores = -model.score_samples(matrix)
            threshold = float(np.percentile(raw_scores, 97))
        elif model_type == "one_class_svm":
            from sklearn.svm import OneClassSVM

            model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(matrix)
            raw_scores = -model.decision_function(matrix).reshape(-1)
            threshold = float(np.percentile(raw_scores, 97))
        elif model_type == "autoencoder_reconstruction_error":
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(matrix)
            scaled = scaler.transform(matrix)
            model = (
                MLPRegressor(
                    hidden_layer_sizes=(8,),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                    early_stopping=False,
                ).fit(scaled, scaled),
                scaler,
            )
            raw_scores = np.mean((model[0].predict(scaled) - scaled) ** 2, axis=1)
            threshold = float(np.percentile(raw_scores, 97))
        else:
            model = matrix
            center = np.mean(matrix, axis=0)
            covariance = np.cov(matrix, rowvar=False) + np.eye(matrix.shape[1]) * 1e-6
            inverse = np.linalg.pinv(covariance)
            raw_scores = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", matrix - center, inverse, matrix - center)))
            threshold = float(np.percentile(raw_scores, 97))

        with self._lock:
            self._model = model
            self._model_threshold = max(threshold, 1e-8)
            self._normal_vectors.clear()
            self._normal_vectors.extend(vectors[-self._normal_vectors.maxlen :])
            self._model_version += 1
            self._persist_model()
        logger.info(
            "Trained anomaly model",
            extra={
                "anomaly_context": self.context_name,
                "model": model_type,
                "model_version": self._model_version,
                "training_samples": len(vectors),
            },
        )
        return self._model_version

    def retrain_from_observed(self, minimum_samples: int = 8) -> Optional[int]:
        with self._lock:
            samples = list(self._normal_vectors)
        if len(samples) < minimum_samples:
            return None
        return self._fit_vectors(samples)

    def _fit_vectors(self, vectors: Sequence[np.ndarray]) -> int:
        """Fit from previously extracted normal vectors to support periodic refresh."""
        # Reuse fit's model-specific implementation without reinterpreting feature vectors.
        matrix = np.vstack(vectors)
        model_type = self.config["model"]
        if model_type.startswith("isolation_forest"):
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(n_estimators=100, contamination="auto", random_state=42).fit(matrix)
            raw_scores = -model.score_samples(matrix)
            threshold = float(np.percentile(raw_scores, 97))
        elif model_type == "one_class_svm":
            from sklearn.svm import OneClassSVM

            model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(matrix)
            raw_scores = -model.decision_function(matrix).reshape(-1)
            threshold = float(np.percentile(raw_scores, 97))
        elif model_type == "autoencoder_reconstruction_error":
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(matrix)
            scaled = scaler.transform(matrix)
            model = (
                MLPRegressor(hidden_layer_sizes=(8,), max_iter=500, random_state=42).fit(scaled, scaled),
                scaler,
            )
            raw_scores = np.mean((model[0].predict(scaled) - scaled) ** 2, axis=1)
            threshold = float(np.percentile(raw_scores, 97))
        else:
            model = matrix
            center = np.mean(matrix, axis=0)
            covariance = np.cov(matrix, rowvar=False) + np.eye(matrix.shape[1]) * 1e-6
            inverse = np.linalg.pinv(covariance)
            raw_scores = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", matrix - center, inverse, matrix - center)))
            threshold = float(np.percentile(raw_scores, 97))
        with self._lock:
            self._model = model
            self._model_threshold = max(threshold, 1e-8)
            self._model_version += 1
            self._persist_model()
        return self._model_version

    def _score_with_model(self, vector: np.ndarray) -> float:
        model_type = self.config["model"]
        try:
            if model_type.startswith("isolation_forest"):
                raw = -float(self._model.score_samples(vector.reshape(1, -1))[0])
            elif model_type == "one_class_svm":
                raw = -float(self._model.decision_function(vector.reshape(1, -1))[0])
            elif model_type == "autoencoder_reconstruction_error":
                model, scaler = self._model
                scaled = scaler.transform(vector.reshape(1, -1))
                raw = float(np.mean((model.predict(scaled) - scaled) ** 2))
            else:
                matrix = self._model
                center = np.mean(matrix, axis=0)
                covariance = np.cov(matrix, rowvar=False) + np.eye(matrix.shape[1]) * 1e-6
                inverse = np.linalg.pinv(covariance)
                delta = vector - center
                raw = math.sqrt(max(0.0, float(delta @ inverse @ delta)))
            ratio = raw / max(self._model_threshold or 1e-8, 1e-8)
            return _clamp(0.5 + 0.5 * (ratio - 1.0))
        except (ValueError, TypeError, AttributeError, FloatingPointError) as exc:
            logger.warning(
                "Anomaly model scoring failed for %s: %s",
                self.context_name,
                exc,
            )
            return 0.0
