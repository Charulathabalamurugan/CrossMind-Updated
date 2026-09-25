import time
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

from config import settings

logger = logging.getLogger("crossmind.continuous_learning")


@dataclass
class LearningSignal:
    query: str
    doc_id: str
    original_score: float
    corrected_score: float
    user_role: str
    signal_type: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceSnapshot:
    timestamp: float
    metrics: Dict[str, float]
    data_drift_score: float
    concept_drift_score: float
    sample_count: int


class ContinuousLearningEngine:
    def __init__(self):
        self._signals: deque = deque(maxlen=10000)
        self._performance_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._enabled = settings.CONTINUOUS_LEARNING_ENABLED
        self._min_signals_for_update = 50
        self._drift_threshold = 0.15
        self._last_update = 0.0
        self._update_interval = 3600
        self._pending_signals = 0

    def record_signal(
        self,
        query: str,
        doc_id: str,
        original_score: float,
        corrected_score: float,
        user_role: str,
        signal_type: str = "relevance_correction",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningSignal:
        if not self._enabled:
            return LearningSignal(
                query=query,
                doc_id=doc_id,
                original_score=original_score,
                corrected_score=corrected_score,
                user_role=user_role,
                signal_type=signal_type,
            )
        signal = LearningSignal(
            query=query,
            doc_id=doc_id,
            original_score=original_score,
            corrected_score=corrected_score,
            user_role=user_role,
            signal_type=signal_type,
            metadata=metadata or {},
        )
        with self._lock:
            self._signals.append(signal)
            self._pending_signals += 1
        logger.info(
            f"Learning signal recorded: query='{query[:50]}...', "
            f"original={original_score:.3f}, corrected={corrected_score:.3f}, "
            f"type={signal_type}"
        )
        return signal

    def record_performance_snapshot(
        self,
        metrics: Dict[str, float],
        data_drift_score: float = 0.0,
        concept_drift_score: float = 0.0,
        sample_count: int = 0,
    ) -> ModelPerformanceSnapshot:
        snapshot = ModelPerformanceSnapshot(
            timestamp=time.time(),
            metrics=metrics,
            data_drift_score=data_drift_score,
            concept_drift_score=concept_drift_score,
            sample_count=sample_count,
        )
        with self._lock:
            self._performance_history.append(snapshot)
        return snapshot

    def get_recent_signals(self, limit: int = 100) -> List[LearningSignal]:
        with self._lock:
            return list(self._signals)[-limit:]

    def get_performance_history(self, limit: int = 100) -> List[ModelPerformanceSnapshot]:
        with self._lock:
            return list(self._performance_history)[-limit:]

    def should_update(self) -> bool:
        if not self._enabled:
            return False
        with self._lock:
            elapsed = time.time() - self._last_update
            return self._pending_signals >= self._min_signals_for_update and elapsed >= self._update_interval

    def detect_drift(self) -> Dict[str, Any]:
        with self._lock:
            if len(self._performance_history) < 10:
                return {"drift_detected": False, "reason": "insufficient_history"}

            recent = list(self._performance_history)[-10:]
            older = list(self._performance_history)[-20:-10] if len(self._performance_history) >= 20 else []

            if not older:
                return {"drift_detected": False, "reason": "insufficient_comparison_window"}

            recent_metrics = {}
            older_metrics = {}

            for key in recent[0].metrics.keys():
                recent_vals = [s.metrics.get(key, 0.0) for s in recent]
                older_vals = [s.metrics.get(key, 0.0) for s in older]
                recent_metrics[key] = sum(recent_vals) / len(recent_vals)
                older_metrics[key] = sum(older_vals) / len(older_vals)

            drift_scores = {}
            for key in recent_metrics:
                if older_metrics[key] != 0:
                    drift = abs(recent_metrics[key] - older_metrics[key]) / older_metrics[key]
                    drift_scores[key] = drift

            max_drift = max(drift_scores.values()) if drift_scores else 0.0
            drift_detected = max_drift > self._drift_threshold

            return {
                "drift_detected": drift_detected,
                "max_drift_score": round(max_drift, 4),
                "drift_scores": {k: round(v, 4) for k, v in drift_scores.items()},
                "recent_metrics": {k: round(v, 4) for k, v in recent_metrics.items()},
                "older_metrics": {k: round(v, 4) for k, v in older_metrics.items()},
            }

    def trigger_update(self, pipeline_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.should_update():
            return {"updated": False, "reason": "conditions_not_met"}

        drift_result = self.detect_drift()

        with self._lock:
            signals_to_process = list(self._signals)
            self._pending_signals = 0
            self._last_update = time.time()

        logger.info(f"Continuous learning update triggered. Processing {len(signals_to_process)} signals. Drift: {drift_result}")

        return {
            "updated": True,
            "signals_processed": len(signals_to_process),
            "drift_analysis": drift_result,
            "timestamp": time.time(),
            "pipeline_context": pipeline_context or {},
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            signal_types = {}
            for s in self._signals:
                signal_types[s.signal_type] = signal_types.get(s.signal_type, 0) + 1

            return {
                "enabled": self._enabled,
                "total_signals": len(self._signals),
                "pending_signals": self._pending_signals,
                "signal_types": signal_types,
                "performance_snapshots": len(self._performance_history),
                "last_update": self._last_update,
                "min_signals_for_update": self._min_signals_for_update,
                "drift_threshold": self._drift_threshold,
                "should_update": self.should_update(),
            }


_continuous_learning_instance = None


def get_continuous_learning_engine() -> ContinuousLearningEngine:
    global _continuous_learning_instance
    if _continuous_learning_instance is None:
        _continuous_learning_instance = ContinuousLearningEngine()
    return _continuous_learning_instance