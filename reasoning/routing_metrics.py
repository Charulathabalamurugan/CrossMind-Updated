"""
Routing metrics tracker for model selection, latency, and quality scoring.
"""
import time
import logging
from typing import Dict, Any, List, Optional
from config import settings

logger = logging.getLogger("crossmind.routing_metrics")


class RoutingMetrics:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._lock = __import__("threading").Lock()

    def record_query(
        self,
        query: str,
        complexity: str,
        model: str,
        latency_ms: float,
        quality_score: float = 0.0,
        evidence_count: int = 0,
    ) -> Dict[str, Any]:
        record = {
            "query": query[:200],
            "complexity": complexity,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "quality_score": round(quality_score, 3),
            "evidence_count": evidence_count,
            "timestamp": time.time(),
        }
        with self._lock:
            self._records.append(record)
        return record

    def get_summary(self, limit: int = 100) -> Dict[str, Any]:
        with self._lock:
            recent = self._records[-limit:] if self._records else []
        if not recent:
            return {"total_queries": 0}

        model_latency: Dict[str, List[float]] = {}
        model_quality: Dict[str, List[float]] = {}
        complexity_counts: Dict[str, int] = {}

        for rec in recent:
            model = rec.get("model", "unknown")
            model_latency.setdefault(model, []).append(rec.get("latency_ms", 0.0))
            model_quality.setdefault(model, []).append(rec.get("quality_score", 0.0))
            complexity_counts[rec.get("complexity", "unknown")] = complexity_counts.get(rec.get("complexity", "unknown"), 0) + 1

        avg_latency = {}
        avg_quality = {}
        for model, lats in model_latency.items():
            avg_latency[model] = round(sum(lats) / len(lats), 2)
        for model, quals in model_quality.items():
            avg_quality[model] = round(sum(quals) / len(quals), 3)

        return {
            "total_queries": len(recent),
            "complexity_distribution": complexity_counts,
            "model_selection_distribution": {model: len(lats) for model, lats in model_latency.items()},
            "avg_latency_ms_by_model": avg_latency,
            "avg_quality_by_model": avg_quality,
            "overall_avg_latency_ms": round(sum(r.get("latency_ms", 0.0) for r in recent) / len(recent), 2),
            "overall_avg_quality": round(sum(r.get("quality_score", 0.0) for r in recent) / len(recent), 3),
        }

    def get_recent_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return self._records[-limit:] if self._records else []


_routing_metrics_instance: Optional[RoutingMetrics] = None


def get_routing_metrics() -> RoutingMetrics:
    global _routing_metrics_instance
    if _routing_metrics_instance is None:
        _routing_metrics_instance = RoutingMetrics()
    return _routing_metrics_instance
