from typing import Any, Dict, List, Optional


class PrometheusMonitor:
    """Minimal compatibility monitor used by the runtime and compatibility tests."""

    def __init__(self):
        self._metrics: List[Dict[str, Any]] = []

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None):
        self._metrics.append({"name": name, "value": value, "labels": labels or {}})

    def get_summary(self) -> Dict[str, Any]:
        return {
            "metric_count": len(self._metrics),
            "metrics": self._metrics[-20:],
        }


def get_prometheus_monitor() -> PrometheusMonitor:
    return PrometheusMonitor()


__all__ = ["PrometheusMonitor", "get_prometheus_monitor"]
