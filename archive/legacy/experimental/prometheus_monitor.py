from typing import Any, Dict, List, Optional


class PrometheusMonitor:
    """Compatibility monitor for metric collection and reporting."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        self.metrics.setdefault(name, []).append(float(value))

    def get_summary(self) -> Dict[str, Any]:
        return {name: {"count": len(values), "avg": round(sum(values) / len(values), 4) if values else 0.0} for name, values in self.metrics.items()}


def get_prometheus_monitor() -> PrometheusMonitor:
    return PrometheusMonitor()


__all__ = ["PrometheusMonitor", "get_prometheus_monitor"]
