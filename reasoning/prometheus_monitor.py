import time
import logging
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.monitor")

_prometheus_counter: Dict[str, int] = {}
_prometheus_gauge: Dict[str, float] = {}
_prometheus_histogram: Dict[str, list] = {}

class PrometheusMonitor:
    def __init__(self):
        self.enabled = settings.PROMETHEUS_ENABLED
        self._request_count = 0
        self._error_count = 0
        self._latencies: list = []

    def increment_counter(self, name: str, value: int = 1):
        _prometheus_counter[name] = _prometheus_counter.get(name, 0) + value

    def set_gauge(self, name: str, value: float):
        _prometheus_gauge[name] = value

    def record_latency(self, name: str, latency_ms: float):
        _prometheus_histogram.setdefault(name, []).append(latency_ms)

    def get_metrics_output(self) -> str:
        lines = ["# HELP crossmind_requests_total Total API requests", "# TYPE crossmind_requests_total counter"]
        lines.append(f"crossmind_requests_total {self._request_count}")
        lines.append("# HELP crossmind_errors_total Total API errors")
        lines.append("# TYPE crossmind_errors_total counter")
        lines.append(f"crossmind_errors_total {self._error_count}")
        for name, count in _prometheus_counter.items():
            lines.append(f"# HELP crossmind_{name} Custom counter")
            lines.append(f"# TYPE crossmind_{name} counter")
            lines.append(f"crossmind_{name} {count}")
        for name, value in _prometheus_gauge.items():
            lines.append(f"# HELP crossmind_{name} Custom gauge")
            lines.append(f"# TYPE crossmind_{name} gauge")
            lines.append(f"crossmind_{name} {value}")
        for name, latencies in _prometheus_histogram.items():
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                lines.append(f"# HELP crossmind_{name}_latency_ms Average latency")
                lines.append(f"# TYPE crossmind_{name}_latency_ms gauge")
                lines.append(f"crossmind_{name}_latency_ms {avg_latency:.2f}")
        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "open_telemetry_enabled": settings.OPENTELEMETRY_ENABLED,
        }

monitor_instance = None

def get_prometheus_monitor() -> PrometheusMonitor:
    global monitor_instance
    if monitor_instance is None:
        monitor_instance = PrometheusMonitor()
    return monitor_instance
