import time
import logging
from typing import Any, Dict, List
from collections import defaultdict

logger = logging.getLogger("crossmind.benchmark")

class BenchmarkCollector:
    def __init__(self):
        self._metrics: List[Dict[str, Any]] = []
        self._phase_times: Dict[str, float] = defaultdict(float)
        self._counters = defaultdict(int)
        self._start_times: Dict[str, float] = {}

    def start_phase(self, phase_name: str):
        self._start_times[phase_name] = time.time()

    def end_phase(self, phase_name: str):
        if phase_name in self._start_times:
            elapsed = time.time() - self._start_times[phase_name]
            self._phase_times[phase_name] += elapsed
            self._counters[phase_name] += 1
            del self._start_times[phase_name]

    def record_metric(self, metric_name: str, value: float, tags: Dict[str, Any] = None):
        self._metrics.append({
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time(),
        })

    def record_retrieval(self, dense_ms: float, sparse_ms: float, dense_count: int, sparse_count: int):
        self._phase_times["dense_retrieval"] += dense_ms
        self._phase_times["sparse_retrieval"] += sparse_ms
        self._counters["retrieval_calls"] += 1
        self.record_metric("dense_retrieval_ms", dense_ms, {"count": dense_count})
        self.record_metric("sparse_retrieval_ms", sparse_ms, {"count": sparse_count})

    def record_reasoning(self, agent_ms: float, model: str):
        self._phase_times["agent_reasoning"] += agent_ms
        self._counters["reasoning_calls"] += 1
        self.record_metric("agent_reasoning_ms", agent_ms, {"model": model})

    def record_validation(self, validation_score: float, rule_count: int):
        self.record_metric("validation_score", validation_score, {"rule_count": rule_count})

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "total_calls": sum(self._counters.values()),
            "phase_times": dict(self._phase_times),
            "phase_counts": dict(self._counters),
            "phase_averages": {
                k: round(v / max(self._counters.get(k, 1), 1), 3)
                for k, v in self._phase_times.items()
            },
        }
        if self._metrics:
            latest = self._metrics[-50:]
            summary["recent_metrics"] = latest
        return summary

    def reset(self):
        self._metrics.clear()
        self._phase_times.clear()
        self._counters.clear()
        self._start_times.clear()

_benchmark_instance = None

def get_benchmark_collector() -> BenchmarkCollector:
    global _benchmark_instance
    if _benchmark_instance is None:
        _benchmark_instance = BenchmarkCollector()
    return _benchmark_instance