import time
import logging
import threading
from typing import Dict, Any, List, Optional
from config import settings
from collections import defaultdict

logger = logging.getLogger("crossmind.drift")

class DriftDetector:
    def __init__(self):
        self.enabled = settings.DRIFT_DETECTION_ENABLED
        self._baseline: Dict[str, List[float]] = defaultdict(list)
        self._current: Dict[str, List[float]] = defaultdict(list)
        self._last_check = 0.0
        self._lock = threading.Lock()
        self._drift_log: List[Dict[str, Any]] = []

    def record_embedding(self, domain: str, embedding_norm: float):
        with self._lock:
            self._current[domain].append(embedding_norm)

    def set_baseline(self, domain: str, norms: List[float]):
        with self._lock:
            self._baseline[domain] = norms

    def check_drift(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"drift_detected": False, "check_interval_hours": settings.DRIFT_DETECTION_INTERVAL_HOURS}

        elapsed_hours = (time.time() - self._last_check) / 3600 if self._last_check > 0 else float("inf")
        if elapsed_hours < settings.DRIFT_DETECTION_INTERVAL_HOURS:
            return {"drift_detected": False, "next_check_in_hours": round(settings.DRIFT_DETECTION_INTERVAL_HOURS - elapsed_hours, 1)}

        self._last_check = time.time()
        drift_results = {}
        drift_detected = False

        for domain in set(list(self._baseline.keys()) + list(self._current.keys())):
            baseline = self._baseline.get(domain, [])
            current = self._current.get(domain, [])
            if not baseline or not current:
                continue

            try:
                import statistics
                baseline_mean = statistics.mean(baseline)
                current_mean = statistics.mean(current)
                baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0
                current_std = statistics.stdev(current) if len(current) > 1 else 0
                ks_like_stat = abs(current_mean - baseline_mean) / max(baseline_std, 1e-10)
                is_drift = ks_like_stat > settings.DRIFT_KS_THRESHOLD
                accuracy_impact = abs(current_mean - baseline_mean)
                is_accuracy_drift = accuracy_impact > settings.DRIFT_ACCURACY_THRESHOLD

                drift_results[domain] = {
                    "ks_like_statistic": round(ks_like_stat, 4),
                    "baseline_mean": round(baseline_mean, 4),
                    "current_mean": round(current_mean, 4),
                    "baseline_std": round(baseline_std, 4),
                    "current_std": round(current_std, 4),
                    "drift_detected": is_drift,
                    "accuracy_impact": round(accuracy_impact, 4),
                    "accuracy_drift": is_accuracy_drift,
                }
                if is_drift or is_accuracy_drift:
                    drift_detected = True
            except Exception as exc:
                logger.warning(f"Drift check failed for {domain}: {exc}")
                drift_results[domain] = {"error": str(exc)}

        log_entry = {
            "timestamp": time.time(),
            "drift_detected": drift_detected,
            "results": drift_results,
        }
        self._drift_log.append(log_entry)
        self._current.clear()

        return {
            "drift_detected": drift_detected,
            "check_interval_hours": settings.DRIFT_DETECTION_INTERVAL_HOURS,
            "results": drift_results,
        }

    def needs_retraining(self) -> bool:
        check = self.check_drift()
        if check.get("drift_detected"):
            return True
        stats = get_dldb().get_stats() if 'dldb_instance' in dir() or True else {"total_feedback": 0}
        return False

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._drift_log)

_drift_instance = None

def get_drift_detector() -> DriftDetector:
    global _drift_instance
    if _drift_instance is None:
        _drift_instance = DriftDetector()
    return _drift_instance
