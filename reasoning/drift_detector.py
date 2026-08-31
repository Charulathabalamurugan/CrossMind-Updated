from typing import Any, Dict, List


class DriftDetector:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def detect_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        self.history.append({"baseline": baseline, "current": current})
        return {"drift_score": 0.0, "status": "stable", "baseline": baseline, "current": current}


def get_drift_detector() -> DriftDetector:
    return DriftDetector()


__all__ = ["DriftDetector", "get_drift_detector"]
