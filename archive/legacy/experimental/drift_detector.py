from typing import Any, Dict, List


class DriftDetector:
    """Tracks distribution drift across observed evidence signals."""

    def detect(self, observed: List[float], baseline: List[float] | None = None) -> Dict[str, Any]:
        baseline = baseline or observed
        if not observed:
            return {"drift_score": 0.0, "status": "stable"}
        mean_observed = sum(observed) / len(observed)
        mean_baseline = sum(baseline) / len(baseline) if baseline else mean_observed
        drift_score = abs(mean_observed - mean_baseline)
        return {"drift_score": round(drift_score, 4), "status": "drift" if drift_score > 0.25 else "stable"}


def get_drift_detector() -> DriftDetector:
    return DriftDetector()


__all__ = ["DriftDetector", "get_drift_detector"]
