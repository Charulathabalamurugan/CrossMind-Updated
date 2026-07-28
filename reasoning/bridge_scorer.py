import math
from typing import Any, Dict, List, Optional

logger = None


def compute_bridge_strength(
    paths: List[Dict[str, Any]],
    evidence_count: int,
    domain_count: int,
) -> Dict[str, Any]:
    if not paths:
        return {
            "bridge_strength": 0.0,
            "rating": "none",
            "path_count": 0,
            "domain_count": domain_count,
        }
    max_score = max((p.get("path_score", 0.0) for p in paths), default=0.0)
    mean_score = sum(p.get("path_score", 0.0) for p in paths) / len(paths)
    cross_domain_count = sum(1 for p in paths if p.get("cross_domain", False))
    novelty_ratio = cross_domain_count / max(len(paths), 1)
    strength = round(
        (0.5 * max_score + 0.3 * mean_score + 0.2 * novelty_ratio * 100),
        1,
    )
    if strength >= 75:
        rating = "strong"
    elif strength >= 50:
        rating = "moderate"
    elif strength >= 25:
        rating = "weak"
    else:
        rating = "none"
    return {
        "bridge_strength": strength,
        "rating": rating,
        "path_count": len(paths),
        "cross_domain_path_count": cross_domain_count,
        "domain_count": domain_count,
        "max_path_score": max_score,
        "mean_path_score": round(mean_score, 1),
    }

_bridge_scorer_instance = None


def get_bridge_scorer():
    global _bridge_scorer_instance
    if _bridge_scorer_instance is None:
        _bridge_scorer_instance = _BridgeScorer()
    return _bridge_scorer_instance


class _BridgeScorer:
    compute_bridge_strength = staticmethod(compute_bridge_strength)