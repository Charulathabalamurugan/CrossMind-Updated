from typing import Any, Dict, List


class ConflictDetector:
    """Detect contradictory evidence in retrieved document sets."""

    @staticmethod
    def detect_conflicts(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        for index, left in enumerate(evidence):
            for right in evidence[index + 1:]:
                left_text = str((left.get("payload") or {}).get("content", "")).lower()
                right_text = str((right.get("payload") or {}).get("content", "")).lower()
                if not left_text or not right_text:
                    continue
                left_terms = {term for term in ["increases", "decreases", "reduces", "improves", "worsens", "inhibits", "activates"] if term in left_text}
                right_terms = {term for term in ["increases", "decreases", "reduces", "improves", "worsens", "inhibits", "activates"] if term in right_text}
                overlap = left_terms & right_terms
                if overlap:
                    conflicts.append({
                        "source_id_1": left.get("id"),
                        "source_id_2": right.get("id"),
                        "conflict_type": "directional_disagreement",
                        "details": "Evidence contains opposite directional claims around the same mechanism.",
                    })
        return conflicts


__all__ = ["ConflictDetector"]
