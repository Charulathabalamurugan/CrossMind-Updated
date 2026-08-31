from typing import Any, Dict, List


class EnhancedAbductiveEngine:
    """Evidence-first abductive reasoning engine with compatible public API."""

    def perform_abductive_reasoning(
        self,
        query: str,
        retrieved_evidence: List[Dict[str, Any]],
        filter_metadata: Dict[str, Any],
        validator: Any = None,
    ) -> Dict[str, Any]:
        evidence_count = len(retrieved_evidence)
        hypotheses = []
        for item in retrieved_evidence[:5]:
            payload = item.get("payload") or {}
            title = payload.get("title") or "Untitled evidence"
            score = float(item.get("score", 0.0))
            hypotheses.append({
                "title": title,
                "support": round(score, 3),
                "explanation": f"This evidence supports a likely explanation for '{query}' based on the retrieved context.",
            })

        best_hypothesis = hypotheses[0] if hypotheses else {"title": "general evidence", "support": 0.0, "explanation": "No direct evidence matched the query."}
        return {
            "query": query,
            "hypothesis_count": len(hypotheses),
            "best_hypothesis": best_hypothesis,
            "evidence_count": evidence_count,
            "status": "ok",
            "reasoning_mode": "enhanced_abductive",
        }


_engine_instance: EnhancedAbductiveEngine | None = None


def get_enhanced_abductive_engine() -> EnhancedAbductiveEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EnhancedAbductiveEngine()
    return _engine_instance


__all__ = ["EnhancedAbductiveEngine", "get_enhanced_abductive_engine"]
