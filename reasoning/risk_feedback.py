import time
import logging
import threading
from typing import Dict, Any, List, Optional
from ingestion.active_learning import get_active_learning_engine

logger = logging.getLogger("crossmind.risk_feedback")

class RiskAssessor:
    @staticmethod
    def assess(query: str, doc_id: str, score: float, evidence_domains: List[str]) -> Dict[str, Any]:
        risk_factors = []
        base_risk = 0.1
        if score < 0.5:
            risk_factors.append("low_relevance_score")
            base_risk += 0.2
        if len(evidence_domains) > 2:
            risk_factors.append("multi_domain_complexity")
            base_risk += 0.1
        if len(query.split()) < 5:
            risk_factors.append("underspecified_query")
            base_risk += 0.15
        risk_score = min(1.0, base_risk)
        action = "accept"
        if risk_score > 0.6:
            action = "quarantine"
        elif risk_score > 0.35:
            action = "review"
        return {
            "risk_score": round(risk_score, 3),
            "risk_factors": risk_factors,
            "action": action,
            "confidence_adjusted": round(score * (1.0 - risk_score * 0.5), 3),
        }

class RiskControlledFeedbackEngine:
    def __init__(self):
        self.active_learning = get_active_learning_engine()
        self._feedback_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def submit_feedback(self, query: str, doc_id: str, score: float, user_role: str, evidence_domains: List[str] = None) -> Dict[str, Any]:
        assessment = RiskAssessor.assess(query, doc_id, score, evidence_domains or [])
        entry = {
            "query": query,
            "doc_id": doc_id,
            "raw_score": score,
            "user_role": user_role,
            "timestamp": time.time(),
            "risk_assessment": assessment,
        }
        if assessment["action"] == "accept":
            self.active_learning.record_feedback(query, doc_id, assessment["confidence_adjusted"], user_role)
            entry["status"] = "accepted"
        elif assessment["action"] == "review":
            entry["status"] = "pending_review"
        else:
            entry["status"] = "quarantined"
        with self._lock:
            self._feedback_history.append(entry)
        return entry

    def get_risk_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._feedback_history)
            if total == 0:
                return {"total": 0, "actions": {}}
            actions = {}
            for entry in self._feedback_history:
                a = entry.get("risk_assessment", {}).get("action", "unknown")
                actions[a] = actions.get(a, 0) + 1
            avg_risk = sum(e.get("risk_assessment", {}).get("risk_score", 0.0) for e in self._feedback_history) / total
            return {
                "total": total,
                "actions": actions,
                "average_risk_score": round(avg_risk, 3),
            }

_risk_feedback_instance: Optional[RiskControlledFeedbackEngine] = None

def get_risk_feedback_engine() -> RiskControlledFeedbackEngine:
    global _risk_feedback_instance
    if _risk_feedback_instance is None:
        _risk_feedback_instance = RiskControlledFeedbackEngine()
    return _risk_feedback_instance
