import time
import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional
from config import settings

logger = logging.getLogger("crossmind.feedback")

class FeedbackEntry:
    def __init__(
        self,
        query: str,
        doc_id: str,
        relevance_score: float,
        user_role: str,
        session_id: str = "default",
        risk_level: str = "low",
    ):
        self.query = query
        self.doc_id = doc_id
        self.relevance_score = relevance_score
        self.user_role = user_role
        self.session_id = session_id
        self.risk_level = risk_level
        self.timestamp = time.time()
        self.applied = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "relevance_score": self.relevance_score,
            "user_role": self.user_role,
            "session_id": self.session_id,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
            "applied": self.applied,
        }

class FeedbackCollector:
    def __init__(self):
        self._feedback: List[FeedbackEntry] = []
        self._lock = threading.Lock()
        self._retrain_threshold = settings.ACTIVE_LEARNING_MIN_FEEDBACK
        self._enabled = settings.ACTIVE_LEARNING_ENABLED

    def submit(
        self,
        query: str,
        doc_id: str,
        relevance_score: float,
        user_role: str = "researcher",
        session_id: str = "default",
        risk_level: str = "low",
    ) -> FeedbackEntry:
        entry = FeedbackEntry(query, doc_id, relevance_score, user_role, session_id, risk_level)
        with self._lock:
            self._feedback.append(entry)
        logger.info(
            f"Feedback submitted: query='{query[:50]}...', doc={doc_id}, score={relevance_score}, risk={risk_level}"
        )
        return entry

    def get_unapplied(self) -> List[FeedbackEntry]:
        with self._lock:
            return [e for e in self._feedback if not e.applied]

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._feedback]

    def count(self) -> int:
        with self._lock:
            return len(self._feedback)

    def mark_applied(self, doc_id: str):
        with self._lock:
            for entry in self._feedback:
                if entry.doc_id == doc_id:
                    entry.applied = True

    def should_retrain(self) -> bool:
        if not self._enabled:
            return False
        unapplied = len(self.get_unapplied())
        return unapplied >= self._retrain_threshold

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._feedback)
            applied = sum(1 for e in self._feedback if e.applied)
            by_risk = defaultdict(int)
            for e in self._feedback:
                by_risk[e.risk_level] += 1
            avg_score = (
                sum(e.relevance_score for e in self._feedback) / total
                if total > 0
                else 0.0
            )
            return {
                "total": total,
                "applied": applied,
                "unapplied": total - applied,
                "by_risk_level": dict(by_risk),
                "average_relevance": round(avg_score, 3),
                "should_retrain": self.should_retrain(),
            }

_feedback_collector_instance = None

def get_feedback_collector() -> FeedbackCollector:
    global _feedback_collector_instance
    if _feedback_collector_instance is None:
        _feedback_collector_instance = FeedbackCollector()
    return _feedback_collector_instance