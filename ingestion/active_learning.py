import threading
import logging
import time
from typing import Dict, Any, List, Optional
from config import settings

logger = logging.getLogger("crossmind.active_learning")

class ActiveLearningEngine:
    def __init__(self):
        self._feedback: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_retrain = time.time()
        self.enabled = settings.ACTIVE_LEARNING_ENABLED

    def record_feedback(self, query: str, doc_id: str, score: float, user_role: str):
        if not self.enabled:
            return
        with self._lock:
            self._feedback.append({
                "query": query,
                "doc_id": doc_id,
                "score": score,
                "user_role": user_role,
                "timestamp": time.time(),
            })

    def get_feedback_count(self) -> int:
        with self._lock:
            return len(self._feedback)

    def should_retrain(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            count = len(self._feedback)
            elapsed = time.time() - self._last_retrain
            return count >= settings.ACTIVE_LEARNING_MIN_FEEDBACK and elapsed >= settings.ACTIVE_LEARNING_RETRAIN_INTERVAL

    def retrain(self, pipeline):
        if not self.should_retrain():
            return False
        logger.info("Active learning retrain triggered.")
        self._last_retrain = time.time()
        return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "feedback_count": len(self._feedback),
                "last_retrain": self._last_retrain,
            }

_active_learning_instance: Optional[ActiveLearningEngine] = None

def get_active_learning_engine() -> ActiveLearningEngine:
    global _active_learning_instance
    if _active_learning_instance is None:
        _active_learning_instance = ActiveLearningEngine()
    return _active_learning_instance
