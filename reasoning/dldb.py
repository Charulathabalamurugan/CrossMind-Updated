import json
import os
import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from config import settings

logger = logging.getLogger("crossmind.dldb")

FEEDBACK_STORE = os.getenv("DLDB_PATH", "./data/dldb.json")

class DLDB:
    def __init__(self):
        self.enabled = settings.DLDB_ENABLED
        self._lock = threading.Lock()
        self._feedback: List[Dict[str, Any]] = []
        self._model_scores: Dict[str, float] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if not self.enabled:
            return
        try:
            if os.path.exists(FEEDBACK_STORE):
                with open(FEEDBACK_STORE, "r") as fh:
                    data = json.load(fh)
                    self._feedback = data.get("feedback", [])
                    self._model_scores = data.get("model_scores", {})
                logger.info(f"Loaded {len(self._feedback)} feedback records from DLDB.")
        except Exception as exc:
            logger.error(f"DLDB load failed: {exc}")

    def _save_to_disk(self):
        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(FEEDBACK_STORE), exist_ok=True)
            with open(FEEDBACK_STORE, "w") as fh:
                json.dump({
                    "feedback": self._feedback,
                    "model_scores": self._model_scores,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, fh, indent=2)
        except Exception as exc:
            logger.error(f"DLDB save failed: {exc}")

    def record_feedback(
        self,
        query: str,
        doc_id: str,
        relevance_score: float,
        user_role: str,
        model_version: str = "Yuuki-RxG-nano-v1",
    ) -> Dict[str, Any]:
        entry = {
            "id": f"fb_{len(self._feedback):06d}",
            "query": query,
            "doc_id": doc_id,
            "relevance_score": relevance_score,
            "user_role": user_role,
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._feedback.append(entry)
        self._save_to_disk()
        return entry

    def get_low_confidence_queries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        with self._lock:
            return [f for f in self._feedback if f.get("relevance_score", 0.0) < threshold]

    def compute_model_score(self, model_version: str = "Yuuki-RxG-nano-v1") -> float:
        with self._lock:
            relevant = [f for f in self._feedback if f.get("model_version") == model_version]
            if not relevant:
                return 0.0
            avg_score = sum(f.get("relevance_score", 0.0) for f in relevant) / len(relevant)
            self._model_scores[model_version] = avg_score
            return avg_score

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_feedback": len(self._feedback),
                "model_scores": self._model_scores,
                "low_confidence_count": len([f for f in self._feedback if f.get("relevance_score", 0.0) < 0.5]),
                "last_feedback": self._feedback[-1] if self._feedback else None,
            }

dldb_instance = None

def get_dldb() -> DLDB:
    global dldb_instance
    if dldb_instance is None:
        dldb_instance = DLDB()
    return dldb_instance
