"""
User profiles, saved queries, history, and per-user isolation.
"""
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
from config import settings

logger = logging.getLogger("crossmind.user_profiles")

# Prototype in-memory store; replace with PostgreSQL/Redis in production.
_user_profiles: Dict[str, Dict[str, Any]] = {}
_user_history: Dict[str, List[Dict[str, Any]]] = {}
_user_saved_queries: Dict[str, List[Dict[str, Any]]] = {}
_lock = threading.Lock()


class UserProfileService:
    @staticmethod
    def get_profile(user_id: str) -> Dict[str, Any]:
        with _lock:
            if user_id not in _user_profiles:
                _user_profiles[user_id] = {
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "preferences": {"theme": "light", "default_role": "viewer"},
                    "stats": {"queries": 0, "saved": 0, "feedback_count": 0},
                }
            return dict(_user_profiles[user_id])

    @staticmethod
    def update_preferences(user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        with _lock:
            profile = _user_profiles.setdefault(user_id, {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "preferences": {},
                "stats": {"queries": 0, "saved": 0, "feedback_count": 0},
            })
            profile["preferences"] = {**profile.get("preferences", {}), **preferences}
            return dict(profile)

    @staticmethod
    def record_query(user_id: str, query: str, result_summary: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:500],
            "result_summary": {
                "confidence": result_summary.get("confidence_calibration", {}).get("calibrated_confidence"),
                "discovery_score": result_summary.get("cross_domain_scoring", {}).get("overall_score"),
                "evidence_count": len(result_summary.get("retrieved_evidence", [])),
            },
        }
        with _lock:
            _user_history.setdefault(user_id, []).append(entry)
            profile = _user_profiles.setdefault(user_id, {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "preferences": {},
                "stats": {"queries": 0, "saved": 0, "feedback_count": 0},
            })
            profile["stats"]["queries"] = profile.get("stats", {}).get("queries", 0) + 1
        return entry

    @staticmethod
    def save_query(user_id: str, query: str, tags: List[str] = None, note: str = "") -> Dict[str, Any]:
        saved = {
            "saved_id": f"sq_{int(time.time() * 1000)}",
            "user_id": user_id,
            "query": query,
            "tags": tags or [],
            "note": note,
            "created_at": datetime.utcnow().isoformat(),
        }
        with _lock:
            _user_saved_queries.setdefault(user_id, []).append(saved)
            profile = _user_profiles.setdefault(user_id, {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "preferences": {},
                "stats": {"queries": 0, "saved": 0, "feedback_count": 0},
            })
            profile["stats"]["saved"] = profile.get("stats", {}).get("saved", 0) + 1
        return saved

    @staticmethod
    def get_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with _lock:
            return list(_user_history.get(user_id, []))[-limit:]

    @staticmethod
    def get_saved_queries(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with _lock:
            return list(_user_saved_queries.get(user_id, []))[-limit:]

    @staticmethod
    def delete_saved_query(user_id: str, saved_id: str) -> bool:
        with _lock:
            queries = _user_saved_queries.get(user_id, [])
            for i, q in enumerate(queries):
                if q.get("saved_id") == saved_id:
                    queries.pop(i)
                    return True
            return False
