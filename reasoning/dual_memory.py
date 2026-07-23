import time
import logging
import threading
from typing import Dict, Any, List, Optional
from reasoning.memory_service import get_memory_service

logger = logging.getLogger("crossmind.dual_memory")

class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.interactions: List[Dict[str, Any]] = []
        self.context_window: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def add_interaction(self, query: str, result: Dict[str, Any]):
        with self._lock:
            self.last_accessed = time.time()
            self.interactions.append({
                "timestamp": time.time(),
                "query": query,
                "result_summary": result.get("agent_reasoning", {}).get("output_text", "")[:200],
                "confidence": result.get("confidence_calibration", {}).get("calibrated_confidence", 0.0),
                "decision": result.get("confidence_calibration", {}).get("decision", "unknown"),
            })
            if len(self.interactions) > 20:
                self.interactions.pop(0)
            self.context_window = {
                "recent_queries": [i["query"] for i in self.interactions[-5:]],
                "recent_decisions": [i["decision"] for i in self.interactions[-5:]],
                "session_duration_s": round(time.time() - self.created_at, 1),
            }

    def get_context(self) -> Dict[str, Any]:
        with self._lock:
            self.last_accessed = time.time()
            return dict(self.context_window)

class DualMemoryArchitecture:
    def __init__(self):
        self.long_term = get_memory_service()
        self.sessions: Dict[str, SessionMemory] = {}
        self._lock = threading.Lock()
        self._persona_profiles: Dict[str, Dict[str, Any]] = {}

    def get_or_create_session(self, session_id: str, user_role: str = "researcher") -> SessionMemory:
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMemory(session_id)
            self.sessions[session_id].last_accessed = time.time()
            return self.sessions[session_id]

    def record_interaction(self, session_id: str, query: str, result: Dict[str, Any], user_role: str):
        session = self.get_or_create_session(session_id, user_role)
        session.add_interaction(query, result)
        self.long_term.add_interaction(query, result)
        self._update_persona(session_id, query, result)

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        long_term_summary = self.long_term.get_summary()
        return {
            "session_id": session_id,
            "session_memory": session.get_context(),
            "long_term_profile": long_term_summary,
        }

    def _update_persona(self, session_id: str, query: str, result: Dict[str, Any]):
        with self._lock:
            if session_id not in self._persona_profiles:
                self._persona_profiles[session_id] = {
                    "cognitive_style": "Analytical & Exploratory",
                    "preferred_domains": [],
                    "safety_focus_level": "Balanced",
                    "interaction_count": 0,
                    "confidence_history": [],
                }
            profile = self._persona_profiles[session_id]
            profile["interaction_count"] += 1
            domains = result.get("pre_filter", {}).get("detected_domains", [])
            for domain in domains:
                if domain not in profile["preferred_domains"]:
                    profile["preferred_domains"].append(domain)
            conf = result.get("confidence_calibration", {}).get("calibrated_confidence", 0.0)
            profile["confidence_history"].append(conf)
            if len(profile["confidence_history"]) > 10:
                profile["confidence_history"].pop(0)
            avg_conf = sum(profile["confidence_history"]) / len(profile["confidence_history"])
            if avg_conf < 0.5:
                profile["safety_focus_level"] = "Rigorous"
            elif avg_conf > 0.75:
                profile["safety_focus_level"] = "Balanced"

    def get_persona(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._persona_profiles.get(session_id, {
                "cognitive_style": "Analytical & Exploratory",
                "preferred_domains": [],
                "safety_focus_level": "Balanced",
                "interaction_count": 0,
            })

    def cleanup_stale_sessions(self, max_age_s: int = 3600):
        cutoff = time.time() - max_age_s
        with self._lock:
            stale = [sid for sid, session in self.sessions.items() if session.last_accessed < cutoff]
            for sid in stale:
                del self.sessions[sid]

_dual_memory_instance: Optional[DualMemoryArchitecture] = None

def get_dual_memory() -> DualMemoryArchitecture:
    global _dual_memory_instance
    if _dual_memory_instance is None:
        _dual_memory_instance = DualMemoryArchitecture()
    return _dual_memory_instance
