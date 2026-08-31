import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DLDB:
    """Lightweight persistent audit/feedback store used for local runtime compatibility."""

    def __init__(self, path: str = "/tmp/crossmind_dldb"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        state_file = self.path / "dldb.json"
        if state_file.exists():
            try:
                self._events = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                self._events = []

    def _save(self):
        state_file = self.path / "dldb.json"
        state_file.write_text(json.dumps(self._events, indent=2), encoding="utf-8")

    def record_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "payload": payload or {},
        }
        self._events.append(entry)
        self._save()
        return entry

    def list_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._events[-limit:]


_dldb_instance: Optional[DLDB] = None


def get_dldb() -> DLDB:
    global _dldb_instance
    if _dldb_instance is None:
        _dldb_instance = DLDB(path="/tmp/crossmind_dldb")
    return _dldb_instance


__all__ = ["DLDB", "get_dldb"]
