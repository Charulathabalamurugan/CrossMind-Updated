from typing import Any, Dict


class PriorityRouter:
    def route(self, query: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        metadata = metadata or {}
        role = metadata.get("user_role", "researcher")
        return {"route": "research", "priority": "normal", "user_role": role}


def get_priority_router() -> PriorityRouter:
    return PriorityRouter()


__all__ = ["PriorityRouter", "get_priority_router"]
