from typing import Any, Dict


class ResourceAllocator:
    def allocate(self, query: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        metadata = metadata or {}
        return {
            "budget_tokens": 3500,
            "agent_count": metadata.get("agent_count", 2),
            "execution_mode": "deep" if "battery" in (query or "").lower() else "medium",
        }


def get_resource_allocator() -> ResourceAllocator:
    return ResourceAllocator()


__all__ = ["ResourceAllocator", "get_resource_allocator"]
