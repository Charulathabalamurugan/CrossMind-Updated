from typing import Any, Dict


class OPAEnforcer:
    """Policy enforcement stub that evaluates a simple allowlist."""

    def evaluate(self, query: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        metadata = metadata or {}
        allowed = metadata.get("user_role", "researcher") in {"admin", "analyst", "researcher"}
        return {"allowed": allowed, "policy": "default_research_policy"}


def get_opa_enforcer() -> OPAEnforcer:
    return OPAEnforcer()


__all__ = ["OPAEnforcer", "get_opa_enforcer"]
