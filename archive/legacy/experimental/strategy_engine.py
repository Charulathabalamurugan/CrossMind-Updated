from typing import Any, Dict, List


class StrategyEngine:
    """Standalone strategy host used by runtime and diagnostics."""

    def __init__(self):
        self._versions: List[str] = ["baseline"]

    def select_plan(self, query: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        metadata = metadata or {}
        text = (query or "").lower()
        if any(term in text for term in ["battery", "solar", "energy", "market", "portfolio"]):
            return {"plan": "deep", "budget_tokens": 6000, "model": "multi-agent"}
        return {"plan": "medium", "budget_tokens": 3500, "model": "hybrid"}

    def get_version(self) -> str:
        return self._versions[-1]


def get_strategy_engine() -> StrategyEngine:
    return StrategyEngine()


__all__ = ["StrategyEngine", "get_strategy_engine"]
