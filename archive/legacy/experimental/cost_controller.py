class CostController:
    def __init__(self):
        self._history = []

    def track_query(self, query: str, execution_mode: str, budget_tokens: int, estimate_cost: float):
        result = {
            "query": query,
            "execution_mode": execution_mode,
            "budget_tokens": budget_tokens,
            "estimated_cost": estimate_cost,
        }
        self._history.append(result)
        return result

    def enforce_budget(self, query_cost: float, max_cost: float = 0.25):
        within_budget = query_cost <= max_cost
        return {
            "within_budget": within_budget,
            "current_cost": query_cost,
            "max_budget": max_cost,
            "decision": "approved" if within_budget else "blocked",
        }


def get_cost_controller() -> CostController:
    return CostController()


__all__ = ["CostController", "get_cost_controller"]
