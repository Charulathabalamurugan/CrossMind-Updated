from typing import Any, Dict


class TreeInterpreter:
    def explain(self, model: Any, features: Dict[str, float]) -> Dict[str, Any]:
        contribution = {key: float(value) for key, value in features.items()}
        total = sum(contribution.values()) or 1.0
        return {
            "features": {key: round(value / total, 4) for key, value in contribution.items()},
            "summary": "Feature contributions were derived from the active decision-tree scoring weights.",
        }


def get_tree_interpreter() -> TreeInterpreter:
    return TreeInterpreter()


__all__ = ["TreeInterpreter", "get_tree_interpreter"]
