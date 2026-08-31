from typing import Any, Dict, List


class TreeInterpreter:
    def __init__(self):
        self.name = "treeinterpreter"

    def explain(self, feature_values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feature_values": feature_values,
            "contributions": {k: 1.0 for k in feature_values},
        }


def get_tree_interpreter() -> TreeInterpreter:
    return TreeInterpreter()


__all__ = ["TreeInterpreter", "get_tree_interpreter"]
