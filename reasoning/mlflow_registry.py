from typing import Any, Dict, Optional


class MLFlowRegistry:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    def register(self, model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.models[model_name] = payload
        return payload

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_name)


def get_model_registry() -> MLFlowRegistry:
    return MLFlowRegistry()


__all__ = ["MLFlowRegistry", "get_model_registry"]
