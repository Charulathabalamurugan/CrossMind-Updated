from typing import Any, Dict, Optional


class MLFlowRegistry:
    """A minimal model-registry stub for local experiment tracking."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}

    def register_model(self, model_name: str, version: str, metrics: Optional[Dict[str, Any]] = None):
        self.models[model_name] = {"version": version, "metrics": metrics or {}}
        return self.models[model_name]

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_name)


def get_model_registry() -> MLFlowRegistry:
    return MLFlowRegistry()


__all__ = ["MLFlowRegistry", "get_model_registry"]
