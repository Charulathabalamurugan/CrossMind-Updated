import json
import os
import logging
import time
from typing import Dict, Any, List, Optional
from config import settings

logger = logging.getLogger("crossmind.mlflow")

REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "./data/registry.json")

class ModelRegistry:
    def __init__(self):
        self.enabled = settings.MODEL_REGISTRY_ENABLED
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if not self.enabled:
            return
        try:
            if os.path.exists(REGISTRY_PATH):
                with open(REGISTRY_PATH, "r") as fh:
                    self._registry = json.load(fh)
                logger.info(f"Loaded model registry with {len(self._registry)} entries.")
        except Exception as exc:
            logger.error(f"Registry load failed: {exc}")

    def _save_to_disk(self):
        if not self.enabled:
            return
        try:
            os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
            with open(REGISTRY_PATH, "w") as fh:
                json.dump(self._registry, fh, indent=2)
        except Exception as exc:
            logger.error(f"Registry save failed: {exc}")

    def register_model(self, name: str, version: str, artifacts: Dict[str, Any]) -> str:
        key = f"{name}:{version}"
        entry = {
            "name": name,
            "version": version,
            "artifact": artifacts,
            "registered_at": time.time(),
            "registered_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self._registry[key] = entry
        self._save_to_disk()
        return key

    def get_model(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        if version == "latest":
            matches = {k: v for k, v in self._registry.items() if k.startswith(f"{name}:")}
            if not matches:
                return None
            return max(matches.values(), key=lambda x: x.get("registered_at", 0))
        return self._registry.get(f"{name}:{version}")

    def list_versions(self, name: str) -> List[str]:
        return [k.split(":")[1] for k in self._registry if k.startswith(f"{name}:")]

    def get_rollback(self, name: str, rollback_to_version: str) -> bool:
        version = self.get_model(name, rollback_to_version)
        if version:
            current_key = f"{name}:{self.list_versions(name)[-1]}"
            self._registry[current_key] = version
            self._save_to_disk()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_models": len(self._registry),
            "names": list(set(k.split(":")[0] for k in self._registry.keys())),
        }

registry_instance = None

def get_model_registry() -> ModelRegistry:
    global registry_instance
    if registry_instance is None:
        registry_instance = ModelRegistry()
    return registry_instance
