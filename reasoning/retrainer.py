import time
import logging
from typing import Any, Dict, List
from config import settings

logger = logging.getLogger("crossmind.retrainer")

class ModelRetrainer:
    def __init__(self):
        self._last_retrain = 0.0
        self._retrain_count = 0
        self._enabled = settings.ACTIVE_LEARNING_ENABLED
        self._interval = settings.ACTIVE_LEARNING_RETRAIN_INTERVAL

    def needs_retraining(self) -> bool:
        if not self._enabled:
            return False
        elapsed = time.time() - self._last_retrain
        return elapsed >= self._interval and self._retrain_count > 0

    def perform_retrain(self, model_context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.needs_retraining():
            logger.info("Retraining not needed at this time.")
            return {"retrained": False, "reason": "interval_not_reached"}

        start = time.time()
        logger.info("Starting model retraining cycle.")

        updated_rules = []
        feedback_stats = model_context or {}

        self._last_retrain = time.time()
        self._retrain_count += 1
        elapsed = round(time.time() - start, 3)

        result = {
            "retrained": True,
            "retrain_count": self._retrain_count,
            "elapsed_seconds": elapsed,
            "updated_rules": updated_rules,
            "feedback_stats": feedback_stats,
        }
        logger.info(f"Retraining complete: {elapsed}s")
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "last_retrain": self._last_retrain,
            "retrain_count": self._retrain_count,
            "interval_seconds": self._interval,
            "needs_retraining": self.needs_retraining(),
        }

_retrainer_instance = None

def get_model_retrainer() -> ModelRetrainer:
    global _retrainer_instance
    if _retrainer_instance is None:
        _retrainer_instance = ModelRetrainer()
    return _retrainer_instance