import time
import logging
from typing import Any, Dict, List

logger = logging.getLogger("crossmind.rule_updater")

class RuleUpdater:
    def __init__(self):
        self._custom_rules: List[Dict[str, Any]] = []
        self._updated_at: float = 0.0

    def add_rule(self, rule: Dict[str, Any]) -> bool:
        rule_id = rule.get("rule_id")
        if not rule_id:
            logger.warning("Rule must have a rule_id.")
            return False
        for existing in self._custom_rules:
            if existing.get("rule_id") == rule_id:
                logger.warning(f"Rule {rule_id} already exists; updating.")
                existing.update(rule)
                self._updated_at = time.time()
                return True
        self._custom_rules.append(rule)
        self._updated_at = time.time()
        logger.info(f"Rule {rule_id} added.")
        return True

    def remove_rule(self, rule_id: str) -> bool:
        for i, rule in enumerate(self._custom_rules):
            if rule.get("rule_id") == rule_id:
                self._custom_rules.pop(i)
                self._updated_at = time.time()
                logger.info(f"Rule {rule_id} removed.")
                return True
        logger.warning(f"Rule {rule_id} not found for removal.")
        return False

    def get_rules(self) -> List[Dict[str, Any]]:
        return list(self._custom_rules)

    def get_all_rules(self) -> List[Dict[str, Any]]:
        return list(self._custom_rules)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "custom_rule_count": len(self._custom_rules),
            "last_updated_at": self._updated_at,
            "rules": self._custom_rules,
        }

_rule_updater_instance = None

def get_rule_updater() -> RuleUpdater:
    global _rule_updater_instance
    if _rule_updater_instance is None:
        _rule_updater_instance = RuleUpdater()
    return _rule_updater_instance