from typing import Any, Dict, List


class DatalogEngine:
    def __init__(self):
        self.rules: List[str] = []

    def add_rule(self, rule: str):
        self.rules.append(rule)

    def evaluate(self, facts: List[str]) -> Dict[str, Any]:
        return {
            "facts": facts,
            "rules": list(self.rules),
            "matches": len(facts),
            "valid": True,
        }


def get_datalog_engine() -> DatalogEngine:
    return DatalogEngine()


__all__ = ["DatalogEngine", "get_datalog_engine"]
