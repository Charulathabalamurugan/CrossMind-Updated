from typing import Any, Dict, List


class OPAEnforcer:
    def __init__(self):
        self.policies: List[str] = []

    def add_policy(self, policy: str):
        self.policies.append(policy)

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "allowed": True,
            "policies": list(self.policies),
            "input": input_data,
        }


def get_opa_enforcer() -> OPAEnforcer:
    return OPAEnforcer()


__all__ = ["OPAEnforcer", "get_opa_enforcer"]
