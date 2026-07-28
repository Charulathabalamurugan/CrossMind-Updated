import logging
import time
from typing import List, Dict, Any, Optional, Set
from config import settings

logger = logging.getLogger("crossmind.datalog")

class DatalogEngine:
    def __init__(self):
        self.enabled = settings.DATALOG_RULES_ENABLED
        self.rules: List[Dict[str, Any]] = []
        self.facts: Dict[str, Set[str]] = {}
        self.rule_log: List[Dict[str, Any]] = []
        self._load_default_rules()

    def _load_default_rules(self):
        self.rules = [
            {
                "rule_id": "BIOCOMPATIBILITY_RULE",
                "head": "safe_for_cns_delivery(X)",
                "body": ["nanocarrier(X)", "biocompatible(X)", "low_cytotoxicity(X)"],
                "severity": "high",
                "description": "X is safe for CNS delivery if it is a nanocarrier with biocompatibility and low cytotoxicity.",
            },
            {
                "rule_id": "TEMPORAL_RELEVANCE_RULE",
                "head": "current_evidence(X, Y)",
                "body": ["published_after(Y, 2020)", "supports(X, Y)"],
                "severity": "medium",
                "description": "Evidence Y is current and supports X if Y was published after 2020.",
            },
            {
                "rule_id": "CITATION_GROUNDING_RULE",
                "head": "grounded_claim(Z)",
                "body": ["claim(Z)", "supported_by_evidence(Z, Y)", "citation_exists(Y)"],
                "severity": "high",
                "description": "Claim Z is grounded if there exists cited evidence Y supporting it.",
            },
            {
                "rule_id": "DOMAIN_BRIDGE_RULE",
                "head": "cross_domain_bridge(A, B)",
                "body": ["domain(A, D1)", "domain(B, D2)", "D1 != D2", "shared_entity(A, B, E)"],
                "severity": "medium",
                "description": "A and B form a cross-domain bridge if they share an entity E but belong to different domains.",
            },
            {
                "rule_id": "THERAPEUTIC_VALIDITY_RULE",
                "head": "therapeutic_valid(X)",
                "body": ["drug_target(X, T)", "target_valid(T)", "delivery_route(X, R)", "route_approved(R)"],
                "severity": "critical",
                "description": "X is therapeutically valid if its drug target is valid and delivery route is approved.",
            },
        ]

    def assert_fact(self, predicate: str, args: tuple):
        key = f"{predicate}({','.join(str(a) for a in args)})"
        if predicate not in self.facts:
            self.facts[predicate] = set()
        self.facts[predicate].add(key)

    def retract_fact(self, predicate: str, args: tuple):
        key = f"{predicate}({','.join(str(a) for a in args)})"
        if predicate in self.facts and key in self.facts[predicate]:
            self.facts[predicate].discard(key)

    def query(self, goal: str) -> List[Dict[str, Any]]:
        results = []
        for rule in self.rules:
            head = rule["head"]
            if goal.startswith(head.split("(")[0]):
                bindings = self._resolve_rule(rule)
                results.extend(bindings)
        return results

    def _resolve_rule(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        body = rule.get("body", [])
        head_predicate = rule["head"].split("(")[0]
        head_args = rule["head"].split("(")[1].rstrip(")").split(",")
        
        if not body:
            return [{}]
        
        solutions = [{}]
        for literal in body:
            next_solutions = []
            pred = literal.split("(")[0]
            args = literal.split("(")[1].rstrip(")").split(",")
            
            for sol in solutions:
                grounded = self._ground_args(args, sol)
                if grounded is None:
                    continue
                fact_key = f"{pred}({grounded})"
                if pred in self.facts and fact_key in self.facts[pred]:
                    next_solutions.append(sol)
            solutions = next_solutions

        bindings = []
        for sol in solutions:
            binding = {}
            for i, arg in enumerate(head_args):
                if arg.startswith("?"):
                    binding[arg] = sol.get(f"_{arg}", arg)
            bindings.append(binding)
        return bindings

    def _ground_args(self, args: List[str], sol: Dict[str, str]) -> Optional[str]:
        grounded = []
        for arg in args:
            arg = arg.strip()
            if arg.startswith("?") and arg in sol:
                grounded.append(sol[arg])
            elif not arg.startswith("?"):
                grounded.append(arg)
            else:
                return None
        return ", ".join(grounded)

    def validate_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        for entity in entities:
            self.assert_fact("entity", (entity.get("entity", ""), entity.get("type", "unknown")))
            self.assert_fact("confidence", (entity.get("entity", ""), str(entity.get("confidence", 0.0))))
        
        results = []
        for rule in self.rules:
            matches = self.query(rule["head"])
            for match in matches:
                results.append({
                    "rule_id": rule["rule_id"],
                    "description": rule["description"],
                    "severity": rule["severity"],
                    "binding": match,
                    "passed": True,
                })
        
        log_entry = {
            "timestamp": time.time(),
            "entities_processed": len(entities),
            "rules_evaluated": len(self.rules),
            "rules_fired": len(results),
            "results": results,
        }
        self.rule_log.append(log_entry)
        
        passed = all(r["passed"] for r in results) if results else True
        return {
            "validated": passed,
            "results": results,
            "total_rules": len(self.rules),
            "fired_rules": len(results),
            "log_entry": log_entry,
        }

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self.rule_log)

    def add_rule(self, rule: Dict[str, Any]):
        self.rules.append(rule)
        logger.info(f"Datalog rule added: {rule.get('rule_id', 'unknown')}")

    def remove_rule(self, rule_id: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.get("rule_id") == rule_id:
                self.rules.pop(i)
                logger.info(f"Datalog rule removed: {rule_id}")
                return True
        return False

_datalog_instance = None

def get_datalog_engine() -> DatalogEngine:
    global _datalog_instance
    if _datalog_instance is None:
        _datalog_instance = DatalogEngine()
    return _datalog_instance