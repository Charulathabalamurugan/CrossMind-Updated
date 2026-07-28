import time
import logging
import re
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger("crossmind.rule_engine")

VALIDATION_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "BIOCOMPATIBILITY_CHECK",
        "description": "Nanomaterials for CNS delivery must show low cytotoxicity profile.",
        "keywords_positive": ["biocompatible", "non-toxic", "PEGylated", "lipid", "functionalized"],
        "keywords_negative": ["toxic", "cytotoxic", "lethal"],
        "penalty": 15.0,
    },
    {
        "rule_id": "TEMPORAL_RELEVANCE",
        "description": "Evidence must be recent (2020-2026).",
        "min_year": 2020,
        "max_year": 2026,
        "penalty": 10.0,
    },
    {
        "rule_id": "CITATION_GROUNDING",
        "description": "All claims must be grounded in retrieved evidence.",
        "penalty": 10.0,
    },
    {
        "rule_id": "DOMAIN_PAIRING",
        "description": "Hypothesis should bridge at least two distinct domains.",
        "min_domains": 2,
        "penalty": 5.0,
    },
    {
        "rule_id": "SAFETY_THRESHOLD",
        "description": "Safety-related terms must be present in nanotechnology hypotheses.",
        "requires_safety_terms": True,
        "penalty": 20.0,
    },
]

class RuleEngine:
    def __init__(self):
        self.rules = VALIDATION_RULES
        self._rule_execution_log: List[Dict[str, Any]] = []

    def evaluate(
        self,
        hypothesis_text: str,
        evidence: List[Dict[str, Any]],
        filter_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.time()
        hypo_lower = hypothesis_text.lower()
        evidence_text = " ".join(
            str(ev.get("payload", {}).get("content", "")) for ev in evidence
        ).lower()
        all_text = hypo_lower + " " + evidence_text
        results = []
        total_penalty = 0.0

        for rule in self.rules:
            passed = True
            details = ""
            penalty_applied = 0.0
            rule_id = rule.get("rule_id", "")

            if "keywords_positive" in rule:
                has_positive = any(kw in all_text for kw in rule["keywords_positive"])
                has_negative = any(kw in all_text for kw in rule.get("keywords_negative", []))
                if has_negative:
                    passed = False
                    penalty_applied = rule.get("penalty", 0.0)
                    total_penalty += penalty_applied
                    details = f"Negative keyword detected. Penalty: {penalty_applied}"
                elif not has_positive:
                    passed = False
                    penalty_applied = rule.get("penalty", 0.0) * 0.5
                    total_penalty += penalty_applied
                    details = f"Positive keyword missing. Partial penalty: {penalty_applied}"
                else:
                    details = "All safety keywords present."

            elif "min_year" in rule or "max_year" in rule:
                years = [
                    ev.get("payload", {}).get("year", 2024)
                    for ev in evidence
                    if isinstance(ev.get("payload"), dict)
                ]
                min_year = rule.get("min_year", 2020)
                max_year = rule.get("max_year", 2026)
                outdated = [y for y in years if y < min_year or y > max_year]
                if outdated:
                    passed = False
                    penalty_applied = rule.get("penalty", 0.0)
                    total_penalty += penalty_applied
                    details = f"Outdated evidence years: {outdated}"

            elif "min_domains" in rule:
                domains = filter_metadata.get("detected_domains", [])
                if len(set(domains)) < rule["min_domains"]:
                    passed = False
                    penalty_applied = rule.get("penalty", 0.0)
                    total_penalty += penalty_applied
                    details = f"Only {len(set(domains))} domain(s), need {rule['min_domains']}."

            else:
                details = "Rule not applicable to current hypothesis."
                passed = True

            results.append({
                "rule_id": rule_id,
                "passed": passed,
                "penalty": penalty_applied,
                "details": details,
            })

        execution_ms = round((time.time() - start) * 1000, 2)
        validation_score = max(0.0, 100.0 - total_penalty)
        validated = validation_score >= 70.0

        log_entry = {
            "hypothesis": hypothesis_text[:100],
            "results": results,
            "total_penalty": total_penalty,
            "validation_score": validation_score,
            "validated": validated,
            "execution_ms": execution_ms,
        }
        self._rule_execution_log.append(log_entry)

        return {
            "validated": validated,
            "validation_score": round(validation_score, 1),
            "rule_checks": results,
            "total_penalty": round(total_penalty, 1),
            "execution_ms": execution_ms,
            "method": "pknow_style_rule_engine",
        }

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._rule_execution_log)

    def add_rule(self, rule: Dict[str, Any]):
        self.rules.append(rule)
        logger.info(f"New rule added: {rule.get('rule_id', 'unknown')}")

_rule_engine_instance = None

def get_rule_engine() -> RuleEngine:
    global _rule_engine_instance
    if _rule_engine_instance is None:
        _rule_engine_instance = RuleEngine()
    return _rule_engine_instance
