import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("crossmind.hypothesis_generator")

DECISION_TREE_RULES = [
    {
        "condition": lambda f: "nanoparticle" in str(f.get("detected_entities", [])).lower()
        or "nanomaterial" in str(f.get("detected_entities", [])).lower(),
        "domain": "nanotechnology",
        "template": "Nanocarrier-based delivery mechanism targeting {entity}.",
    },
    {
        "condition": lambda f: "protein" in str(f.get("detected_entities", [])).lower()
        or "biomarker" in str(f.get("detected_entities", [])).lower()
        or any(tag in str(f.get("detected_entities", [])).lower() for tag in ["Alzheimer", "beta-amyloid", "tau", "amyloid"]),
        "domain": "neuroscience",
        "template": "Protein aggregation pathway modulated by {entity}.",
    },
    {
        "condition": lambda f: "drug" in str(f.get("detected_entities", [])).lower()
        or "delivery" in str(f.get("detected_entities", [])).lower()
        or "therapeutic" in str(f.get("detected_entities", [])).lower(),
        "domain": "pharmacology",
        "template": "Therapeutic delivery mechanism for {entity}.",
    },
    {
        "condition": lambda f: "cross" in str(f.get("detected_domains", [])).lower()
        or len(f.get("detected_domains", [])) >= 2,
        "domain": "cross_domain",
        "template": "Cross-domain interaction between {entity_a} and {entity_b}.",
    },
]

def _get_filler(entities: List[str], domains: List[str]) -> Dict[str, str]:
    entity_a = entities[0] if entities else "the mechanism"
    entity_b = entities[1] if len(entities) > 1 else "the pathway"
    return {"entity": entity_a, "entity_a": entity_a, "entity_b": entity_b}

class HypothesisGenerator:
    def __init__(self):
        self.rule_count = len(DECISION_TREE_RULES)
        self._decision_history: List[Dict[str, Any]] = []

    def generate(self, filter_metadata: Dict[str, Any], evidence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        entities = filter_metadata.get("extracted_entities", [])
        domains = filter_metadata.get("detected_domains", [])
        applied_rule = None
        hypothesis = ""
        for rule in DECISION_TREE_RULES:
            try:
                if rule["condition"](filter_metadata):
                    filler = _get_filler(entities, domains)
                    hypothesis = rule["template"].format(**filler)
                    applied_rule = rule["condition"].__name__ if hasattr(rule["condition"], "__name__") else rule.get("domain")
                    break
            except Exception:
                continue
        if not hypothesis:
            filler = _get_filler(entities, domains)
            hypothesis = f"Cross-domain mechanism involving {filler['entity_a']} and {filler['entity_b']}."
            applied_rule = "default"
        domain = domains[0] if domains else "general"
        decision_tree_log = {
            "entities_evaluated": entities,
            "domains_detected": domains,
            "rule_matched": applied_rule,
            "template_used": hypothesis,
        }
        self._decision_history.append(decision_tree_log)
        return {
            "hypothesis": hypothesis,
            "domain": domain,
            "decision_tree_log": decision_tree_log,
            "confidence": 0.85 if evidence and len(evidence) > 0 else 0.5,
            "method": "decision_tree_v1",
        }

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._decision_history)

_hypothesis_gen_instance = None

def get_hypothesis_generator() -> HypothesisGenerator:
    global _hypothesis_gen_instance
    if _hypothesis_gen_instance is None:
        _hypothesis_gen_instance = HypothesisGenerator()
    return _hypothesis_gen_instance
