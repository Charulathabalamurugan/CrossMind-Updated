import logging
from typing import List, Dict, Any, Optional, Tuple
from config import settings

logger = logging.getLogger("crossmind.wfa")

class WFANode:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.transitions: Dict[str, "WFANode"] = {}

    def add_transition(self, symbol: str, target: "WFANode", weight: float = 1.0):
        self.transitions[symbol] = (target, weight)

    def get_transition(self, symbol: str) -> Optional[Tuple["WFANode", float]]:
        return self.transitions.get(symbol)

class WFAEngine:
    def __init__(self):
        self.start = WFANode("start")
        self._accepting: Dict[str, WFANode] = {}
        self._built = False
        self._build_default()

    def _build_default(self):
        q_factual = WFANode("factual", weight=1.0)
        q_complex = WFANode("complex", weight=0.8)
        q_causal = WFANode("causal", weight=0.6)
        q_cross_domain = WFANode("cross_domain", weight=0.9)
        high_conf = WFANode("high_conf", weight=0.95)
        mid_conf = WFANode("mid_conf", weight=0.7)
        low_conf = WFANode("low_conf", weight=0.4)

        self.start.add_transition("factual", q_factual, 1.0)
        self.start.add_transition("complex", q_complex, 0.8)
        self.start.add_transition("causal", q_causal, 0.6)
        self.start.add_transition("cross_domain", q_cross_domain, 0.9)

        q_factual.add_transition("high_conf", high_conf, 0.95)
        q_factual.add_transition("mid_conf", mid_conf, 0.7)
        q_factual.add_transition("low_conf", low_conf, 0.4)

        q_complex.add_transition("high_conf", high_conf, 0.85)
        q_complex.add_transition("mid_conf", mid_conf, 0.6)
        q_complex.add_transition("low_conf", low_conf, 0.3)

        q_causal.add_transition("high_conf", high_conf, 0.75)
        q_causal.add_transition("mid_conf", mid_conf, 0.5)
        q_causal.add_transition("low_conf", low_conf, 0.2)

        q_cross_domain.add_transition("high_conf", high_conf, 0.9)
        q_cross_domain.add_transition("mid_conf", mid_conf, 0.65)
        q_cross_domain.add_transition("low_conf", low_conf, 0.35)

        self._accepting = {
            high_conf.name: high_conf,
            mid_conf.name: mid_conf,
            low_conf.name: low_conf,
        }
        self._built = True

    def classify_query(self, query_type: str, confidence: float) -> Tuple[str, float]:
        conf_level = "high_conf" if confidence >= 0.85 else ("mid_conf" if confidence >= 0.5 else "low_conf")
        node = self.start
        step1 = node.get_transition(query_type)
        if step1:
            node, w1 = step1
        else:
            return "low_conf", 0.3
        step2 = node.get_transition(conf_level)
        if step2:
            node, w2 = step2
        else:
            return "low_conf", 0.3
        final_weight = w1 * w2
        return node.name, round(final_weight, 3)

    def fast_path_decision(self, query_type: str, confidence: float, domain_count: int) -> Dict[str, Any]:
        wfa_result, final_weight = self.classify_query(query_type, confidence)
        use_fast_path = (
            wfa_result == "high_conf"
            and domain_count <= 2
            and confidence >= 0.85
        )
        return {
            "use_fast_path": use_fast_path,
            "wfa_result": wfa_result,
            "final_weight": final_weight,
            "decision": "decision_tree" if use_fast_path else "escalate_to_graphrag",
            "reason": (
                "High confidence, single domain, WFA fast path accepted"
                if use_fast_path
                else f"WFA result={wfa_result}, weight={final_weight}, domains={domain_count}"
            ),
        }

_wfa_instance = None

def get_wfa_engine() -> WFAEngine:
    global _wfa_instance
    if _wfa_instance is None:
        _wfa_instance = WFAEngine()
    return _wfa_instance
