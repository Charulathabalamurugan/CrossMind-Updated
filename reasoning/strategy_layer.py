import logging
import time
from typing import Any, Dict, List, Optional

from config import settings
from reasoning.query_classifier import get_query_classifier

logger = logging.getLogger("crossmind.strategy_layer")


class UnifiedRouter:
    """Single routing decision point for query classification, execution strategy, and budget allocation."""

    def __init__(self):
        self.classifier = get_query_classifier()
        self._route_cache: Dict[str, Dict[str, Any]] = {}

    def route(self, query: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = filter_metadata or {}
        cache_key = (query.strip(), metadata.get("user_role", "researcher"))
        if cache_key in self._route_cache:
            return dict(self._route_cache[cache_key])

        classification = self.classifier.classify(query)
        complexity = str(classification.get("complexity", "medium")).lower()
        query_type = str(classification.get("query_type", "factual")).lower()
        domain = str(classification.get("predicted_domain", "general")).lower()

        if complexity in {"low", "factual"}:
            execution_mode = "fast"
            retrieval_mode = "optimized_simple_vector"
            budget_tokens = 1500
            requires_multi_agent = False
            model = settings.LITELLM_MODEL_NAME if settings.LITELLM_ENABLED else "lite"
        elif complexity == "medium":
            execution_mode = "medium"
            retrieval_mode = "hybrid_rag_kg"
            budget_tokens = 3500
            requires_multi_agent = settings.MULTI_AGENT_ENABLED
            model = settings.ZAYA1B_MODEL_NAME if settings.ZAYA1B_ENABLED else settings.LITELLM_MODEL_NAME
        else:
            execution_mode = "deep"
            retrieval_mode = "hybrid_rag_kg"
            budget_tokens = 6000
            requires_multi_agent = settings.MULTI_AGENT_ENABLED
            model = settings.ZAYA1_8B_MODEL_NAME

        route = {
            "query": query,
            "predicted_domain": domain,
            "query_type": query_type,
            "complexity": complexity,
            "confidence": float(classification.get("confidence", 0.0)),
            "execution_mode": execution_mode,
            "model": model,
            "retrieval_strategy": retrieval_mode,
            "agent_count": 1 if execution_mode == "fast" else 3,
            "requires_multi_agent": requires_multi_agent,
            "requires_graph_rag": execution_mode in {"medium", "deep"},
            "budget_tokens": budget_tokens,
            "budget_cost_estimate": round(budget_tokens * 0.0004, 4),
            "selected_agents": [domain] if domain != "general" else ["general"],
            "timestamp": time.time(),
        }
        self._route_cache[cache_key] = route
        return dict(route)

    def expand_retrieval_query(self, query: str, entities: Optional[List[str]] = None) -> str:
        parts = [query]
        if entities:
            parts.extend([entity for entity in entities[:3] if entity and entity.lower() not in query.lower()])
        return " ".join(parts)


class QualityGate:
    """Quality gate that rejects low-confidence, weak-evidence results and triggers a fallback."""

    def __init__(self):
        self.default_thresholds = {"proceed": 0.75, "investigate": 0.50}

    def evaluate(self, calibrated_confidence: float, validation_score: float, evidence_count: int, discovery_score: float) -> Dict[str, Any]:
        confidence = float(calibrated_confidence)
        validation = float(validation_score)
        discovery = float(discovery_score)
        passed = confidence >= self.default_thresholds["proceed"] and validation >= 0.6 and evidence_count > 0 and discovery >= 40.0
        if passed:
            return {
                "passed": True,
                "decision": "proceed",
                "fallback": "none",
                "reason": "Confidence and evidence quality exceed minimum thresholds.",
            }

        if confidence >= self.default_thresholds["investigate"]:
            return {
                "passed": False,
                "decision": "requery",
                "fallback": "re_run_retrieval_with_expanded_context",
                "reason": "Confidence is borderline, so the system re-runs retrieval with expanded context.",
            }

        return {
            "passed": False,
            "decision": "human_review",
            "fallback": "escalate_to_human_review",
            "reason": "Evidence quality is too low for an autonomous answer.",
        }


class CostController:
    """Tracks approximate per-query cost and enforces a budget guardrail."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._total_cost = 0.0

    def track_query(self, query: str, execution_mode: str, budget_tokens: int, estimate_cost: float) -> Dict[str, Any]:
        record = {
            "query": query[:120],
            "execution_mode": execution_mode,
            "budget_tokens": int(budget_tokens),
            "estimated_cost": round(float(estimate_cost), 4),
            "timestamp": time.time(),
        }
        self._history.append(record)
        self._total_cost += record["estimated_cost"]
        return dict(record)

    def enforce_budget(self, query_cost: float, max_cost: float = 0.25) -> Dict[str, Any]:
        within_budget = query_cost <= max_cost
        return {
            "within_budget": within_budget,
            "max_budget": max_cost,
            "current_cost": round(float(query_cost), 4),
            "remaining_budget": round(max(0.0, max_cost - query_cost), 4),
            "decision": "approved" if within_budget else "blocked",
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_queries": len(self._history),
            "total_cost": round(self._total_cost, 4),
            "recent": self._history[-5:],
        }


_router_instance: Optional[UnifiedRouter] = None
_quality_gate_instance: Optional[QualityGate] = None
_cost_controller_instance: Optional[CostController] = None


def get_unified_router() -> UnifiedRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = UnifiedRouter()
    return _router_instance


def get_quality_gate() -> QualityGate:
    global _quality_gate_instance
    if _quality_gate_instance is None:
        _quality_gate_instance = QualityGate()
    return _quality_gate_instance


def get_cost_controller() -> CostController:
    global _cost_controller_instance
    if _cost_controller_instance is None:
        _cost_controller_instance = CostController()
    return _cost_controller_instance
