import threading
import time
import math
import statistics
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from config import settings


class CostComponent(str, Enum):
    TOKENS = "tokens"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    AGENT = "agent"
    STORAGE = "storage"


class BudgetDecision(str, Enum):
    APPROVE = "approve"
    DOWNGRADE = "downgrade"
    REJECT = "reject"


@dataclass
class CostEstimate:
    component: CostComponent
    estimated_cost: float
    confidence_lower: float
    confidence_upper: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.value,
            "estimated_cost": self.estimated_cost,
            "confidence_interval": [self.confidence_lower, self.confidence_upper],
            "metadata": self.metadata,
        }


@dataclass
class TotalCostEstimate:
    estimates: List[CostEstimate]
    total_estimated: float
    total_lower: float
    total_upper: float
    budget_limit: float
    decision: BudgetDecision
    downgrade_plan: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimates": [e.to_dict() for e in self.estimates],
            "total_estimated": self.total_estimated,
            "confidence_interval": [self.total_lower, self.total_upper],
            "budget_limit": self.budget_limit,
            "decision": self.decision.value,
            "downgrade_plan": self.downgrade_plan,
            "timestamp": self.timestamp,
        }


@dataclass
class ActualCostRecord:
    query_id: str
    tenant_id: str
    estimated: TotalCostEstimate
    actual_costs: Dict[CostComponent, float]
    actual_total: float
    timestamp: float = field(default_factory=time.time)

    def variance(self) -> Dict[str, float]:
        return {
            comp.value: actual - est.estimated_cost
            for comp, actual in self.actual_costs.items()
            for est in self.estimated.estimates
            if est.component == comp
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "tenant_id": self.tenant_id,
            "estimated": self.estimated.to_dict(),
            "actual_costs": {k.value: v for k, v in self.actual_costs.items()},
            "actual_total": self.actual_total,
            "timestamp": self.timestamp,
        }


@dataclass
class ModelPricing:
    input_cost_per_1k: float
    output_cost_per_1k: float
    base_cost: float = 0.0


DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    "zaya1_8b": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0, base_cost=0.0),
    "zaya1b": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0, base_cost=0.0),
    "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006, base_cost=0.0),
    "gpt-4o": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.015, base_cost=0.0),
    "claude-3-haiku": ModelPricing(input_cost_per_1k=0.00025, output_cost_per_1k=0.00125, base_cost=0.0),
    "claude-3-sonnet": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015, base_cost=0.0),
    "simulator": ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0, base_cost=0.0),
}


@dataclass
class RetrievalPricing:
    vector_search_cost_per_query: float = 0.0001
    bm25_cost_per_query: float = 0.00005
    rerank_cost_per_candidate: float = 0.00001
    graph_traversal_cost_per_hop: float = 0.0002


@dataclass
class AgentPricing:
    base_cost_per_step: float = 0.001
    tool_call_cost: float = 0.0005


@dataclass
class StoragePricing:
    cost_per_mb_per_month: float = 0.023
    cost_per_1k_vectors: float = 0.01


class CostPredictor:
    _instance: Optional["CostPredictor"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model_pricing: Dict[str, ModelPricing] = DEFAULT_MODEL_PRICING.copy()
        self._retrieval_pricing = RetrievalPricing()
        self._agent_pricing = AgentPricing()
        self._storage_pricing = StoragePricing()
        self._history: deque[ActualCostRecord] = deque(maxlen=10000)
        self._tenant_budgets: Dict[str, float] = {}
        self._history_lock = threading.RLock()
        self._budget_lock = threading.RLock()
        self._calibration_factors: Dict[CostComponent, float] = {
            CostComponent.TOKENS: 1.0,
            CostComponent.MODEL: 1.0,
            CostComponent.RETRIEVAL: 1.0,
            CostComponent.AGENT: 1.0,
            CostComponent.STORAGE: 1.0,
        }
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "CostPredictor":
        return cls()

    def set_model_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._model_pricing[model] = pricing

    def set_tenant_budget(self, tenant_id: str, budget: float) -> None:
        with self._budget_lock:
            self._tenant_budgets[tenant_id] = budget

    def get_tenant_budget(self, tenant_id: str) -> float:
        with self._budget_lock:
            return self._tenant_budgets.get(tenant_id, float('inf'))

    def estimate_token_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        confidence: float = 0.95,
    ) -> CostEstimate:
        pricing = self._model_pricing.get(model, DEFAULT_MODEL_PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1000.0) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * pricing.output_cost_per_1k
        base_cost = pricing.base_cost
        estimated = (input_cost + output_cost + base_cost) * self._calibration_factors[CostComponent.TOKENS]

        z = 1.96 if confidence >= 0.95 else 1.645
        std_dev = estimated * 0.1
        margin = z * std_dev

        return CostEstimate(
            component=CostComponent.TOKENS,
            estimated_cost=estimated,
            confidence_lower=max(0.0, estimated - margin),
            confidence_upper=estimated + margin,
            metadata={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_per_1k": pricing.input_cost_per_1k,
                "output_cost_per_1k": pricing.output_cost_per_1k,
            },
        )

    def estimate_model_cost(
        self,
        model: str,
        execution_mode: str,
        confidence: float = 0.95,
    ) -> CostEstimate:
        mode_multipliers = {
            "fast": 1.0,
            "standard": 2.0,
            "deep": 4.0,
            "exhaustive": 8.0,
        }
        multiplier = mode_multipliers.get(execution_mode, 2.0)
        pricing = self._model_pricing.get(model, DEFAULT_MODEL_PRICING["gpt-4o-mini"])
        base_estimate = (pricing.base_cost or 0.01) * multiplier
        estimated = base_estimate * self._calibration_factors[CostComponent.MODEL]

        z = 1.96 if confidence >= 0.95 else 1.645
        std_dev = estimated * 0.15
        margin = z * std_dev

        return CostEstimate(
            component=CostComponent.MODEL,
            estimated_cost=estimated,
            confidence_lower=max(0.0, estimated - margin),
            confidence_upper=estimated + margin,
            metadata={"model": model, "execution_mode": execution_mode, "multiplier": multiplier},
        )

    def estimate_retrieval_cost(
        self,
        num_vector_searches: int = 1,
        num_bm25_searches: int = 1,
        num_rerank_candidates: int = 20,
        graph_hops: int = 0,
        confidence: float = 0.95,
    ) -> CostEstimate:
        rp = self._retrieval_pricing
        vector_cost = num_vector_searches * rp.vector_search_cost_per_query
        bm25_cost = num_bm25_searches * rp.bm25_cost_per_query
        rerank_cost = num_rerank_candidates * rp.rerank_cost_per_candidate
        graph_cost = graph_hops * rp.graph_traversal_cost_per_hop
        estimated = (vector_cost + bm25_cost + rerank_cost + graph_cost) * self._calibration_factors[CostComponent.RETRIEVAL]

        z = 1.96 if confidence >= 0.95 else 1.645
        std_dev = estimated * 0.2
        margin = z * std_dev

        return CostEstimate(
            component=CostComponent.RETRIEVAL,
            estimated_cost=estimated,
            confidence_lower=max(0.0, estimated - margin),
            confidence_upper=estimated + margin,
            metadata={
                "num_vector_searches": num_vector_searches,
                "num_bm25_searches": num_bm25_searches,
                "num_rerank_candidates": num_rerank_candidates,
                "graph_hops": graph_hops,
            },
        )

    def estimate_agent_cost(
        self,
        num_steps: int,
        num_tool_calls: int,
        confidence: float = 0.95,
    ) -> CostEstimate:
        ap = self._agent_pricing
        step_cost = num_steps * ap.base_cost_per_step
        tool_cost = num_tool_calls * ap.tool_call_cost
        estimated = (step_cost + tool_cost) * self._calibration_factors[CostComponent.AGENT]

        z = 1.96 if confidence >= 0.95 else 1.645
        std_dev = estimated * 0.25
        margin = z * std_dev

        return CostEstimate(
            component=CostComponent.AGENT,
            estimated_cost=estimated,
            confidence_lower=max(0.0, estimated - margin),
            confidence_upper=estimated + margin,
            metadata={"num_steps": num_steps, "num_tool_calls": num_tool_calls},
        )

    def estimate_storage_cost(
        self,
        storage_mb: int,
        num_vectors: int,
        duration_months: int = 1,
        confidence: float = 0.95,
    ) -> CostEstimate:
        sp = self._storage_pricing
        storage_cost = (storage_mb / 1024.0) * sp.cost_per_mb_per_month * duration_months
        vector_cost = (num_vectors / 1000.0) * sp.cost_per_1k_vectors * duration_months
        estimated = (storage_cost + vector_cost) * self._calibration_factors[CostComponent.STORAGE]

        z = 1.96 if confidence >= 0.95 else 1.645
        std_dev = estimated * 0.1
        margin = z * std_dev

        return CostEstimate(
            component=CostComponent.STORAGE,
            estimated_cost=estimated,
            confidence_lower=max(0.0, estimated - margin),
            confidence_upper=estimated + margin,
            metadata={
                "storage_mb": storage_mb,
                "num_vectors": num_vectors,
                "duration_months": duration_months,
            },
        )

    def estimate_total_cost(
        self,
        tenant_id: str,
        model: str,
        execution_mode: str,
        input_tokens: int,
        output_tokens: int,
        num_vector_searches: int = 1,
        num_bm25_searches: int = 1,
        num_rerank_candidates: int = 20,
        graph_hops: int = 0,
        num_steps: int = 3,
        num_tool_calls: int = 2,
        storage_mb: int = 10,
        num_vectors: int = 1000,
        confidence: float = 0.95,
    ) -> TotalCostEstimate:
        estimates = [
            self.estimate_token_cost(model, input_tokens, output_tokens, confidence),
            self.estimate_model_cost(model, execution_mode, confidence),
            self.estimate_retrieval_cost(
                num_vector_searches, num_bm25_searches, num_rerank_candidates, graph_hops, confidence
            ),
            self.estimate_agent_cost(num_steps, num_tool_calls, confidence),
            self.estimate_storage_cost(storage_mb, num_vectors, confidence=confidence),
        ]

        total_estimated = sum(e.estimated_cost for e in estimates)
        total_lower = sum(e.confidence_lower for e in estimates)
        total_upper = sum(e.confidence_upper for e in estimates)

        budget_limit = self.get_tenant_budget(tenant_id)
        decision, downgrade_plan = self._make_budget_decision(
            total_estimated, total_upper, budget_limit, estimates, model, execution_mode
        )

        return TotalCostEstimate(
            estimates=estimates,
            total_estimated=total_estimated,
            total_lower=total_lower,
            total_upper=total_upper,
            budget_limit=budget_limit,
            decision=decision,
            downgrade_plan=downgrade_plan,
        )

    def _make_budget_decision(
        self,
        total_estimated: float,
        total_upper: float,
        budget_limit: float,
        estimates: List[CostEstimate],
        model: str,
        execution_mode: str,
    ) -> Tuple[BudgetDecision, Optional[Dict[str, Any]]]:
        if budget_limit == float('inf'):
            return BudgetDecision.APPROVE, None

        if total_upper <= budget_limit:
            return BudgetDecision.APPROVE, None

        if total_estimated <= budget_limit < total_upper:
            return BudgetDecision.DOWNGRADE, self._generate_downgrade_plan(
                estimates, model, execution_mode, budget_limit
            )

        return BudgetDecision.REJECT, self._generate_downgrade_plan(
            estimates, model, execution_mode, budget_limit
        )

    def _generate_downgrade_plan(
        self,
        estimates: List[CostEstimate],
        model: str,
        execution_mode: str,
        budget_limit: float,
    ) -> Dict[str, Any]:
        current_total = sum(e.estimated_cost for e in estimates)
        plan = {"original_model": model, "original_mode": execution_mode, "steps": []}

        mode_downgrades = ["exhaustive", "deep", "standard", "fast"]
        current_mode_idx = mode_downgrades.index(execution_mode) if execution_mode in mode_downgrades else 1

        for i in range(current_mode_idx + 1, len(mode_downgrades)):
            new_mode = mode_downgrades[i]
            new_model_est = self.estimate_model_cost(model, new_mode, 0.95)
            model_est = next((e for e in estimates if e.component == CostComponent.MODEL), None)
            if model_est:
                savings = model_est.estimated_cost - new_model_est.estimated_cost
                new_total = current_total - savings
                plan["steps"].append(
                    {
                        "action": "downgrade_execution_mode",
                        "from": execution_mode,
                        "to": new_mode,
                        "estimated_savings": savings,
                        "new_total_estimate": new_total,
                    }
                )
                if new_total <= budget_limit:
                    plan["recommended_mode"] = new_mode
                    plan["recommended_model"] = model
                    return plan

        fallback_models = ["zaya1b", "zaya1_8b", "simulator"]
        for fallback_model in fallback_models:
            if fallback_model == model:
                continue
            new_model_est = self.estimate_model_cost(fallback_model, "fast", 0.95)
            model_est = next((e for e in estimates if e.component == CostComponent.MODEL), None)
            if model_est:
                savings = model_est.estimated_cost - new_model_est.estimated_cost
                new_total = current_total - savings
                plan["steps"].append(
                    {
                        "action": "fallback_model",
                        "from": model,
                        "to": fallback_model,
                        "estimated_savings": savings,
                        "new_total_estimate": new_total,
                    }
                )
                if new_total <= budget_limit:
                    plan["recommended_model"] = fallback_model
                    plan["recommended_mode"] = "fast"
                    return plan

        plan["recommended_model"] = "simulator"
        plan["recommended_mode"] = "fast"
        return plan

    def record_actual_cost(self, record: ActualCostRecord) -> None:
        with self._history_lock:
            self._history.append(record)
            self._update_calibration(record)

    def _update_calibration(self, record: ActualCostRecord) -> None:
        variances = record.variance()
        for comp_str, variance in variances.items():
            comp = CostComponent(comp_str)
            if abs(variance) > 0.001:
                est = next((e for e in record.estimated.estimates if e.component == comp), None)
                if est and est.estimated_cost > 0:
                    ratio = record.actual_costs[comp] / est.estimated_cost
                    alpha = 0.1
                    self._calibration_factors[comp] = (
                        self._calibration_factors[comp] * (1 - alpha) + ratio * alpha
                    )

    def get_calibration_factors(self) -> Dict[str, float]:
        return {k.value: v for k, v in self._calibration_factors.items()}

    def get_cost_history(
        self, tenant_id: Optional[str] = None, limit: int = 100
    ) -> List[ActualCostRecord]:
        with self._history_lock:
            records = list(self._history)
            if tenant_id:
                records = [r for r in records if r.tenant_id == tenant_id]
            return records[-limit:]

    def get_accuracy_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        with self._history_lock:
            records = list(self._history)
            if tenant_id:
                records = [r for r in records if r.tenant_id == tenant_id]

            if not records:
                return {"sample_size": 0}

            by_component: Dict[CostComponent, List[float]] = defaultdict(list)
            for record in records:
                for comp, actual in record.actual_costs.items():
                    est = next((e for e in record.estimated.estimates if e.component == comp), None)
                    if est and est.estimated_cost > 0:
                        by_component[comp].append(actual / est.estimated_cost)

            stats = {"sample_size": len(records), "by_component": {}}
            for comp, ratios in by_component.items():
                if ratios:
                    stats["by_component"][comp.value] = {
                        "mean_ratio": statistics.mean(ratios),
                        "median_ratio": statistics.median(ratios),
                        "stdev": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
                        "min_ratio": min(ratios),
                        "max_ratio": max(ratios),
                    }

            return stats

    def reset_calibration(self) -> None:
        with self._history_lock:
            self._calibration_factors = {
                CostComponent.TOKENS: 1.0,
                CostComponent.MODEL: 1.0,
                CostComponent.RETRIEVAL: 1.0,
                CostComponent.AGENT: 1.0,
                CostComponent.STORAGE: 1.0,
            }

    def clear_history(self) -> None:
        with self._history_lock:
            self._history.clear()


def get_cost_predictor() -> CostPredictor:
    return CostPredictor.get_instance()


__all__ = [
    "CostPredictor",
    "get_cost_predictor",
    "CostComponent",
    "BudgetDecision",
    "CostEstimate",
    "TotalCostEstimate",
    "ActualCostRecord",
    "ModelPricing",
    "RetrievalPricing",
    "AgentPricing",
    "StoragePricing",
]