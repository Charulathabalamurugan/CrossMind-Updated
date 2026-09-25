import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestCostPredictor:
    """Tests for CostPredictor functionality."""

    def setup_method(self):
        from reasoning.cost_prediction import CostPredictor, get_cost_predictor
        CostPredictor._instance = None
        self.predictor = get_cost_predictor()
        self.predictor.reset_calibration()
        self.predictor.clear_history()

    def test_token_cost_estimation(self):
        from reasoning.cost_prediction import CostComponent
        estimate = self.predictor.estimate_token_cost("gpt-4o-mini", 1000, 500, confidence=0.95)
        assert estimate.component == CostComponent.TOKENS
        assert estimate.estimated_cost > 0
        assert estimate.confidence_lower >= 0
        assert estimate.confidence_upper > estimate.confidence_lower
        assert estimate.metadata["model"] == "gpt-4o-mini"
        assert estimate.metadata["input_tokens"] == 1000
        assert estimate.metadata["output_tokens"] == 500

    def test_model_cost_estimation(self):
        from reasoning.cost_prediction import CostComponent
        estimate = self.predictor.estimate_model_cost("gpt-4o", "deep", confidence=0.95)
        assert estimate.component == CostComponent.MODEL
        assert estimate.estimated_cost > 0
        assert estimate.metadata["execution_mode"] == "deep"
        assert estimate.metadata["multiplier"] == 4.0

    def test_retrieval_cost_estimation(self):
        from reasoning.cost_prediction import CostComponent
        estimate = self.predictor.estimate_retrieval_cost(
            num_vector_searches=2,
            num_bm25_searches=1,
            num_rerank_candidates=20,
            graph_hops=3,
            confidence=0.95,
        )
        assert estimate.component == CostComponent.RETRIEVAL
        assert estimate.estimated_cost > 0
        assert estimate.metadata["num_vector_searches"] == 2
        assert estimate.metadata["graph_hops"] == 3

    def test_agent_cost_estimation(self):
        from reasoning.cost_prediction import CostComponent
        estimate = self.predictor.estimate_agent_cost(num_steps=5, num_tool_calls=3, confidence=0.95)
        assert estimate.component == CostComponent.AGENT
        assert estimate.estimated_cost > 0
        assert estimate.metadata["num_steps"] == 5
        assert estimate.metadata["num_tool_calls"] == 3

    def test_storage_cost_estimation(self):
        from reasoning.cost_prediction import CostComponent
        estimate = self.predictor.estimate_storage_cost(
            storage_mb=500, num_vectors=5000, duration_months=2, confidence=0.95
        )
        assert estimate.component == CostComponent.STORAGE
        assert estimate.estimated_cost > 0
        assert estimate.metadata["storage_mb"] == 500
        assert estimate.metadata["duration_months"] == 2

    def test_total_cost_estimate_approve(self):
        from reasoning.cost_prediction import BudgetDecision
        self.predictor.set_tenant_budget("tenant_1", 10.0)
        total = self.predictor.estimate_total_cost(
            tenant_id="tenant_1",
            model="zaya1_8b",
            execution_mode="fast",
            input_tokens=100,
            output_tokens=50,
            num_vector_searches=1,
            num_bm25_searches=0,
            num_rerank_candidates=0,
            graph_hops=0,
            num_steps=1,
            num_tool_calls=0,
            storage_mb=1,
            num_vectors=100,
        )
        assert total.decision == BudgetDecision.APPROVE
        assert total.downgrade_plan is None
        assert total.budget_limit == 10.0

    def test_total_cost_estimate_downgrade(self):
        from reasoning.cost_prediction import BudgetDecision
        self.predictor.set_tenant_budget("tenant_2", 0.3)
        total = self.predictor.estimate_total_cost(
            tenant_id="tenant_2",
            model="gpt-4o",
            execution_mode="deep",
            input_tokens=10000,
            output_tokens=5000,
            num_vector_searches=5,
            num_bm25_searches=5,
            num_rerank_candidates=100,
            graph_hops=10,
            num_steps=10,
            num_tool_calls=10,
            storage_mb=100,
            num_vectors=10000,
        )
        assert total.decision == BudgetDecision.DOWNGRADE
        assert total.downgrade_plan is not None
        assert "steps" in total.downgrade_plan
        assert "recommended_model" in total.downgrade_plan
        assert "recommended_mode" in total.downgrade_plan

    def test_total_cost_estimate_reject(self):
        from reasoning.cost_prediction import BudgetDecision
        self.predictor.set_tenant_budget("tenant_3", 0.0)
        total = self.predictor.estimate_total_cost(
            tenant_id="tenant_3",
            model="gpt-4o",
            execution_mode="exhaustive",
            input_tokens=100000,
            output_tokens=50000,
            num_vector_searches=10,
            num_bm25_searches=10,
            num_rerank_candidates=200,
            graph_hops=20,
            num_steps=20,
            num_tool_calls=20,
            storage_mb=10000,
            num_vectors=1000000,
        )
        assert total.decision == BudgetDecision.REJECT
        assert total.downgrade_plan is not None

    def test_actual_cost_recording_and_calibration(self):
        from reasoning.cost_prediction import ActualCostRecord, CostComponent, BudgetDecision, CostEstimate, TotalCostEstimate

        estimates = [
            CostEstimate(CostComponent.TOKENS, 0.1, 0.05, 0.15, {}),
            CostEstimate(CostComponent.MODEL, 0.05, 0.02, 0.08, {}),
            CostEstimate(CostComponent.RETRIEVAL, 0.02, 0.01, 0.03, {}),
            CostEstimate(CostComponent.AGENT, 0.01, 0.005, 0.015, {}),
            CostEstimate(CostComponent.STORAGE, 0.005, 0.002, 0.008, {}),
        ]
        estimated_total = TotalCostEstimate(
            estimates=estimates,
            total_estimated=0.185,
            total_lower=0.087,
            total_upper=0.283,
            budget_limit=1.0,
            decision=BudgetDecision.APPROVE,
        )

        actual_costs = {
            CostComponent.TOKENS: 0.12,
            CostComponent.MODEL: 0.06,
            CostComponent.RETRIEVAL: 0.025,
            CostComponent.AGENT: 0.012,
            CostComponent.STORAGE: 0.006,
        }
        record = ActualCostRecord(
            query_id="q1",
            tenant_id="tenant_1",
            estimated=estimated_total,
            actual_costs=actual_costs,
            actual_total=0.223,
        )
        self.predictor.record_actual_cost(record)

        history = self.predictor.get_cost_history(tenant_id="tenant_1", limit=10)
        assert len(history) == 1
        assert history[0].query_id == "q1"

        factors = self.predictor.get_calibration_factors()
        assert all(k in factors for k in ["tokens", "model", "retrieval", "agent", "storage"])

    def test_accuracy_stats(self):
        from reasoning.cost_prediction import ActualCostRecord, CostComponent, BudgetDecision, CostEstimate, TotalCostEstimate

        for i in range(10):
            estimates = [
                CostEstimate(CostComponent.TOKENS, 0.1, 0.05, 0.15, {}),
                CostEstimate(CostComponent.MODEL, 0.05, 0.02, 0.08, {}),
            ]
            estimated_total = TotalCostEstimate(
                estimates=estimates,
                total_estimated=0.15,
                total_lower=0.07,
                total_upper=0.23,
                budget_limit=1.0,
                decision=BudgetDecision.APPROVE,
            )
            actual_costs = {
                CostComponent.TOKENS: 0.11,
                CostComponent.MODEL: 0.055,
            }
            record = ActualCostRecord(
                query_id=f"q{i}",
                tenant_id="tenant_stats",
                estimated=estimated_total,
                actual_costs=actual_costs,
                actual_total=0.165,
            )
            self.predictor.record_actual_cost(record)

        stats = self.predictor.get_accuracy_stats(tenant_id="tenant_stats")
        assert stats["sample_size"] == 10
        assert "tokens" in stats["by_component"]
        assert "model" in stats["by_component"]
        assert "mean_ratio" in stats["by_component"]["tokens"]

    def test_model_pricing_override(self):
        from reasoning.cost_prediction import ModelPricing
        custom_pricing = ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.02, base_cost=0.005)
        self.predictor.set_model_pricing("custom-model", custom_pricing)
        estimate = self.predictor.estimate_token_cost("custom-model", 1000, 1000)
        assert estimate.estimated_cost > 0
        assert estimate.metadata["input_cost_per_1k"] == 0.01

    def test_cost_estimate_to_dict(self):
        from reasoning.cost_prediction import CostEstimate, CostComponent
        estimate = CostEstimate(
            component=CostComponent.TOKENS,
            estimated_cost=0.1,
            confidence_lower=0.05,
            confidence_upper=0.15,
            metadata={"model": "test"},
        )
        d = estimate.to_dict()
        assert d["component"] == "tokens"
        assert d["estimated_cost"] == 0.1
        assert d["confidence_interval"] == [0.05, 0.15]
        assert d["metadata"]["model"] == "test"

    def test_total_cost_estimate_to_dict(self):
        from reasoning.cost_prediction import TotalCostEstimate, CostEstimate, CostComponent, BudgetDecision
        estimates = [CostEstimate(CostComponent.TOKENS, 0.1, 0.05, 0.15, {})]
        total = TotalCostEstimate(
            estimates=estimates,
            total_estimated=0.1,
            total_lower=0.05,
            total_upper=0.15,
            budget_limit=1.0,
            decision=BudgetDecision.APPROVE,
            downgrade_plan=None,
        )
        d = total.to_dict()
        assert d["total_estimated"] == 0.1
        assert d["decision"] == "approve"
        assert d["downgrade_plan"] is None


class TestScaleManager:
    """Tests for ScaleManager functionality."""

    def setup_method(self):
        from reasoning.scale_manager import ScaleManager, get_scale_manager, RateLimitConfig, TenantQuotaConfig
        ScaleManager._instance = None
        self.manager = get_scale_manager()
        self.manager.configure_redis = Mock(return_value=False)
        self.manager._rate_limiter = Mock()
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})
        self.manager._rate_limiter.get_current_usage.return_value = 0
        self.manager._rate_limiter.reset = Mock()
        self.manager._rate_limiter.is_using_redis.return_value = False

    def test_tenant_configuration(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(
            rate_limit=RateLimitConfig(requests_per_window=100, window_seconds=60.0),
            max_concurrent=10,
            max_queue_size=50,
            priority=5,
        )
        self.manager.configure_tenant("tenant_1", config)
        retrieved = self.manager.get_tenant_config("tenant_1")
        assert retrieved is not None
        assert retrieved.max_concurrent == 10
        assert retrieved.max_queue_size == 50
        assert retrieved.priority == 5

    def test_rate_limit_check(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(requests_per_window=10, window_seconds=60.0))
        self.manager.configure_tenant("tenant_rl", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True, "remaining": 9})
        allowed, info = self.manager.check_rate_limit("tenant_rl", 1)
        assert allowed is True
        assert info["allowed"] is True

    def test_rate_limit_exceeded(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(requests_per_window=10, window_seconds=60.0))
        self.manager.configure_tenant("tenant_rl2", config)
        self.manager._rate_limiter.check_and_increment.return_value = (False, {"allowed": False, "retry_after": 30.0})
        allowed, info = self.manager.check_rate_limit("tenant_rl2", 1)
        assert allowed is False
        assert info["allowed"] is False

    def test_circuit_breaker_closed_state(self):
        from reasoning.scale_manager import CircuitBreakerConfig, CircuitState
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1.0)
        self.manager.configure_circuit_breaker("test_service", config)
        breaker = self.manager.get_circuit_breaker("test_service")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute() is True

    def test_circuit_breaker_opens_after_failures(self):
        from reasoning.scale_manager import CircuitBreakerConfig, CircuitState
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        self.manager.configure_circuit_breaker("fail_service", config)
        breaker = self.manager.get_circuit_breaker("fail_service")
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() is False

    def test_circuit_breaker_half_open_recovery(self):
        from reasoning.scale_manager import CircuitBreakerConfig, CircuitState
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout_seconds=0.1)
        self.manager.configure_circuit_breaker("recover_service", config)
        breaker = self.manager.get_circuit_breaker("recover_service")
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        time.sleep(0.15)
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_execute_with_circuit_breaker_success(self):
        from reasoning.scale_manager import CircuitBreakerConfig
        config = CircuitBreakerConfig(failure_threshold=3)
        self.manager.configure_circuit_breaker("success_service", config)
        result = self.manager.execute_with_circuit_breaker("success_service", lambda: "ok")
        assert result == "ok"

    def test_execute_with_circuit_breaker_fallback(self):
        from reasoning.scale_manager import CircuitBreakerConfig, CircuitState
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=10.0)
        self.manager.configure_circuit_breaker("fallback_service", config)
        breaker = self.manager.get_circuit_breaker("fallback_service")
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        result = self.manager.execute_with_circuit_breaker("fallback_service", lambda: "fail", fallback=lambda: "fallback_ok")
        assert result == "fallback_ok"

    def test_concurrency_control(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_concurrent=2)
        self.manager.configure_tenant("tenant_cc", config)
        acquired1 = self.manager.acquire_concurrency("tenant_cc", blocking=False)
        acquired2 = self.manager.acquire_concurrency("tenant_cc", blocking=False)
        acquired3 = self.manager.acquire_concurrency("tenant_cc", blocking=False)
        assert acquired1 is True
        assert acquired2 is True
        assert acquired3 is False
        self.manager.release_concurrency("tenant_cc")
        self.manager.release_concurrency("tenant_cc")

    def test_global_concurrency_limit(self):
        self.manager.set_global_max_concurrent(2)
        acquired1 = self.manager.acquire_concurrency("tenant_g1", blocking=False)
        acquired2 = self.manager.acquire_concurrency("tenant_g2", blocking=False)
        acquired3 = self.manager.acquire_concurrency("tenant_g3", blocking=False)
        assert acquired1 is True
        assert acquired2 is True
        assert acquired3 is False
        self.manager.release_concurrency("tenant_g1")
        self.manager.release_concurrency("tenant_g2")

    def test_load_shedding_reject_new(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig, LoadSheddingStrategy
        import time
        self.manager.set_load_shedding_strategy(LoadSheddingStrategy.REJECT_NEW)
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_queue_size=1, max_concurrent=1)
        self.manager.configure_tenant("tenant_ls1", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})

        def slow_fn():
            time.sleep(0.5)

        ok1, id1, info1 = self.manager.enqueue_request("tenant_ls1", slow_fn)
        time.sleep(0.1)
        ok2, id2, info2 = self.manager.enqueue_request("tenant_ls1", slow_fn)
        time.sleep(0.1)
        ok3, id3, info3 = self.manager.enqueue_request("tenant_ls1", slow_fn)

        assert ok1 is True
        assert ok2 is True
        assert ok3 is False
        assert info3["error"] == "queue_full"
        assert info3["strategy"] == "reject_new"

    def test_load_shedding_drop_oldest(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig, LoadSheddingStrategy
        import time
        self.manager.set_load_shedding_strategy(LoadSheddingStrategy.DROP_OLDEST)
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_queue_size=1, max_concurrent=1)
        self.manager.configure_tenant("tenant_ls2", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})

        def slow_fn():
            time.sleep(0.5)

        ok1, id1, info1 = self.manager.enqueue_request("tenant_ls2", slow_fn, priority=1)
        time.sleep(0.1)
        ok2, id2, info2 = self.manager.enqueue_request("tenant_ls2", slow_fn, priority=5)
        time.sleep(0.1)
        ok3, id3, info3 = self.manager.enqueue_request("tenant_ls2", slow_fn, priority=3)

        assert ok1 is True
        assert ok2 is True
        assert ok3 is True
        assert "dropped" in info3

    def test_load_shedding_drop_lowest_priority(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig, LoadSheddingStrategy
        import time
        self.manager.set_load_shedding_strategy(LoadSheddingStrategy.DROP_LOWEST_PRIORITY)
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_queue_size=1, max_concurrent=1)
        self.manager.configure_tenant("tenant_ls3", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})

        def slow_fn():
            time.sleep(0.5)

        ok1, id1, info1 = self.manager.enqueue_request("tenant_ls3", slow_fn, priority=5)
        time.sleep(0.1)
        ok2, id2, info2 = self.manager.enqueue_request("tenant_ls3", slow_fn, priority=1)
        time.sleep(0.1)
        ok3, id3, info3 = self.manager.enqueue_request("tenant_ls3", slow_fn, priority=3)

        assert ok1 is True
        assert ok2 is True
        assert ok3 is True
        assert "dropped" in info3
        assert info3["dropped"] == id2

    def test_queue_processor(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_concurrent=5, max_queue_size=10)
        self.manager.configure_tenant("tenant_qp", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})
        results = []
        def make_fn(val):
            return lambda: results.append(val)
        ok1, id1, _ = self.manager.enqueue_request("tenant_qp", make_fn(1), priority=1, timeout=1.0)
        ok2, id2, _ = self.manager.enqueue_request("tenant_qp", make_fn(2), priority=2, timeout=1.0)
        time.sleep(0.3)
        assert 1 in results
        assert 2 in results

    def test_health_checks(self):
        self.manager.register_health_check("check1", lambda: True)
        self.manager.register_health_check("check2", lambda: False)
        health = self.manager.run_health_checks()
        assert health["status"] == "degraded"
        assert health["checks"]["check1"]["healthy"] is True
        assert health["checks"]["check2"]["healthy"] is False
        self.manager.unregister_health_check("check1")
        health2 = self.manager.run_health_checks()
        assert "check1" not in health2["checks"]

    def test_degradation_mode(self):
        assert self.manager.is_degraded() is False
        self.manager.set_degradation_mode(True)
        assert self.manager.is_degraded() is True
        health = self.manager.run_health_checks()
        assert health["status"] == "degraded"
        assert health["degradation_mode"] is True
        self.manager.set_degradation_mode(False)
        assert self.manager.is_degraded() is False

    def test_health_summary(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig, CircuitBreakerConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0))
        self.manager.configure_tenant("tenant_hs", config)
        self.manager.configure_circuit_breaker("svc1", CircuitBreakerConfig())
        self.manager._rate_limiter.is_using_redis.return_value = False
        summary = self.manager.get_health_summary()
        assert "status" in summary
        assert "tenants_configured" in summary
        assert "circuit_breakers" in summary
        assert "queue_depths" in summary
        assert "rate_limiter_backend" in summary

    def test_tenant_status(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_concurrent=5, max_queue_size=10, priority=3)
        self.manager.configure_tenant("tenant_status", config)
        self.manager._rate_limiter.get_current_usage.return_value = 50
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})
        status = self.manager.get_tenant_status("tenant_status")
        assert status["tenant_id"] == "tenant_status"
        assert status["concurrency"]["max"] == 5
        assert status["queue"]["max_size"] == 10
        assert status["priority"] == 3

    def test_reset_tenant_limits(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(100, 60.0), max_concurrent=5)
        self.manager.configure_tenant("tenant_reset", config)
        self.manager.reset_tenant_limits("tenant_reset")
        self.manager._rate_limiter.reset.assert_called_with("tenant:tenant_reset", 60.0)

    def test_concurrent_requests_thread_safety(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(1000, 60.0), max_concurrent=50)
        self.manager.configure_tenant("tenant_concurrent", config)
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})

        def acquire_release():
            acquired = self.manager.acquire_concurrency("tenant_concurrent", blocking=True, timeout=1.0)
            if acquired:
                time.sleep(0.01)
                self.manager.release_concurrency("tenant_concurrent")
            return acquired

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(acquire_release) for _ in range(30)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)
        assert self.manager._global_active_count == 0

    def test_sliding_window_rate_limiter_memory_fallback(self):
        from reasoning.scale_manager import SlidingWindowRateLimiter, RateLimitConfig
        limiter = SlidingWindowRateLimiter(redis_client=None, fallback_memory=True)
        config = RateLimitConfig(requests_per_window=5, window_seconds=60.0)
        for i in range(5):
            allowed, info = limiter.check_and_increment("test_key", config, 1)
            assert allowed is True
        allowed, info = limiter.check_and_increment("test_key", config, 1)
        assert allowed is False
        assert info["retry_after"] > 0

    def test_sliding_window_rate_limiter_reset(self):
        from reasoning.scale_manager import SlidingWindowRateLimiter, RateLimitConfig
        limiter = SlidingWindowRateLimiter(redis_client=None, fallback_memory=True)
        config = RateLimitConfig(requests_per_window=5, window_seconds=60.0)
        for i in range(3):
            limiter.check_and_increment("reset_key", config, 1)
        limiter.reset("reset_key", 60.0)
        usage = limiter.get_current_usage("reset_key", 60.0)
        assert usage == 0


class TestCostScaleIntegration:
    """Integration tests for CostPredictor and ScaleManager together."""

    def setup_method(self):
        from reasoning.cost_prediction import CostPredictor, get_cost_predictor
        from reasoning.scale_manager import ScaleManager, get_scale_manager
        CostPredictor._instance = None
        ScaleManager._instance = None
        self.predictor = get_cost_predictor()
        self.predictor.reset_calibration()
        self.predictor.clear_history()
        self.manager = get_scale_manager()
        self.manager.configure_redis = Mock(return_value=False)
        self.manager._rate_limiter = Mock()
        self.manager._rate_limiter.check_and_increment.return_value = (True, {"allowed": True})
        self.manager._rate_limiter.get_current_usage.return_value = 0
        self.manager._rate_limiter.reset = Mock()
        self.manager._rate_limiter.is_using_redis.return_value = False

    def test_budget_enforcement_with_rate_limiting(self):
        from reasoning.scale_manager import RateLimitConfig, TenantQuotaConfig
        from reasoning.cost_prediction import BudgetDecision
        self.predictor.set_tenant_budget("tenant_int", 0.12)
        config = TenantQuotaConfig(rate_limit=RateLimitConfig(10, 60.0))
        self.manager.configure_tenant("tenant_int", config)
        total = self.predictor.estimate_total_cost(
            tenant_id="tenant_int",
            model="gpt-4o",
            execution_mode="deep",
            input_tokens=5000,
            output_tokens=2000,
        )
        assert total.decision == BudgetDecision.DOWNGRADE
        allowed, info = self.manager.check_rate_limit("tenant_int", 1)
        assert allowed is True

    def test_circuit_breaker_with_cost_tracking(self):
        from reasoning.scale_manager import CircuitBreakerConfig
        from reasoning.cost_prediction import ActualCostRecord, CostComponent, BudgetDecision, CostEstimate, TotalCostEstimate
        self.manager.configure_circuit_breaker("costly_service", CircuitBreakerConfig(failure_threshold=2))
        estimates = [CostEstimate(CostComponent.MODEL, 0.5, 0.4, 0.6, {})]
        estimated_total = TotalCostEstimate(estimates, 0.5, 0.4, 0.6, 1.0, BudgetDecision.APPROVE)
        actual_costs = {CostComponent.MODEL: 0.6}
        record = ActualCostRecord("q1", "t1", estimated_total, actual_costs, 0.6)
        self.predictor.record_actual_cost(record)
        breaker = self.manager.get_circuit_breaker("costly_service")
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.value == "open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])