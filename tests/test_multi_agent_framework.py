import unittest

from reasoning.multi_agent import (
    MultiAgentOrchestrator,
    CriticAgent,
    SynthesizerAgent,
    StrategyEngine,
    AgentMemory,
)
from reasoning.strategy_layer import get_unified_router, get_quality_gate, get_cost_controller


class TestMultiAgentFramework(unittest.TestCase):
    def test_orchestrator_routes_to_specialist_agents(self):
        orchestrator = MultiAgentOrchestrator()
        evidence = [
            {"id": "e1", "payload": {"domain": "energy", "content": "Battery storage reduces peak demand."}},
            {"id": "e2", "payload": {"domain": "finance", "content": "Market volatility is influenced by energy prices."}},
        ]
        result = orchestrator.orchestrate(
            "How do battery storage costs affect volatility in energy markets?",
            evidence,
            {"detected_domains": ["energy", "finance"], "session_id": "sess-1"},
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreaterEqual(len(result["agent_runs"]), 2)
        self.assertIn("synthesized_answer", result)

    def test_critic_flags_unverified_claim(self):
        critic = CriticAgent()
        verdict = critic.evaluate(
            "Battery systems eliminate volatility in all markets.",
            [
                {"id": "e1", "payload": {"domain": "energy", "content": "Battery storage can reduce peak demand but does not eliminate volatility."}},
            ],
        )
        self.assertFalse(verdict["passed"])
        self.assertGreater(len(verdict["flags"]), 0)

    def test_synthesizer_merges_conflicting_inputs(self):
        synthesizer = SynthesizerAgent()
        answer = synthesizer.synthesize(
            query="How do energy storage and market volatility interact?",
            agent_reports=[
                {"agent_id": "energy_agent", "summary": "Storage reduces peak demand."},
                {"agent_id": "finance_agent", "summary": "Energy prices drive volatility."},
            ],
            critic_report={"flags": [], "passed": True},
        )
        self.assertIn("energy", answer["final_answer"].lower())
        self.assertIn("finance", answer["final_answer"].lower())

    def test_strategy_engine_selects_deep_plan(self):
        strategy = StrategyEngine()
        plan = strategy.plan(
            "What are the cross-domain effects of battery storage, grid policy, and portfolio risk across energy and finance?"
        )
        self.assertEqual(plan["execution_mode"], "deep")
        self.assertGreater(plan["budget_tokens"], 0)

    def test_agent_memory_tracks_session_and_long_term_context(self):
        memory = AgentMemory()
        memory.record_interaction("sess-1", "energy", "Battery costs affect grid pricing.", {"passed": True})
        self.assertIn("Battery costs", memory.get_session_context("sess-1")["recent_interactions"][-1]["content"])
        self.assertGreater(len(memory.get_long_term_memory("energy")), 0)

    def test_unified_router_quality_and_budget_layer(self):
        router = get_unified_router()
        route = router.route("How do battery storage and market volatility interact across energy and finance?")
        self.assertEqual(route["execution_mode"], "deep")
        self.assertGreater(route["budget_tokens"], 0)

        gate = get_quality_gate()
        decision = gate.evaluate(0.8, 0.7, 5, 80.0)
        self.assertTrue(decision["passed"])

        controller = get_cost_controller()
        report = controller.track_query("battery volatility query", "deep", 6000, 0.18)
        self.assertEqual(report["execution_mode"], "deep")
        self.assertTrue(controller.enforce_budget(0.18)["within_budget"])


if __name__ == "__main__":
    unittest.main()
