import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from enum import Enum


class TestAgentRegistryLifecycle:
    """Test agent registry lifecycle management."""

    def test_agent_registration_and_retrieval(self):
        """Agents should be registered and retrievable by ID."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability,
            RegisteredAgent, get_agent_registry, AgentState
        )
        
        class TestAgent(Agent):
            def __init__(self, agent_id: str, domain: str):
                self._agent_id = agent_id
                self._domain = domain
                self._started = False
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id=self._agent_id,
                    name=f"TestAgent-{self._agent_id}",
                    domain=self._domain,
                    capabilities=[AgentCapability.REASONING],
                    version="1.0.0"
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                self._started = True
            
            def stop(self):
                self._started = False
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "healthy" if self._started else "stopped"}
        
        registry = AgentRegistry()
        
        agent = TestAgent("test_agent_1", "test_domain")
        registered = registry.register(agent, domain="test_domain")
        
        assert registered.descriptor.agent_id == "test_agent_1"
        assert registered.state == AgentState.REGISTERED
        
        retrieved = registry.get("test_agent_1")
        assert retrieved is not None
        assert retrieved.descriptor.agent_id == "test_agent_1"

    def test_agent_lifecycle_start_stop(self):
        """Agents should transition through correct lifecycle states."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability,
            AgentState, LifecycleHooks
        )
        
        lifecycle_events = []
        
        class TestAgent(Agent):
            def __init__(self):
                self._started = False
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="lifecycle_agent",
                    name="LifecycleAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                self._started = True
                lifecycle_events.append("start")
            
            def stop(self):
                self._started = False
                lifecycle_events.append("stop")
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "healthy" if self._started else "stopped"}
        
        registry = AgentRegistry()
        agent = TestAgent()
        registered = registry.register(agent)
        
        hooks = LifecycleHooks(
            on_start=lambda a: lifecycle_events.append("hook_start"),
            on_stop=lambda a: lifecycle_events.append("hook_stop")
        )
        registered.hooks = hooks
        
        assert registry.start_agent("lifecycle_agent") is True
        assert registered.state == AgentState.RUNNING
        assert "hook_start" in lifecycle_events
        assert "start" in lifecycle_events
        
        assert registry.stop_agent("lifecycle_agent") is True
        assert registered.state == AgentState.STOPPED
        assert "hook_stop" in lifecycle_events
        assert "stop" in lifecycle_events

    def test_agent_duplicate_registration_fails(self):
        """Registering agent with same ID should fail."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability
        )
        
        class TestAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="duplicate_agent",
                    name="DuplicateAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
        
        registry = AgentRegistry()
        agent1 = TestAgent()
        registry.register(agent1)
        
        agent2 = TestAgent()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(agent2)

    def test_agent_unregistration(self):
        """Unregistering agent should remove it and clean up indexes."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class TestAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="unregister_agent",
                    name="UnregisterAgent",
                    domain="test_domain",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
        
        registry = AgentRegistry()
        agent = TestAgent()
        registry.register(agent)
        
        assert registry.get("unregister_agent") is not None
        assert "test_domain" in registry.get_domains()
        
        result = registry.unregister("unregister_agent")
        assert result is True
        
        assert registry.get("unregister_agent") is None
        assert "test_domain" not in registry.get_domains()

    def test_agent_listing_by_domain_and_capability(self):
        """Agents should be listable by domain and capability."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability
        )
        
        class TestAgent(Agent):
            def __init__(self, agent_id: str, domain: str, capabilities: List[AgentCapability]):
                self._agent_id = agent_id
                self._domain = domain
                self._capabilities = capabilities
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id=self._agent_id,
                    name=f"Agent-{self._agent_id}",
                    domain=self._domain,
                    capabilities=self._capabilities
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
        
        registry = AgentRegistry()
        
        agent1 = TestAgent("agent1", "energy", [AgentCapability.REASONING, AgentCapability.EVIDENCE_FILTER])
        agent2 = TestAgent("agent2", "energy", [AgentCapability.REASONING])
        agent3 = TestAgent("agent3", "finance", [AgentCapability.REASONING, AgentCapability.SYNTHESIS])
        
        registry.register(agent1, domain="energy")
        registry.register(agent2, domain="energy")
        registry.register(agent3, domain="finance")
        
        energy_agents = registry.list_by_domain("energy")
        assert len(energy_agents) == 2
        
        finance_agents = registry.list_by_domain("finance")
        assert len(finance_agents) == 1
        
        reasoning_agents = registry.list_by_capability(AgentCapability.REASONING)
        assert len(reasoning_agents) == 3
        
        synthesis_agents = registry.list_by_capability(AgentCapability.SYNTHESIS)
        assert len(synthesis_agents) == 1

    def test_agent_state_transitions(self):
        """Agent state transitions should follow valid paths."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class TestAgent(Agent):
            def __init__(self):
                self._started = False
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="state_agent",
                    name="StateAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                self._started = True
            
            def stop(self):
                self._started = False
        
        registry = AgentRegistry()
        agent = TestAgent()
        registered = registry.register(agent)
        
        assert registered.state == AgentState.REGISTERED
        
        registry.start_agent("state_agent")
        assert registered.state == AgentState.RUNNING
        
        registry.stop_agent("state_agent")
        assert registered.state == AgentState.STOPPED
        
        registry.start_agent("state_agent")
        assert registered.state == AgentState.RUNNING
        
        registry.unregister("state_agent")
        assert registered.state == AgentState.UNREGISTERED

    def test_start_stop_all_agents(self):
        """Starting and stopping all agents should work correctly."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class TestAgent(Agent):
            def __init__(self, agent_id: str):
                self._agent_id = agent_id
                self._started = False
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id=self._agent_id,
                    name=f"Agent-{self._agent_id}",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                self._started = True
            
            def stop(self):
                self._started = False
        
        registry = AgentRegistry()
        
        for i in range(5):
            agent = TestAgent(f"agent_{i}")
            registry.register(agent)
        
        start_results = registry.start_all()
        assert all(start_results.values())
        assert all(r.state == AgentState.RUNNING for r in registry.list_all())
        
        stop_results = registry.stop_all()
        assert all(stop_results.values())
        assert all(r.state == AgentState.STOPPED for r in registry.list_all())


class TestAgentHealthChecks:
    """Test agent health check functionality."""

    def test_health_check_running_agent(self):
        """Health check for running agent should return healthy status."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class HealthyAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="healthy_agent",
                    name="HealthyAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                pass
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "healthy", "custom_metric": 100}
        
        registry = AgentRegistry()
        agent = HealthyAgent()
        registered = registry.register(agent)
        registry.start_agent("healthy_agent")
        
        health = registry.health_check("healthy_agent")
        
        assert health["agent_id"] == "healthy_agent"
        assert health["state"] == "running"
        assert health["is_healthy"] is True
        assert health["custom_metric"] == 100

    def test_health_check_stopped_agent(self):
        """Health check for stopped agent should reflect state."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class TestAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="stopped_agent",
                    name="StoppedAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "stopped"}
        
        registry = AgentRegistry()
        agent = TestAgent()
        registered = registry.register(agent)
        
        health = registry.health_check("stopped_agent")
        
        assert health["state"] == "registered"
        assert health["is_healthy"] is False

    def test_health_check_error_agent(self):
        """Agent with errors should be marked unhealthy after threshold."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState, RegisteredAgent
        )
        
        class TestAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="error_agent",
                    name="ErrorAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
        
        registry = AgentRegistry()
        agent = TestAgent()
        registered = registry.register(agent)
        registry.start_agent("error_agent")
        
        for i in range(5):
            registered.record_error(f"Error {i}")
        
        health = registry.health_check("error_agent")
        
        assert health["state"] == "error"
        assert health["error_count"] == 5
        assert health["is_healthy"] is False

    def test_health_check_with_hooks(self):
        """Health check should include hook-provided metadata."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, LifecycleHooks
        )
        
        class TestAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="hook_agent",
                    name="HookAgent",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "healthy"}
        
        registry = AgentRegistry()
        agent = TestAgent()
        registered = registry.register(agent)
        registry.start_agent("hook_agent")
        
        def custom_health_check(agent):
            return {"hook_data": "custom_value", "timestamp": time.time()}
        
        registered.hooks.on_health_check = custom_health_check
        
        health = registry.health_check("hook_agent")
        
        assert "hook_data" in health
        assert health["hook_data"] == "custom_value"

    def test_bulk_health_check(self):
        """Health check for all agents should return aggregated stats."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class TestAgent(Agent):
            def __init__(self, agent_id: str, should_fail: bool = False):
                self._agent_id = agent_id
                self._should_fail = should_fail
                self._started = False
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id=self._agent_id,
                    name=f"Agent-{self._agent_id}",
                    domain="test",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"result": "test"}
            
            def start(self):
                self._started = True
            
            def health_check(self) -> Dict[str, Any]:
                if self._should_fail:
                    return {"status": "unhealthy"}
                return {"status": "healthy"}
        
        registry = AgentRegistry()
        
        for i in range(3):
            agent = TestAgent(f"agent_{i}")
            registry.register(agent)
            registry.start_agent(f"agent_{i}")
        
        failing_agent = TestAgent("failing_agent", should_fail=True)
        registry.register(failing_agent)
        registry.start_agent("failing_agent")
        
        health = registry.health_check()
        
        assert health["total_agents"] == 4
        assert "running" in health["by_state"]
        assert health["by_state"]["running"] == 4
        assert len(health["agents"]) == 4


class TestMultiAgentOrchestratorIntegration:
    """Test multi-agent orchestrator with registry integration."""

    def test_orchestrator_registers_default_agents(self):
        """Orchestrator should register default specialist agents."""
        from reasoning.multi_agent import get_multi_agent_orchestrator, reset_multi_agent_orchestrator
        from reasoning.agent_registry import get_agent_registry, reset_agent_registry
        
        reset_multi_agent_orchestrator()
        reset_agent_registry()
        
        orchestrator = get_multi_agent_orchestrator()
        
        stats = orchestrator.get_stats()
        assert stats["active_agents"] >= 12
        assert len(stats["domains"]) >= 10
        assert stats["registry"]["total_agents"] >= 12

    def test_orchestrator_custom_agent_registration(self):
        """Orchestrator should support custom agent registration."""
        from reasoning.multi_agent import get_multi_agent_orchestrator, reset_multi_agent_orchestrator
        from reasoning.agent_registry import get_agent_registry, reset_agent_registry, AgentCapability
        from reasoning.agent_registry import Agent, AgentDescriptor
        
        reset_multi_agent_orchestrator()
        reset_agent_registry()
        
        orchestrator = get_multi_agent_orchestrator()
        
        class CustomAgent(Agent):
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id="custom_agent",
                    name="CustomAgent",
                    domain="custom",
                    capabilities=[AgentCapability.REASONING, AgentCapability.PLANNING]
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"agent_id": "custom_agent", "domain": "custom", "summary": "Custom analysis"}
        
        registered = orchestrator.register_agent("custom", CustomAgent())
        
        assert registered.descriptor.agent_id == "custom_agent"
        assert registered.descriptor.domain == "custom"
        
        stats = orchestrator.get_stats()
        assert "custom" in stats["domains"]

    def test_orchestrator_agent_health_monitoring(self):
        """Orchestrator should provide agent health information."""
        from reasoning.multi_agent import get_multi_agent_orchestrator, reset_multi_agent_orchestrator
        from reasoning.agent_registry import get_agent_registry, reset_agent_registry
        
        reset_multi_agent_orchestrator()
        reset_agent_registry()
        
        orchestrator = get_multi_agent_orchestrator()
        
        health = orchestrator.get_agent_health()
        
        assert "total_agents" in health
        assert health["total_agents"] >= 12
        assert "by_state" in health
        assert "agents" in health

    def test_orchestrator_start_stop_all(self):
        """Orchestrator should be able to start/stop all agents."""
        from reasoning.multi_agent import get_multi_agent_orchestrator, reset_multi_agent_orchestrator
        from reasoning.agent_registry import get_agent_registry, reset_agent_registry
        
        reset_multi_agent_orchestrator()
        reset_agent_registry()
        
        orchestrator = get_multi_agent_orchestrator()
        
        stop_results = orchestrator.stop()
        assert all(stop_results.values())
        
        start_results = orchestrator.start()
        assert all(start_results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])