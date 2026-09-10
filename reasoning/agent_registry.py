import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("crossmind.agent_registry")


class AgentState(Enum):
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentCapability(Enum):
    REASONING = "reasoning"
    EVIDENCE_FILTER = "evidence_filter"
    DOMAIN_RANKER = "domain_ranker"
    CONFIDENCE_CHECK = "confidence_check"
    SYNTHESIS = "synthesis"
    CRITIQUE = "critique"
    PLANNING = "planning"
    MEMORY = "memory"
    MESSAGE_BUS = "message_bus"
    CUSTOM = "custom"


@dataclass
class AgentDescriptor:
    agent_id: str
    name: str
    domain: str
    capabilities: List[AgentCapability]
    version: str = "1.0.0"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def has_capability(self, capability: AgentCapability) -> bool:
        return capability in self.capabilities

    def matches_tags(self, tags: List[str]) -> bool:
        return any(tag in self.tags for tag in tags)


@dataclass
class LifecycleHooks:
    on_register: Optional[Callable[["RegisteredAgent"], None]] = None
    on_start: Optional[Callable[["RegisteredAgent"], None]] = None
    on_stop: Optional[Callable[["RegisteredAgent"], None]] = None
    on_error: Optional[Callable[["RegisteredAgent", Exception], None]] = None
    on_health_check: Optional[Callable[["RegisteredAgent"], Dict[str, Any]]] = None

    def trigger_on_register(self, agent: "RegisteredAgent"):
        if self.on_register:
            try:
                self.on_register(agent)
            except Exception as e:
                logger.warning("on_register hook failed for %s: %s", agent.descriptor.agent_id, e)

    def trigger_on_start(self, agent: "RegisteredAgent"):
        if self.on_start:
            try:
                self.on_start(agent)
            except Exception as e:
                logger.warning("on_start hook failed for %s: %s", agent.descriptor.agent_id, e)

    def trigger_on_stop(self, agent: "RegisteredAgent"):
        if self.on_stop:
            try:
                self.on_stop(agent)
            except Exception as e:
                logger.warning("on_stop hook failed for %s: %s", agent.descriptor.agent_id, e)

    def trigger_on_error(self, agent: "RegisteredAgent", error: Exception):
        if self.on_error:
            try:
                self.on_error(agent, error)
            except Exception as e:
                logger.warning("on_error hook failed for %s: %s", agent.descriptor.agent_id, e)

    def trigger_on_health_check(self, agent: "RegisteredAgent") -> Dict[str, Any]:
        if self.on_health_check:
            try:
                return self.on_health_check(agent)
            except Exception as e:
                logger.warning("on_health_check hook failed for %s: %s", agent.descriptor.agent_id, e)
        return {}


class Agent(ABC):
    @abstractmethod
    def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_descriptor(self) -> AgentDescriptor:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "timestamp": time.time()}


@dataclass
class RegisteredAgent:
    descriptor: AgentDescriptor
    instance: Agent
    state: AgentState = AgentState.REGISTERED
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks)
    registered_at: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    health_metadata: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None

    def is_healthy(self) -> bool:
        return self.state == AgentState.RUNNING and self.error_count < 5

    def update_health(self, metadata: Dict[str, Any]):
        self.last_health_check = time.time()
        self.health_metadata.update(metadata)

    def record_error(self, error: str):
        self.error_count += 1
        self.last_error = error
        if self.error_count >= 5:
            self.state = AgentState.ERROR


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, RegisteredAgent] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        agent: Agent,
        hooks: Optional[LifecycleHooks] = None,
        domain: Optional[str] = None,
    ) -> RegisteredAgent:
        descriptor = agent.get_descriptor()
        agent_id = descriptor.agent_id

        with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"Agent with ID {agent_id} already registered")

            registered = RegisteredAgent(
                descriptor=descriptor,
                instance=agent,
                hooks=hooks or LifecycleHooks(),
                state=AgentState.REGISTERED,
            )

            self._agents[agent_id] = registered

            domain_key = domain or descriptor.domain.lower()
            self._domain_index.setdefault(domain_key, set()).add(agent_id)

            for capability in descriptor.capabilities:
                self._capability_index.setdefault(capability, set()).add(agent_id)

            registered.hooks.trigger_on_register(registered)
            logger.info("Registered agent: %s (domain: %s)", agent_id, domain_key)

            return registered

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            registered = self._agents.pop(agent_id, None)
            if not registered:
                return False

            domain_key = registered.descriptor.domain.lower()
            if domain_key in self._domain_index:
                self._domain_index[domain_key].discard(agent_id)
                if not self._domain_index[domain_key]:
                    del self._domain_index[domain_key]

            for capability in registered.descriptor.capabilities:
                if capability in self._capability_index:
                    self._capability_index[capability].discard(agent_id)
                    if not self._capability_index[capability]:
                        del self._capability_index[capability]

            registered.hooks.trigger_on_stop(registered)
            registered.state = AgentState.UNREGISTERED
            logger.info("Unregistered agent: %s", agent_id)
            return True

    def get(self, agent_id: str) -> Optional[RegisteredAgent]:
        with self._lock:
            return self._agents.get(agent_id)

    def list_by_domain(self, domain: str) -> List[RegisteredAgent]:
        with self._lock:
            agent_ids = self._domain_index.get(domain.lower(), set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def list_by_capability(self, capability: AgentCapability) -> List[RegisteredAgent]:
        with self._lock:
            agent_ids = self._capability_index.get(capability, set())
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def list_all(self) -> List[RegisteredAgent]:
        with self._lock:
            return list(self._agents.values())

    def get_domains(self) -> List[str]:
        with self._lock:
            return list(self._domain_index.keys())

    def start_agent(self, agent_id: str) -> bool:
        with self._lock:
            registered = self._agents.get(agent_id)
            if not registered:
                return False

            if registered.state not in (AgentState.REGISTERED, AgentState.STOPPED):
                logger.warning("Agent %s cannot start from state %s", agent_id, registered.state)
                return False

            registered.state = AgentState.STARTING
            try:
                registered.instance.start()
                registered.state = AgentState.RUNNING
                registered.hooks.trigger_on_start(registered)
                logger.info("Started agent: %s", agent_id)
                return True
            except Exception as e:
                registered.state = AgentState.ERROR
                registered.record_error(str(e))
                registered.hooks.trigger_on_error(registered, e)
                logger.exception("Failed to start agent %s", agent_id)
                return False

    def stop_agent(self, agent_id: str) -> bool:
        with self._lock:
            registered = self._agents.get(agent_id)
            if not registered:
                return False

            if registered.state != AgentState.RUNNING:
                logger.warning("Agent %s cannot stop from state %s", agent_id, registered.state)
                return False

            registered.state = AgentState.STOPPING
            try:
                registered.instance.stop()
                registered.state = AgentState.STOPPED
                registered.hooks.trigger_on_stop(registered)
                logger.info("Stopped agent: %s", agent_id)
                return True
            except Exception as e:
                registered.state = AgentState.ERROR
                registered.record_error(str(e))
                registered.hooks.trigger_on_error(registered, e)
                logger.exception("Failed to stop agent %s", agent_id)
                return False

    def start_all(self) -> Dict[str, bool]:
        with self._lock:
            agent_ids = list(self._agents.keys())
        results = {}
        for agent_id in agent_ids:
            results[agent_id] = self.start_agent(agent_id)
        return results

    def stop_all(self) -> Dict[str, bool]:
        with self._lock:
            agent_ids = list(self._agents.keys())
        results = {}
        for agent_id in agent_ids:
            results[agent_id] = self.stop_agent(agent_id)
        return results

    def health_check(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if agent_id:
                registered = self._agents.get(agent_id)
                if not registered:
                    return {"agent_id": agent_id, "status": "not_found"}
                return self._check_agent_health(registered)
            else:
                return {
                    "total_agents": len(self._agents),
                    "by_state": self._count_by_state(),
                    "agents": {aid: self._check_agent_health(ra) for aid, ra in self._agents.items()},
                }

    def _check_agent_health(self, registered: RegisteredAgent) -> Dict[str, Any]:
        health = {
            "agent_id": registered.descriptor.agent_id,
            "name": registered.descriptor.name,
            "domain": registered.descriptor.domain,
            "state": registered.state.value,
            "registered_at": registered.registered_at,
            "last_health_check": registered.last_health_check,
            "error_count": registered.error_count,
            "last_error": registered.last_error,
            "is_healthy": registered.is_healthy(),
        }

        hook_health = registered.hooks.trigger_on_health_check(registered)
        health.update(hook_health)

        instance_health = registered.instance.health_check()
        health.update(instance_health)

        registered.update_health(health)
        return health

    def _count_by_state(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for registered in self._agents.values():
            state_val = registered.state.value
            counts[state_val] = counts.get(state_val, 0) + 1
        return counts

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "domains": list(self._domain_index.keys()),
                "capabilities": {cap.value: len(ids) for cap, ids in self._capability_index.items()},
                "by_state": self._count_by_state(),
            }


_global_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def reset_agent_registry() -> None:
    global _global_registry
    if _global_registry is not None:
        _global_registry.stop_all()
    _global_registry = None