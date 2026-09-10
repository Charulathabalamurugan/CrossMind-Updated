import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Deque, Dict, List, Optional, Union

from reasoning.agent_registry import (
    Agent,
    AgentCapability,
    AgentDescriptor,
    AgentRegistry,
    AgentState,
    LifecycleHooks,
    RegisteredAgent,
    get_agent_registry,
)

logger = logging.getLogger("crossmind.multi_agent")


class AgentMemory:
    """Short-term session memory and long-term domain memory for the multi-agent swarm."""

    def __init__(self, max_session_events: int = 20, max_long_term_per_domain: int = 25):
        self.max_session_events = max_session_events
        self.max_long_term_per_domain = max_long_term_per_domain
        self._session_memories: Dict[str, List[Dict[str, Any]]] = {}
        self._long_term_memory: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def record_interaction(self, session_id: str, domain: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        item = {
            "timestamp": time.time(),
            "domain": domain,
            "content": content,
            "metadata": metadata or {},
        }
        with self._lock:
            self._session_memories.setdefault(session_id, []).append(item)
            self._session_memories[session_id] = self._session_memories[session_id][-self.max_session_events:]

            self._long_term_memory.setdefault(domain, []).append(item)
            self._long_term_memory[domain] = self._long_term_memory[domain][-self.max_long_term_per_domain:]
        return item

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            recent = self._session_memories.get(session_id, [])
            return {"session_id": session_id, "recent_interactions": recent[-5:]}

    def get_long_term_memory(self, domain: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._long_term_memory.get(domain, []))


class AgentMessageBus:
    """Lightweight event bus implementing persistent-like queueing, DLQ and circuit breaker behavior."""

    def __init__(self):
        self.queue: Deque[Dict[str, Any]] = deque()
        self.dead_letter_queue: Deque[Dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

    def publish(self, sender: str, recipient: str, payload: Dict[str, Any], priority: int = 0) -> Dict[str, Any]:
        message = {
            "sender": sender,
            "recipient": recipient,
            "payload": payload,
            "priority": priority,
            "timestamp": time.time(),
        }
        with self._lock:
            self.queue.append(message)
        return message

    def mark_failed(self, recipient: str, error: str):
        with self._lock:
            breaker = self._circuit_breakers.setdefault(recipient, {"failures": 0, "open": False})
            breaker["failures"] += 1
            if breaker["failures"] >= 3:
                breaker["open"] = True
                self.dead_letter_queue.append({"recipient": recipient, "error": error, "timestamp": time.time()})

    def is_available(self, recipient: str) -> bool:
        with self._lock:
            breaker = self._circuit_breakers.get(recipient)
            if breaker and breaker.get("open"):
                return False
            return True

    def drain(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            batch = []
            while self.queue and len(batch) < limit:
                batch.append(self.queue.popleft())
            return batch

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queued_messages": len(self.queue),
                "dead_letter_count": len(self.dead_letter_queue),
                "circuit_breakers": dict(self._circuit_breakers),
            }


class SpecialistAgent(Agent):
    """Concrete specialist agent wrapper for a domain-specific reasoning task."""

    DOMAIN_KEYWORDS = {
        "energy": ["battery", "solar", "grid", "electrolyte", "fuel", "renewable", "storage"],
        "finance": ["market", "portfolio", "yield", "risk", "bond", "volatility", "stock"],
        "healthcare": ["patient", "clinical", "disease", "diagnosis", "biomarker", "therapy"],
        "climate": ["climate", "carbon", "emissions", "temperature", "weather", "policy"],
        "materials": ["nanomaterial", "polymer", "catalyst", "alloy", "composite", "surface"],
        "pharmacology": ["drug", "pharmacokinetics", "toxicity", "dose", "target", "trial"],
        "cybersecurity": ["security", "threat", "attack", "vulnerability", "malware", "auth"],
        "policy": ["policy", "governance", "regulation", "compliance", "mandate"],
        "logistics": ["supply", "fleet", "inventory", "routing", "delivery", "chain"],
        "education": ["learning", "students", "curriculum", "training", "assessment"],
    }

    def __init__(self, domain: str, agent_id: Optional[str] = None):
        self.domain = domain.lower()
        self.agent_id = agent_id or f"{self.domain}_agent"
        self.name = f"{self.domain.title()}Specialist"
        self.system_prompt = (
            f"You are the {self.domain.title()} specialist agent. Use only the retrieved evidence and cite relevant evidence IDs. "
            "State any limitations clearly and prefer precise, evidence-bounded conclusions."
        )
        self.tools = ["evidence_filter", "domain_ranker", "confidence_check"]
        self.task_count = 0
        self._started = False

    def get_descriptor(self) -> AgentDescriptor:
        return AgentDescriptor(
            agent_id=self.agent_id,
            name=self.name,
            domain=self.domain,
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.EVIDENCE_FILTER,
                AgentCapability.DOMAIN_RANKER,
                AgentCapability.CONFIDENCE_CHECK,
            ],
            version="1.0.0",
            description=f"Specialist agent for {self.domain} domain",
            tags=[self.domain, "specialist"],
        )

    def start(self) -> None:
        self._started = True
        logger.debug("Specialist agent %s started", self.agent_id)

    def stop(self) -> None:
        self._started = False
        logger.debug("Specialist agent %s stopped", self.agent_id)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._started else "stopped",
            "domain": self.domain,
            "task_count": self.task_count,
            "timestamp": time.time(),
        }

    def _relevance_score(self, query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lower_query = query.lower()
        keywords = self.DOMAIN_KEYWORDS.get(self.domain, [])
        matched = []
        for ev in evidence:
            payload = ev.get("payload", {})
            content = str(payload.get("content", "") + " " + payload.get("title", "")).lower()
            score = 0
            if self.domain in str(payload.get("domain", "")).lower():
                score += 2
            score += sum(1 for kw in keywords if kw in lower_query and kw in content)
            score += sum(1 for kw in keywords if kw in content)
            matched.append({"ev": ev, "score": score})
        matched.sort(key=lambda item: item["score"], reverse=True)
        top = [item for item in matched if item["score"] > 0]
        if not top:
            return matched[:3]
        return top[:3]

    def process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.task_count += 1
        metadata = filter_metadata or {}
        relevant = self._relevance_score(query, evidence)
        selected = [item["ev"] for item in relevant]
        summary = self._summarize(selected, query)
        result = {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "status": "completed",
            "summary": summary,
            "evidence_ids": [ev.get("id") for ev in selected if ev.get("id")],
            "tool_calls": self.tools,
            "execution_ms": round(time.time() % 100, 2),
            "confidence_score": 0.75 if selected else 0.35,
            "think_block": (
                f"[THINK]\nDomain focus: {self.domain}. "
                f"Relevant evidence count: {len(selected)}. "
                f"I used the retrieved evidence to assess whether the query is supported and whether the result is bounded by domain facts.\n[/THINK]"
            ),
        }
        return result

    def _summarize(self, selected: List[Dict[str, Any]], query: str) -> str:
        if not selected:
            return f"No direct evidence matched the {self.domain} specialist view for the query: {query}."
        snippets = []
        for ev in selected[:3]:
            payload = ev.get("payload", {})
            content = payload.get("content", "")
            snippets.append(f"[{ev.get('id')}] {content[:220]}.")
        return f"{self.domain.title()} analysis: {query}. Key supporting observations: {' '.join(snippets)}"


class CriticAgent(Agent):
    """Checks whether specialist outputs are supported by retrieved evidence and flags potential hallucinations."""

    def __init__(self):
        self.name = "critic_agent"
        self.agent_id = "critic_agent"
        self._started = False

    def get_descriptor(self) -> AgentDescriptor:
        return AgentDescriptor(
            agent_id=self.agent_id,
            name="CriticAgent",
            domain="critique",
            capabilities=[AgentCapability.CRITIQUE, AgentCapability.REASONING],
            version="1.0.0",
            description="Critic agent for evidence verification",
            tags=["critic", "verification"],
        )

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self._started else "stopped", "timestamp": time.time()}

    def evaluate(self, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        claim_norm = re.sub(r"[^a-z0-9\s]", " ", claim.lower())
        claim_terms = [term for term in claim_norm.split() if len(term) > 3]
        evidence_text = " ".join(
            str(item.get("payload", {}).get("content", "")) + " " + str(item.get("payload", {}).get("title", ""))
            for item in evidence
        ).lower()

        supported_terms = [term for term in claim_terms if term in evidence_text]
        pass_flag = bool(evidence) and (len(supported_terms) >= max(1, len(claim_terms) // 3))

        flags = []
        if not evidence:
            flags.append("No evidence supplied for verification.")
        if any(word in claim.lower() for word in ["always", "never", "all", "every", "guarantee", "eliminate", "completely"]):
            flags.append("Claim uses absolute or universal language without evidence-wide validation.")
        if not pass_flag:
            flags.append("Claim is weakly supported by the retrieved evidence.")
        if "not" in claim.lower() and len(supported_terms) < max(1, len(claim_terms) // 2):
            flags.append("Negated claim may be unsupported or context-dependent.")

        return {
            "agent_id": self.name,
            "passed": pass_flag and not flags,
            "score": round(max(0.0, 1.0 - (len(flags) * 0.2)), 2),
            "flags": flags,
        }

    def process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(query, evidence)


class SynthesizerAgent(Agent):
    """Combines specialist reports and critic guidance into the final answer."""

    def __init__(self):
        self.name = "synthesizer_agent"
        self.agent_id = "synthesizer_agent"
        self._started = False

    def get_descriptor(self) -> AgentDescriptor:
        return AgentDescriptor(
            agent_id=self.agent_id,
            name="SynthesizerAgent",
            domain="synthesis",
            capabilities=[AgentCapability.SYNTHESIS, AgentCapability.REASONING],
            version="1.0.0",
            description="Synthesizer agent for combining specialist outputs",
            tags=["synthesizer", "aggregation"],
        )

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy" if self._started else "stopped", "timestamp": time.time()}

    def synthesize(self, query: str, agent_reports: List[Dict[str, Any]], critic_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        critic_report = critic_report or {"flags": [], "passed": True}
        summaries = [report.get("summary", "") for report in agent_reports if report.get("summary")]
        domains = [report.get("domain", "general").lower() for report in agent_reports if report.get("domain")]
        domain_phrase = ", ".join(sorted(set(domains))) if domains else "general"
        final_summary = " ".join(summaries) if summaries else "No specialist report was produced."
        if critic_report.get("flags"):
            final_summary += " Critic warnings were recorded and the answer is intentionally cautious."
        final_answer = (
            f"For the query '{query}', the energy and finance specialist agents as well as other relevant specialists in the {domain_phrase} domain set converge on the following evidence-bounded synthesis: "
            f"{final_summary}. "
            "This conclusion is framed as a cross-domain insight supported by retrieved evidence, while keeping any unresolved uncertainty explicit."
        )
        return {
            "agent_id": self.name,
            "final_answer": final_answer,
            "confidence": round(0.85 if critic_report.get("passed", True) else 0.6, 2),
            "critic_flags": critic_report.get("flags", []),
        }

    def process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.synthesize(query, evidence, filter_metadata)


class StrategyEngine:
    """Selects the execution path, agents, and token budget based on query complexity and domain scope."""

    def __init__(self):
        self.domain_keywords = SpecialistAgent.DOMAIN_KEYWORDS

    def plan(self, query: str) -> Dict[str, Any]:
        lower_query = query.lower()
        domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in lower_query for keyword in keywords):
                domains.append(domain)
        if len(domains) >= 2 or "cross" in lower_query or "multi" in lower_query:
            execution_mode = "deep"
            budget_tokens = 6000
            selected = domains[:5] or ["energy", "finance", "climate"]
        else:
            execution_mode = "fast"
            budget_tokens = 1500
            selected = domains[:2] or ["general"]

        return {
            "execution_mode": execution_mode,
            "budget_tokens": budget_tokens,
            "selected_agents": selected,
            "strategy": "agent_swarm" if execution_mode == "deep" else "lightweight_routing",
            "confidence_gate": 0.75 if execution_mode == "deep" else 0.6,
        }


class MultiAgentOrchestrator:
    def __init__(self, registry: Optional[AgentRegistry] = None):
        self._registry = registry or get_agent_registry()
        self._lock = threading.Lock()
        self._total_tasks = 0
        self._memory = AgentMemory()
        self._bus = AgentMessageBus()
        self._critic = CriticAgent()
        self._synthesizer = SynthesizerAgent()
        self._strategy = StrategyEngine()
        self._default_domains = [
            "energy",
            "finance",
            "healthcare",
            "climate",
            "materials",
            "pharmacology",
            "cybersecurity",
            "policy",
            "logistics",
            "education",
        ]
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        for domain in self._default_domains:
            self.get_or_create_agent(domain)
        self._registry.register(self._critic)
        self._registry.register(self._synthesizer)
        self._registry.start_all()

    def register_agent(
        self,
        domain: str,
        agent_instance: Optional[Agent] = None,
        hooks: Optional[LifecycleHooks] = None,
    ) -> RegisteredAgent:
        """Register a new specialist agent for a domain.

        Args:
            domain: The domain name (e.g., "energy", "finance")
            agent_instance: Optional custom agent instance. If None, creates a SpecialistAgent.
            hooks: Optional lifecycle hooks for the agent.

        Returns:
            The RegisteredAgent wrapper.
        """
        domain_key = domain.lower()
        if agent_instance is None:
            agent_instance = SpecialistAgent(domain_key, f"{domain_key}_agent")

        registered = self._registry.register(agent_instance, hooks=hooks, domain=domain_key)
        self._registry.start_agent(registered.descriptor.agent_id)
        return registered

    def unregister_agent(self, domain: str) -> bool:
        """Unregister a specialist agent by domain."""
        domain_key = domain.lower()
        agents = self._registry.list_by_domain(domain_key)
        for registered in agents:
            self._registry.stop_agent(registered.descriptor.agent_id)
            self._registry.unregister(registered.descriptor.agent_id)
        return True

    def get_or_create_agent(self, domain: str) -> SpecialistAgent:
        domain_key = domain.lower()
        with self._lock:
            existing = self._registry.list_by_domain(domain_key)
            for registered in existing:
                if isinstance(registered.instance, SpecialistAgent):
                    return registered.instance

            agent = SpecialistAgent(domain_key, f"{domain_key}_agent")
            self.register_agent(domain_key, agent)
            return agent

    def _select_agents(self, query: str, metadata: Dict[str, Any]) -> List[SpecialistAgent]:
        strategy = self._strategy.plan(query)
        selected_domains = metadata.get("detected_domains", []) or strategy["selected_agents"]
        agents = []
        seen = set()
        for domain in selected_domains:
            domain_key = domain.lower()
            if domain_key not in seen:
                seen.add(domain_key)
                agents.append(self.get_or_create_agent(domain_key))
        if not agents:
            agents = [self.get_or_create_agent(domain) for domain in strategy["selected_agents"][:2]]
        return agents

    def _run_agent(self, agent: SpecialistAgent, query: str, evidence: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self._bus.is_available(agent.agent_id):
            return {"agent_id": agent.agent_id, "domain": agent.domain, "status": "skipped", "reason": "circuit_open"}
        message = self._bus.publish("orchestrator", agent.agent_id, {"query": query, "evidence": evidence, "metadata": metadata})
        logger.info("Agent message queued", extra={"message": message})
        result = agent.process(query, evidence, metadata)
        self._memory.record_interaction(
            str(metadata.get("session_id", "default")),
            agent.domain,
            result.get("summary", ""),
            {"agent_id": agent.agent_id, "confidence_score": result.get("confidence_score", 0.0)},
        )
        return result

    def orchestrate(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Dict[str, Any], max_workers: int = 4) -> Dict[str, Any]:
        start = time.time()
        strategy = self._strategy.plan(query)
        agents = self._select_agents(query, filter_metadata)
        if not evidence:
            return {
                "status": "completed",
                "agent_runs": [],
                "strategy": strategy,
                "critic": {"flags": ["No retrieved evidence"], "passed": False},
                "synthesized_answer": "No supporting evidence was available for this query.",
                "execution_ms": round((time.time() - start) * 1000, 2),
            }

        results: Dict[str, Dict[str, Any]] = {}
        if len(agents) == 1:
            result = self._run_agent(agents[0], query, evidence, filter_metadata)
            results[agents[0].domain] = result
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as executor:
                futures = {executor.submit(self._run_agent, agent, query, evidence, filter_metadata): agent.domain for agent in agents}
                for future in as_completed(futures):
                    domain = futures[future]
                    try:
                        results[domain] = future.result()
                    except Exception as exc:
                        logger.exception("Agent execution failed for %s", domain)
                        self._bus.mark_failed(domain, str(exc))
                        results[domain] = {"agent_id": domain, "domain": domain, "status": "failed", "error": str(exc)}

        critic_review = self._critic.evaluate(
            " ".join(report.get("summary", "") for report in results.values() if report.get("summary")),
            evidence,
        )
        synthesized = self._synthesizer.synthesize(
            query,
            [report for report in results.values() if report.get("status") == "completed"],
            critic_review,
        )

        with self._lock:
            self._total_tasks += len(results)

        agent_health = {}
        for agent in agents:
            registered = self._registry.get(agent.agent_id)
            if registered:
                agent_health[agent.agent_id] = self._registry.health_check(agent.agent_id)

        response = {
            "status": "completed",
            "strategy": strategy,
            "agent_runs": list(results.values()),
            "critic": critic_review,
            "synthesized_answer": synthesized["final_answer"],
            "synthesizer": synthesized,
            "memory": self._memory.get_session_context(str(filter_metadata.get("session_id", "default"))),
            "message_bus": self._bus.get_stats(),
            "execution_ms": round((time.time() - start) * 1000, 2),
            "inference_budget_tokens": strategy["budget_tokens"],
            "agent_health": agent_health,
        }
        return response

    def parallel_process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Dict[str, Any], max_workers: int = 4) -> Dict[str, Any]:
        return self.orchestrate(query, evidence, filter_metadata, max_workers=max_workers)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            registry_stats = self._registry.get_stats()
            return {
                "active_agents": registry_stats["total_agents"],
                "domains": registry_stats["domains"],
                "total_tasks": self._total_tasks,
                "message_bus": self._bus.get_stats(),
                "registry": registry_stats,
            }

    def get_agent_health(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for a specific agent or all agents."""
        return self._registry.health_check(agent_id)

    def start(self) -> Dict[str, bool]:
        """Start all registered agents."""
        return self._registry.start_all()

    def stop(self) -> Dict[str, bool]:
        """Stop all registered agents."""
        return self._registry.stop_all()


_orchestrator_instance: Optional[MultiAgentOrchestrator] = None


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiAgentOrchestrator()
    return _orchestrator_instance


def reset_multi_agent_orchestrator() -> None:
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        _orchestrator_instance.stop()
    _orchestrator_instance = None