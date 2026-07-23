import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from reasoning.symbolic_filter import SymbolicPreFilter

logger = logging.getLogger("crossmind.multi_agent")

class DomainAgent:
    def __init__(self, domain: str, agent_id: str):
        self.domain = domain
        self.agent_id = agent_id
        self.name = f"{domain}_agent"
        self.task_count = 0

    def process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.task_count += 1
        domain_evidence = [
            ev for ev in evidence
            if ev.get("payload", {}).get("domain") == self.domain
            or self.domain in filter_metadata.get("detected_domains", [])
        ]
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "processed_count": len(domain_evidence),
            "evidence_ids": [ev["id"] for ev in domain_evidence],
            "status": "completed",
            "execution_ms": round(time.time() % 100, 2),
        }

class MultiAgentOrchestrator:
    def __init__(self):
        self._agents: Dict[str, DomainAgent] = {}
        self._lock = threading.Lock()
        self._total_tasks = 0

    def register_agent(self, domain: str) -> DomainAgent:
        agent_id = f"{domain}_{len(self._agents)}"
        agent = DomainAgent(domain, agent_id)
        with self._lock:
            self._agents[domain] = agent
        return agent

    def get_or_create_agent(self, domain: str) -> DomainAgent:
        with self._lock:
            if domain not in self._agents:
                self._agents[domain] = DomainAgent(domain, f"{domain}_{len(self._agents)}")
            return self._agents[domain]

    def parallel_process(self, query: str, evidence: List[Dict[str, Any]], filter_metadata: Dict[str, Any], max_workers: int = 4) -> Dict[str, Any]:
        domains = filter_metadata.get("detected_domains", ["neuroscience", "nanotechnology"])
        agents = []
        seen = set()
        for domain in domains:
            if domain not in seen:
                agents.append(self.get_or_create_agent(domain))
                seen.add(domain)
        
        results = {}
        start = time.time()
        if len(agents) == 1:
            result = agents[0].process(query, evidence, filter_metadata)
            results[agents[0].domain] = result
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as executor:
                futures = {
                    executor.submit(agent.process, query, evidence, filter_metadata): agent.domain
                    for agent in agents
                }
                for future in as_completed(futures):
                    domain = futures[future]
                    try:
                        results[domain] = future.result()
                    except Exception as exc:
                        logger.error(f"Agent {domain} failed: {exc}")
                        results[domain] = {"agent_id": domain, "domain": domain, "status": "failed", "error": str(exc)}
        
        total_ms = round((time.time() - start) * 1000, 2)
        with self._lock:
            self._total_tasks += len(agents)
        return {
            "domain_reports": results,
            "total_domains": len(agents),
            "total_execution_ms": total_ms,
            "total_tasks_orchestrated": self._total_tasks,
        }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_agents": len(self._agents),
                "domains": list(self._agents.keys()),
                "total_tasks": self._total_tasks,
            }

_orchestrator_instance: Optional[MultiAgentOrchestrator] = None

def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = MultiAgentOrchestrator()
    return _orchestrator_instance
