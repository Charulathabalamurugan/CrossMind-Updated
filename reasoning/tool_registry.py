"""Static, one-way tool registry for the CrossMind neuro-symbolic agent.

Tools are declared once (static contract) and the orchestrator/agent invokes
them through ``ToolRegistry.invoke``. A tool never holds a reference back to
the agent/orchestrator, enforcing a strictly one-way ``agent -> tools``
boundary: tools consume input and return data, they never call back into the
planner.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from config import settings

logger = logging.getLogger("crossmind.tool_registry")


TOOL_SPEC: List[Dict[str, Any]] = [
    {
        "name": "retrieve_hybrid",
        "description": "Hybrid dense + graph retrieval over the knowledge base with RBAC.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_role": {"type": "string"},
                "allowed_domains": {"type": "array"},
                "top_k": {"type": "integer"},
                "filter_metadata": {"type": "object"},
            },
        },
        "one_way": True,
    },
    {
        "name": "retrieve_graph",
        "description": "GraphRAG context expansion over the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "retrieved_evidence": {"type": "array"},
                "query_entities": {"type": "array"},
            },
        },
        "one_way": True,
    },
    {
        "name": "generate",
        "description": "Generate a reasoning result from evidence via LiteLLM/ZAYA with fallback.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "retrieved_evidence": {"type": "array"},
                "filter_metadata": {"type": "object"},
                "graph_context": {"type": "object"},
            },
        },
        "one_way": True,
    },
    {
        "name": "verify_symbolic",
        "description": "Formal symbolic verification (Z3) of the generated hypothesis.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_result": {"type": "object"},
                "retrieved_evidence": {"type": "array"},
            },
        },
        "one_way": True,
    },
    {
        "name": "check_anomaly",
        "description": "Contextual anomaly detection over an input.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {"type": "string"},
                "data": {},
                "metadata": {"type": "object"},
            },
        },
        "one_way": True,
    },
]


class ToolRegistry:
    """Static, one-way tool registry.

    Tools are registered as callables. The registry itself holds no reference
    to the agent/orchestrator, so a tool cannot call back into the planner:
    the contract is strictly ``agent -> tools``.
    """

    ONE_WAY: bool = True

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def spec(self, name: str) -> Optional[Dict[str, Any]]:
        for entry in TOOL_SPEC:
            if entry["name"] == name:
                return entry
        return None

    def register(self, name: str, tool: Callable[..., Any]) -> None:
        if not self.ONE_WAY:
            raise RuntimeError("ToolRegistry enforces a one-way contract; ONE_WAY is False")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def invoke(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        spec = self.spec(name)
        if spec is not None and not spec.get("one_way", True):
            raise RuntimeError(f"Tool '{name}' violates the one-way agent -> tools contract")
        return self._tools[name](*args, **kwargs)

    def available(self) -> List[str]:
        return list(self._tools.keys())


def default_tool_registry(pipeline: Any) -> ToolRegistry:
    """Bind the canonical CrossMind tools to an orchestrator instance.

    Each closure delegates to an existing method/attribute on the pipeline,
    preserving current behaviour while routing every call through the registry
    so the ``agent -> tools`` boundary is explicit and one-way.
    """
    registry = ToolRegistry()

    registry.register(
        "retrieve_hybrid",
        lambda query, user_role, allowed_domains, top_k=5, filter_metadata=None, **kw: (
            pipeline._retrieve_evidence(
                query, user_role, allowed_domains, top_k, filter_metadata,
                is_simple_query=kw.get("is_simple_query", False),
            )
        ),
    )
    registry.register(
        "retrieve_graph",
        lambda retrieved_evidence, query_entities=None, **kw: (
            pipeline.knowledge_graph.graph_rag_context(retrieved_evidence, query_entities or [])
        ),
    )
    registry.register(
        "generate",
        lambda query, retrieved_evidence, filter_metadata, graph_context, **kw: (
            pipeline._generate_reasoning(
                query, retrieved_evidence, filter_metadata, graph_context, kw.get("classification")
            )
        ),
    )
    registry.register(
        "verify_symbolic",
        lambda agent_result, retrieved_evidence, **kw: (
            pipeline.z3_validator.validate_hypothesis(agent_result, retrieved_evidence)
            if settings.Z3_VALIDATION_ENABLED
            else None
        ),
    )
    registry.register(
        "check_anomaly",
        lambda context, data, metadata=None, **kw: (
            pipeline.anomaly_detectors.detect_sync(context, data, metadata or {})
        ),
    )

    return registry


__all__ = ["TOOL_SPEC", "ToolRegistry", "default_tool_registry"]
