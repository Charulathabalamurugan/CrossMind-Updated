from unittest.mock import Mock

import pytest

from reasoning.tool_registry import TOOL_SPEC, ToolRegistry, default_tool_registry


REQUIRED_TOOLS = {"retrieve_hybrid", "retrieve_graph", "generate", "verify_symbolic", "check_anomaly"}


class _StubPipeline:
    """Minimal stand-in exposing the attributes the default tools delegate to."""

    def __init__(self):
        self.knowledge_graph = Mock()
        self.z3_validator = Mock()
        self.anomaly_detectors = Mock()
        self._retrieve_evidence = Mock(return_value=[{"id": "doc:1", "payload": {}}])
        self._generate_reasoning = Mock(return_value={"model": "lite", "output_text": "ok"})


def test_tool_spec_declares_all_required_tools_one_way():
    names = {entry["name"] for entry in TOOL_SPEC}
    assert REQUIRED_TOOLS <= names
    for entry in TOOL_SPEC:
        assert entry.get("one_way") is True


def test_registry_starts_empty_and_enforces_one_way_contract():
    registry = ToolRegistry()
    assert registry.available() == []
    assert registry.has("generate") is False
    with pytest.raises(KeyError):
        registry.invoke("generate", query="q")


def test_registry_register_rejects_duplicates():
    registry = ToolRegistry()
    registry.register("ping", lambda **kw: "pong")
    assert registry.has("ping")
    with pytest.raises(ValueError):
        registry.register("ping", lambda **kw: "pong2")


def test_default_registry_registers_all_tools_and_delegates():
    pipeline = _StubPipeline()
    registry = default_tool_registry(pipeline)

    assert REQUIRED_TOOLS <= set(registry.available())

    evidence = registry.invoke("retrieve_hybrid", query="q", user_role="researcher", allowed_domains=["energy"], top_k=3)
    assert evidence == [{"id": "doc:1", "payload": {}}]
    pipeline._retrieve_evidence.assert_called_once_with(
        "q", "researcher", ["energy"], 3, None, is_simple_query=False
    )

    registry.invoke("retrieve_graph", retrieved_evidence=[], query_entities=["x"])
    pipeline.knowledge_graph.graph_rag_context.assert_called_once_with([], ["x"])

    result = registry.invoke("generate", query="q", retrieved_evidence=[], filter_metadata={}, graph_context={})
    assert result == {"model": "lite", "output_text": "ok"}
    pipeline._generate_reasoning.assert_called_once()

    registry.invoke("check_anomaly", context="retrieval_query", data="q", metadata={"session_id": "s"})
    pipeline.anomaly_detectors.detect_sync.assert_called_once_with("retrieval_query", "q", {"session_id": "s"})
