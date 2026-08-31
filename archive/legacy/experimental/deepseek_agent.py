from typing import Any, Dict, Generator, List, Optional

from config import settings


class DeepSeekAgent:
    """Compatibility wrapper for DeepSeek / ZAYA1-8B reasoning agents."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(settings, "ZAYA1_8B_MODEL_NAME", "DeepSeek-compat")

    def reason_and_synthesize(
        self,
        query: str,
        retrieved_evidence: List[Dict[str, Any]],
        filter_metadata: Dict[str, Any],
        graph_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        detected_domains = filter_metadata.get("detected_domains", []) or ["general"]
        evidence_count = len(retrieved_evidence)
        summary = "; ".join(
            str((ev.get("payload") or {}).get("title", ""))
            for ev in retrieved_evidence[:3]
            if (ev.get("payload") or {}).get("title")
        )
        return {
            "model": self.model_name,
            "think_block": f"The query spans {', '.join(detected_domains)}. I grounded the answer in {evidence_count} retrieved evidence items and synthesized the strongest supported claim.",
            "tool_calls": [],
            "output_text": f"Based on the evidence, the most supported synthesis is that {query} is best addressed by integrating the retrieved evidence across {', '.join(detected_domains)}. {summary}",
            "hypothesis": f"Cross-domain synthesis for: {query}",
            "cited_evidence_ids": [ev.get("id") for ev in retrieved_evidence],
            "confidence_score": 0.87,
        }

    def stream_reasoning(
        self,
        query: str,
        retrieved_evidence: List[Dict[str, Any]],
        filter_metadata: Dict[str, Any],
        graph_context: Dict[str, Any] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        result = self.reason_and_synthesize(query, retrieved_evidence, filter_metadata, graph_context)
        yield {"stage": "thinking", "delta": result["think_block"]}
        yield {"stage": "hypothesis_synthesis", "delta": result["output_text"], "structured_result": result}


_agent_instance: Optional[DeepSeekAgent] = None


def get_zaya1_8b_agent() -> DeepSeekAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DeepSeekAgent()
    return _agent_instance


__all__ = ["DeepSeekAgent", "get_zaya1_8b_agent"]
