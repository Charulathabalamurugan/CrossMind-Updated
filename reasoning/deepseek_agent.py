from typing import Any, Dict, List, Optional


class DeepSeekAgent:
    """Backward-compatible deep reasoning agent wrapper for the runtime stack."""

    def __init__(self, model_name: str = "ZAYA1-8B"):
        self.model_name = model_name

    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        context_text = "\n".join(context or [])
        return f"[deepseek:{self.model_name}] {prompt}\n\nContext:\n{context_text}".strip()

    def reason(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "query": query,
            "output_text": self.generate(query, [str(item.get('payload', {}).get('content', '')) for item in (evidence or [])]),
            "confidence_score": 0.78,
            "cited_evidence_ids": [item.get("id") for item in (evidence or []) if item.get("id")],
        }


def get_zaya1_8b_agent() -> DeepSeekAgent:
    return DeepSeekAgent()


__all__ = ["DeepSeekAgent", "get_zaya1_8b_agent"]
