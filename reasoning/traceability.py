"""Evidence passage selection for transparent, inspectable answers."""
import re
from typing import Any, Dict, List

STOP_WORDS = {"a", "an", "and", "are", "between", "for", "find", "how", "in", "is", "of", "the", "to", "what", "with"}


def build_evidence_traces(query: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the strongest sentence from each result and the matched query terms."""
    query_terms = {term.lower() for term in re.findall(r"[\w'-]+", query) if len(term) > 2 and term.lower() not in STOP_WORDS}
    traces = []
    for item in evidence:
        payload = item.get("payload", {})
        sentences = re.split(r"(?<=[.!?])\s+", payload.get("content", ""))
        ranked = []
        for sentence in sentences:
            tokens = {term.lower() for term in re.findall(r"[\w'-]+", sentence)}
            matched = sorted(query_terms & tokens)
            ranked.append((len(matched), sentence.strip(), matched))
        _, passage, matched = max(ranked, default=(0, "", []), key=lambda value: value[0])
        traces.append({
            "evidence_id": item.get("id"),
            "title": payload.get("title", "Untitled"),
            "passage": passage or payload.get("content", "")[:400],
            "matched_query_terms": matched,
            "support_score": round(float(item.get("score", 0.0)), 3),
        })
    return traces
