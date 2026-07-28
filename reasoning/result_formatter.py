import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("crossmind.result_formatter")

def format_pipeline_result(result: Dict[str, Any]) -> Dict[str, Any]:
    formatted = {
        "status": "success",
        "query": result.get("query", ""),
        "session_id": result.get("session_id"),
        "user_role": result.get("user_role", "researcher"),
    }
    summary = _build_summary(result)
    formatted["summary"] = summary
    evidence_list = _format_evidence(result.get("retrieved_evidence", []))
    formatted["evidence"] = evidence_list
    reasoning = _format_reasoning(result)
    formatted["reasoning"] = reasoning
    validation = _format_validation(result)
    formatted["validation"] = validation
    enriched = _format_enrichment(result)
    formatted["enrichment"] = enriched
    metrics = _format_metrics(result)
    formatted["metrics"] = metrics
    return formatted


def _build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    calibration = result.get("confidence_calibration", {})
    discovery = result.get("cross_domain_scoring", {})
    return {
        "decision": calibration.get("decision", "unknown"),
        "calibrated_confidence": calibration.get("calibrated_confidence", 0.0),
        "discovery_strength": discovery.get("overall_score", 0.0),
        "discovery_rating": discovery.get("rating", "unknown"),
        "final_hypothesis": result.get("agent_reasoning", {}).get("output_text", ""),
        "validation_passed": result.get("post_validation", {}).get("validated", False),
        "z3_validated": result.get("z3_formal_validation", {}).get("validated", False),
        "domains": result.get("pre_filter", {}).get("detected_domains", []),
        "entities": result.get("pre_filter", {}).get("extracted_entities", []),
    }


def _format_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for ev in evidence:
        payload = ev.get("payload", {})
        formatted.append({
            "id": ev.get("id"),
            "title": payload.get("title", "Untitled"),
            "domain": payload.get("domain", "general"),
            "year": payload.get("year", 0),
            "authors": payload.get("authors", []),
            "score": round(float(ev.get("score", 0.0)), 4),
            "source": ev.get("source", "vector_search"),
            "fusion_source": ev.get("retrieval_source", ["dense_vector"]),
        })
    return formatted


def _format_reasoning(result: Dict[str, Any]) -> Dict[str, Any]:
    agent = result.get("agent_reasoning", {})
    return {
        "think_block": agent.get("think_block", ""),
        "output_text": agent.get("output_text", ""),
        "hypothesis": agent.get("hypothesis", ""),
        "tool_calls": agent.get("tool_calls", []),
        "confidence_score": agent.get("confidence_score", 0.0),
        "model": agent.get("model", "unknown"),
    }


def _format_validation(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbolic": result.get("post_validation", {}),
        "z3_formal": result.get("z3_formal_validation", {}),
        "evidence_attribution": result.get("evidence_attribution", {}),
    }


def _format_enrichment(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "graph_rag": result.get("graph_rag", {}),
        "cross_domain_scoring": result.get("cross_domain_scoring", {}),
        "abductive_reasoning": result.get("abductive_reasoning", {}),
        "experimental_blueprint": result.get("experimental_blueprint", {}),
        "collaboration_recommendations": result.get("collaboration_recommendations", {}),
        "multi_agent_report": result.get("multi_agent_orchestration", {}),
        "bridge_scorer": {
            "computed": False,
        },
    }


def _format_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    perf = result.get("performance_metrics", {})
    return {
        "total_time_seconds": perf.get("total_time_seconds", 0.0),
        "pre_filter_ms": perf.get("pre_filter_ms", 0.0),
        "retrieval_ms": perf.get("retrieval_ms", 0.0),
        "agent_reasoning_time_seconds": perf.get("agent_reasoning_time_seconds", 0.0),
        "post_validation_ms": perf.get("post_validation_ms", 0.0),
        "retrieved_chunks_count": perf.get("retrieved_chunks_count", 0),
        "graph_nodes_count": perf.get("graph_nodes_count", 0),
        "multi_hop_paths_count": perf.get("multi_hop_paths_count", 0),
        "sparse_retrieval_ms": perf.get("sparse_retrieval_ms", 0.0),
        "dense_retrieval_ms": perf.get("dense_retrieval_ms", 0.0),
    }