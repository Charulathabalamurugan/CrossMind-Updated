import time
import logging
from typing import Dict, Any, Generator
from reasoning.symbolic_filter import SymbolicPreFilter, SymbolicPostValidator
from reasoning.rxg_nano_agent import YuukiRxGNanoAgent
from vector_store.qdrant_engine import get_qdrant_engine
from ingestion.embedding import get_embedder
from reasoning.knowledge_graph import get_knowledge_graph, DiscoveryScorer, ConfidenceCalibrator
from reasoning.traceability import build_evidence_traces
from reasoning.memory_service import get_memory_service
from reasoning.abductive_engine import AbductiveReasoningEngine

logger = logging.getLogger("crossmind.neuro_symbolic")

class NeuroSymbolicPipeline:
    """
    Complete Phase 3 Orchestrator:
    Step 3a: Symbolic Pre-Filter (<50ms)
    Step 2: Qdrant Vector Retrieval (~5-15ms)
    Step 3b: Yuuki RxG Nano Agentic Reasoning (~2-4s)
    Step 3c: Symbolic Post-Validation (<50ms)
    """
    def __init__(self):
        self.pre_filter = SymbolicPreFilter()
        self.post_validator = SymbolicPostValidator()
        self.agent = YuukiRxGNanoAgent()
        self.embedder = get_embedder()
        self.vector_engine = get_qdrant_engine()
        self.knowledge_graph = get_knowledge_graph()
        self.memory_service = get_memory_service()
        self.abductive_engine = AbductiveReasoningEngine()
        self._cache = {}

    def _enrich_result(self, query: str, user_role: str, filter_metadata: Dict[str, Any], retrieved_evidence: list, agent_result: Dict[str, Any], validation_result: Dict[str, Any], total_time_s: float, agent_time_s: float, confidence_thresholds: Dict[str, float] = None, abductive_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Attach GraphRAG, discovery score, calibrated decision confidence, and abductive reasoning."""
        graph_context = self.knowledge_graph.graph_rag_context(
            retrieved_evidence, filter_metadata.get("extracted_entities", [])
        )
        discovery_score = DiscoveryScorer.score(retrieved_evidence, graph_context)
        confidence_calibration = ConfidenceCalibrator.calibrate(
            agent_result.get("confidence_score", 0.0), discovery_score, validation_result, confidence_thresholds
        )
        agent_result = dict(agent_result)
        agent_result["graph_rag_context"] = graph_context
        agent_result["calibrated_confidence"] = confidence_calibration["calibrated_confidence"]
        return {
            "query": query,
            "user_role": user_role,
            "pre_filter": filter_metadata,
            "retrieved_evidence": retrieved_evidence,
            "graph_rag": graph_context,
            "cross_domain_scoring": discovery_score,
            "evidence_traceability": build_evidence_traces(query, retrieved_evidence),
            "confidence_calibration": confidence_calibration,
            "agent_reasoning": agent_result,
            "post_validation": validation_result,
            "abductive_reasoning": abductive_result,
            "memory_footprint": self.memory_service.get_summary(),
            "performance_metrics": {
                "total_time_seconds": total_time_s,
                "agent_reasoning_time_seconds": agent_time_s,
                "pre_filter_ms": filter_metadata["execution_time_ms"],
                "post_validation_ms": validation_result["execution_time_ms"],
                "retrieved_chunks_count": len(retrieved_evidence),
                "graph_nodes_count": len(graph_context["nodes"]),
                "multi_hop_paths_count": len(graph_context["multi_hop_paths"]),
            },
        }

    def process_query(self, query: str, user_role: str = "researcher", confidence_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        cache_key = (query, user_role, tuple(sorted((confidence_thresholds or {}).items())))
        if cache_key in self._cache:
            logger.info(f"Query cache hit: {query} for role {user_role}")
            return dict(self._cache[cache_key]["result"])

        start_total = time.time()

        # Step 3a: Symbolic Pre-Filter
        filter_metadata = self.pre_filter.process(query)

        # Retrieve MirrorMind memory context
        memory_context = self.memory_service.get_relevant_context(query)
        filter_metadata["memory_context"] = memory_context

        # Step 2: Vector Retrieval with RBAC + BM25 hybrid search RRF
        query_vector = self.embedder.embed_text(query)
        retrieved_evidence = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=filter_metadata["detected_domains"],
            top_k=5,
            query_text=query
        )

        # Step 3b: Yuuki RxG Nano Reasoning
        start_reasoning = time.time()
        graph_seed_context = self.knowledge_graph.graph_rag_context(retrieved_evidence, filter_metadata.get("extracted_entities", []))
        agent_result = self.agent.reason_and_synthesize(query, retrieved_evidence, filter_metadata, graph_seed_context)
        agent_time_s = round(time.time() - start_reasoning, 2)

        # Step 3c: Symbolic Post-Validation
        validation_result = self.post_validator.validate(agent_result, retrieved_evidence)

        # Run Abductive Reasoning Engine in parallel/addition
        abductive_result = self.abductive_engine.perform_abductive_reasoning(
            query, retrieved_evidence, filter_metadata, self.post_validator
        )

        total_time_s = round(time.time() - start_total, 2)

        result = self._enrich_result(query, user_role, filter_metadata, retrieved_evidence, agent_result, validation_result, total_time_s, agent_time_s, confidence_thresholds, abductive_result)

        # Store interaction in long-term memory
        self.memory_service.add_interaction(query, result)

        # Build simulated/recorded events list for stream_query cache hits
        events_recorded = [
            {"event": "step_3a_pre_filter", "data": filter_metadata},
            {"event": "step_2_vector_retrieval", "data": {"retrieved_count": len(retrieved_evidence), "retrieved_evidence": retrieved_evidence}},
            {"event": "step_3b_rxg_nano_reasoning", "data": {"stage": "pre_filter", "delta": f"Symbolic Pre-Filter complete (<50ms). Identified domains: {', '.join(filter_metadata.get('detected_domains', []))}.", "filter_metadata": filter_metadata}},
            {"event": "step_3b_rxg_nano_reasoning", "data": {"stage": "thinking", "delta": agent_result["think_block"]}},
            {"event": "step_3b_rxg_nano_reasoning", "data": {"stage": "hypothesis_synthesis", "delta": agent_result["output_text"], "structured_result": agent_result}},
            {"event": "step_3c_post_validation", "data": validation_result},
            {"event": "step_4_graph_rag", "data": {"graph_rag": result["graph_rag"], "cross_domain_scoring": result["cross_domain_scoring"], "evidence_traceability": result["evidence_traceability"], "confidence_calibration": result["confidence_calibration"], "abductive_reasoning": result["abductive_reasoning"], "memory_footprint": result["memory_footprint"]}},
            {"event": "completed", "data": {"status": "success", "final_hypothesis": agent_result["output_text"], "validation_passed": validation_result["validated"], "calibrated_confidence": result["confidence_calibration"]["calibrated_confidence"]}}
        ]

        self._cache[cache_key] = {
            "result": result,
            "events": events_recorded
        }

        return result

    def stream_query(self, query: str, user_role: str = "researcher") -> Generator[Dict[str, Any], None, None]:
        """
        SSE streaming handler for end-to-end execution.
        """
        cache_key = (query, user_role)
        if cache_key in self._cache:
            logger.info(f"Stream cache hit: {query} for role {user_role}")
            for event in self._cache[cache_key]["events"]:
                yield event
            return

        events_recorded = []

        # Step 3a: Pre-filter
        filter_metadata = self.pre_filter.process(query)

        # Retrieve MirrorMind memory context
        memory_context = self.memory_service.get_relevant_context(query)
        filter_metadata["memory_context"] = memory_context

        evt1 = {
            "event": "step_3a_pre_filter",
            "data": filter_metadata
        }
        events_recorded.append(evt1)
        yield evt1

        # Step 2: Vector Search with RRF hybrid search
        query_vector = self.embedder.embed_text(query)
        retrieved_evidence = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=filter_metadata["detected_domains"],
            top_k=5,
            query_text=query
        )
        evt2 = {
            "event": "step_2_vector_retrieval",
            "data": {
                "retrieved_count": len(retrieved_evidence),
                "retrieved_evidence": retrieved_evidence
            }
        }
        events_recorded.append(evt2)
        yield evt2

        # Step 3b: Yuuki RxG Nano Agent streaming reasoning
        agent_result = None
        graph_seed_context = self.knowledge_graph.graph_rag_context(retrieved_evidence, filter_metadata.get("extracted_entities", []))
        for stream_chunk in self.agent.stream_reasoning(query, retrieved_evidence, filter_metadata, graph_seed_context):
            if "structured_result" in stream_chunk:
                agent_result = stream_chunk["structured_result"]
            evt3 = {
                "event": "step_3b_rxg_nano_reasoning",
                "data": stream_chunk
            }
            events_recorded.append(evt3)
            yield evt3

        if agent_result is None:
            agent_result = self.agent.reason_and_synthesize(query, retrieved_evidence, filter_metadata, graph_seed_context)

        # Step 3c: Post-Validation
        validation_result = self.post_validator.validate(agent_result, retrieved_evidence)
        evt4 = {
            "event": "step_3c_post_validation",
            "data": validation_result
        }
        events_recorded.append(evt4)
        yield evt4

        # Run Abductive Reasoning Engine
        abductive_result = self.abductive_engine.perform_abductive_reasoning(
            query, retrieved_evidence, filter_metadata, self.post_validator
        )

        processed_res = self._enrich_result(query, user_role, filter_metadata, retrieved_evidence, agent_result, validation_result, 0.0, 0.0, abductive_result=abductive_result)
        
        # Store interaction in long-term memory
        self.memory_service.add_interaction(query, processed_res)

        evt_graph = {
            "event": "step_4_graph_rag",
            "data": {
                "graph_rag": processed_res["graph_rag"],
                "cross_domain_scoring": processed_res["cross_domain_scoring"],
                "confidence_calibration": processed_res["confidence_calibration"],
                "abductive_reasoning": processed_res["abductive_reasoning"],
                "memory_footprint": processed_res["memory_footprint"]
            }
        }
        events_recorded.append(evt_graph)
        yield evt_graph

        evt5 = {
            "event": "completed",
            "data": {
                "status": "success",
                "final_hypothesis": agent_result["output_text"],
                "validation_passed": validation_result["validated"],
                "calibrated_confidence": processed_res["confidence_calibration"]["calibrated_confidence"]
            }
        }
        events_recorded.append(evt5)
        yield evt5

        # Cache the query result
        self._cache[cache_key] = {
            "result": processed_res,
            "events": events_recorded
        }

_neuro_symbolic_pipeline = None

def get_neuro_symbolic_pipeline() -> NeuroSymbolicPipeline:
    global _neuro_symbolic_pipeline
    if _neuro_symbolic_pipeline is None:
        _neuro_symbolic_pipeline = NeuroSymbolicPipeline()
    return _neuro_symbolic_pipeline
