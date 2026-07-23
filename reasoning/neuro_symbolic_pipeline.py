import time
import logging
from typing import Any, Dict, Generator, List, Optional
from config import settings

from reasoning.symbolic_filter import SymbolicPreFilter, SymbolicPostValidator
from reasoning.rxg_nano_agent import YuukiRxGNanoAgent
from vector_store.qdrant_engine import get_qdrant_engine
from ingestion.embedding import get_embedder
from reasoning.knowledge_graph import get_knowledge_graph, DiscoveryScorer, ConfidenceCalibrator
from reasoning.traceability import build_evidence_traces
from reasoning.memory_service import get_memory_service
from reasoning.abductive_engine import AbductiveReasoningEngine
from reasoning.multi_agent import get_multi_agent_orchestrator
from reasoning.hybrid_rag_kg import get_hybrid_rag_kg
from reasoning.dual_memory import get_dual_memory
from reasoning.z3_validator import get_z3_validator
from reasoning.experimental_blueprint import get_experimental_blueprint_generator
from reasoning.evidence_attribution import get_evidence_attributor
from reasoning.risk_feedback import get_risk_feedback_engine
from reasoning.collaboration_recommender import get_collaboration_recommender

logger = logging.getLogger("crossmind.neuro_symbolic")


class NeuroSymbolicPipeline:
    """
    Complete Phase 3 Orchestrator with 9 advanced capabilities:
    1. Multi-Agent Orchestration
    2. Dual-Memory Architecture
    3. Formal Z3 Validation
    4. Experimental Blueprint
    5. Hybrid RAG-KG
    6. Evidence Attribution
    7. Risk-Controlled Feedback
    8. Collaboration Recommendations
    9. Graph Browser support
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

        # Advanced engines
        self.multi_agent = get_multi_agent_orchestrator()
        self.hybrid_rag = get_hybrid_rag_kg()
        self.dual_memory = get_dual_memory()
        self.z3_validator = get_z3_validator()
        self.blueprint_generator = get_experimental_blueprint_generator()
        self.attributor = get_evidence_attributor()
        self.risk_feedback = get_risk_feedback_engine()
        self.collab_recommender = get_collaboration_recommender()

    def _enrich_result(
        self,
        query: str,
        user_role: str,
        filter_metadata: Dict[str, Any],
        retrieved_evidence: list,
        agent_result: Dict[str, Any],
        validation_result: Dict[str, Any],
        total_time_s: float,
        agent_time_s: float,
        confidence_thresholds: Dict[str, float] = None,
        abductive_result: Dict[str, Any] = None,
        multi_agent_report: Dict[str, Any] = None,
        z3_validation: Dict[str, Any] = None,
        evidence_attribution: Dict[str, Any] = None,
        experimental_blueprint: Dict[str, Any] = None,
        collaboration_recommendations: Dict[str, Any] = None,
        risk_summary: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        graph_context = self.knowledge_graph.graph_rag_context(
            retrieved_evidence, filter_metadata.get("extracted_entities", [])
        )
        discovery_score = DiscoveryScorer.score(retrieved_evidence, graph_context)
        confidence_calibration = ConfidenceCalibrator.calibrate(
            agent_result.get("confidence_score", 0.0),
            discovery_score,
            validation_result,
            confidence_thresholds,
        )
        agent_result = dict(agent_result)
        agent_result["graph_rag_context"] = graph_context
        agent_result["calibrated_confidence"] = confidence_calibration[
            "calibrated_confidence"
        ]

        # Evidence attribution
        if evidence_attribution is None:
            evidence_attribution = self.attributor.attribute(
                agent_result.get("hypothesis", agent_result.get("output_text", "")),
                retrieved_evidence,
            )

        # Z3 validation
        if z3_validation is None and settings.Z3_VALIDATION_ENABLED:
            z3_validation = self.z3_validator.validate_hypothesis(
                agent_result, retrieved_evidence
            )
        elif z3_validation is None:
            z3_validation = {
                "validated": validation_result.get("validated", False),
                "validation_score": validation_result.get("validation_score", 0.0),
                "rule_checks": validation_result.get("rule_checks", []),
                "execution_mode": "disabled",
            }

        # Experimental blueprint
        if experimental_blueprint is None and settings.EXPERIMENTAL_BLUEPRINT_ENABLED:
            experimental_blueprint = self.blueprint_generator.generate(
                agent_result, abductive_result, retrieved_evidence
            )
        elif experimental_blueprint is None:
            experimental_blueprint = {"status": "disabled"}

        # Collaboration recommendations
        if (
            collaboration_recommendations is None
            and retrieved_evidence
        ):
            collaboration_recommendations = self.collab_recommender.recommend(
                graph_context.get("multi_hop_paths", []),
                filter_metadata.get("detected_domains", []),
            )
        elif collaboration_recommendations is None:
            collaboration_recommendations = {"status": "disabled"}

        return {
            "query": query,
            "user_role": user_role,
            "session_id": filter_metadata.get("session_id"),
            "pre_filter": filter_metadata,
            "retrieved_evidence": retrieved_evidence,
            "graph_rag": graph_context,
            "cross_domain_scoring": discovery_score,
            "evidence_traceability": build_evidence_traces(query, retrieved_evidence),
            "evidence_attribution": evidence_attribution,
            "confidence_calibration": confidence_calibration,
            "agent_reasoning": agent_result,
            "post_validation": validation_result,
            "z3_formal_validation": z3_validation,
            "abductive_reasoning": abductive_result,
            "experimental_blueprint": experimental_blueprint,
            "collaboration_recommendations": collaboration_recommendations,
            "multi_agent_orchestration": multi_agent_report,
            "risk_controlled_feedback": risk_summary,
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

    def process_query(
        self,
        query: str,
        user_role: str = "researcher",
        confidence_thresholds: Dict[str, float] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        cache_key = (
            query,
            user_role,
            session_id,
            tuple(sorted((confidence_thresholds or {}).items())),
        )
        if cache_key in self._cache:
            logger.info(f"Query cache hit: {query} for role {user_role}")
            return dict(self._cache[cache_key]["result"])

        start_total = time.time()

        # Dual memory context
        memory_context = {}
        if settings.DUAL_MEMORY_ENABLED:
            memory_context = self.dual_memory.get_session_context(session_id)

        # Step 3a: Symbolic Pre-Filter
        filter_metadata = self.pre_filter.process(query)
        filter_metadata["session_id"] = session_id
        if memory_context:
            filter_metadata["dual_memory_context"] = memory_context
        memory_context_str = self.memory_service.get_relevant_context(query)
        filter_metadata["memory_context"] = memory_context_str

        # Step 2: Retrieval (Hybrid RAG-KG or standard)
        if settings.MULTI_AGENT_ENABLED:
            hybrid_result = self.hybrid_rag.retrieve(
                query=query,
                user_role=user_role,
                allowed_domains=filter_metadata["detected_domains"],
                top_k=5,
            )
            retrieved_evidence = hybrid_result.get("fused_results", [])
            filter_metadata["retrieval_strategy"] = hybrid_result.get("strategy", "hybrid_rag_kg")
        else:
            query_vector = self.embedder.embed_text(query)
            retrieved_evidence = self.vector_engine.search_with_rbac(
                query_vector=query_vector,
                user_role=user_role,
                allowed_domains=filter_metadata["detected_domains"],
                top_k=5,
                query_text=query,
            )
            filter_metadata["retrieval_strategy"] = "standard_vector"

        # Multi-agent orchestration (parallel domain processing)
        multi_agent_report = None
        if settings.MULTI_AGENT_ENABLED and retrieved_evidence:
            multi_agent_report = self.multi_agent.parallel_process(
                query, retrieved_evidence, filter_metadata
            )

        # Step 3b: Yuuki RxG Nano Reasoning
        start_reasoning = time.time()
        graph_seed_context = self.knowledge_graph.graph_rag_context(
            retrieved_evidence, filter_metadata.get("extracted_entities", [])
        )
        agent_result = self.agent.reason_and_synthesize(
            query, retrieved_evidence, filter_metadata, graph_seed_context
        )
        agent_time_s = round(time.time() - start_reasoning, 2)

        # Step 3c: Symbolic Post-Validation
        validation_result = self.post_validator.validate(
            agent_result, retrieved_evidence
        )

        # Z3 formal validation
        z3_validation = None
        if settings.Z3_VALIDATION_ENABLED:
            z3_validation = self.z3_validator.validate_hypothesis(
                agent_result, retrieved_evidence
            )

        # Abductive Reasoning
        abductive_result = self.abductive_engine.perform_abductive_reasoning(
            query, retrieved_evidence, filter_metadata, self.post_validator
        )

        total_time_s = round(time.time() - start_total, 2)

        # Build risk summary from active learning feedback
        risk_summary = self.risk_feedback.get_risk_summary()

        result = self._enrich_result(
            query,
            user_role,
            filter_metadata,
            retrieved_evidence,
            agent_result,
            validation_result,
            total_time_s,
            agent_time_s,
            confidence_thresholds,
            abductive_result,
            multi_agent_report,
            z3_validation,
            None,
            None,
            None,
            risk_summary,
        )

        # Record feedback for risk-controlled learning
        if settings.RISK_FEEDBACK_ENABLED:
            for ev in retrieved_evidence[:3]:
                self.risk_feedback.submit_feedback(
                    query,
                    ev.get("id"),
                    float(ev.get("score", 0.0)),
                    user_role,
                    [ev.get("payload", {}).get("domain", "general")],
                )

        # Dual memory recording
        if settings.DUAL_MEMORY_ENABLED:
            self.dual_memory.record_interaction(
                session_id, query, result, user_role
            )
        else:
            self.memory_service.add_interaction(query, result)

        events_recorded = [
            {"event": "step_3a_pre_filter", "data": filter_metadata},
            {
                "event": "step_2_vector_retrieval",
                "data": {
                    "retrieved_count": len(retrieved_evidence),
                    "retrieved_evidence": retrieved_evidence,
                    "multi_agent_report": multi_agent_report,
                },
            },
            {
                "event": "step_3b_rxg_nano_reasoning",
                "data": {
                    "stage": "pre_filter",
                    "delta": f"Symbolic Pre-Filter complete (<50ms). Identified domains: {', '.join(filter_metadata.get('detected_domains', []))}.",
                    "filter_metadata": filter_metadata,
                },
            },
            {
                "event": "step_3b_rxg_nano_reasoning",
                "data": {"stage": "thinking", "delta": agent_result["think_block"]},
            },
            {
                "event": "step_3b_rxg_nano_reasoning",
                "data": {
                    "stage": "hypothesis_synthesis",
                    "delta": agent_result["output_text"],
                    "structured_result": agent_result,
                },
            },
            {"event": "step_3c_post_validation", "data": validation_result},
            {
                "event": "step_3c_z3_validation",
                "data": z3_validation or {},
            },
            {
                "event": "step_4_graph_rag",
                "data": {
                    "graph_rag": result["graph_rag"],
                    "cross_domain_scoring": result["cross_domain_scoring"],
                    "evidence_traceability": result["evidence_traceability"],
                    "evidence_attribution": result["evidence_attribution"],
                    "confidence_calibration": result["confidence_calibration"],
                    "abductive_reasoning": result["abductive_reasoning"],
                    "experimental_blueprint": result["experimental_blueprint"],
                    "collaboration_recommendations": result[
                        "collaboration_recommendations"
                    ],
                    "memory_footprint": result["memory_footprint"],
                },
            },
            {
                "event": "completed",
                "data": {
                    "status": "success",
                    "final_hypothesis": agent_result["output_text"],
                    "validation_passed": validation_result["validated"],
                    "z3_validated": (
                        z3_validation.get("validated", False)
                        if z3_validation
                        else False
                    ),
                    "experimental_blueprint_generated": (
                        result.get("experimental_blueprint", {}).get("status") != "disabled"
                        if result.get("experimental_blueprint")
                        else False
                    ),
                    "calibrated_confidence": result["confidence_calibration"][
                        "calibrated_confidence"
                    ],
                },
            },
        ]

        self._cache[cache_key] = {"result": result, "events": events_recorded}
        return result

    def stream_query(
        self, query: str, user_role: str = "researcher", session_id: str = "default"
    ) -> Generator[Dict[str, Any], None, None]:
        cache_key = (query, user_role, session_id)
        if cache_key in self._cache:
            logger.info(f"Stream cache hit: {query} for role {user_role}")
            for event in self._cache[cache_key]["events"]:
                yield event
            return

        events_recorded = []

        filter_metadata = self.pre_filter.process(query)
        filter_metadata["session_id"] = session_id
        memory_context_str = self.memory_service.get_relevant_context(query)
        filter_metadata["memory_context"] = memory_context_str

        evt1 = {"event": "step_3a_pre_filter", "data": filter_metadata}
        events_recorded.append(evt1)
        yield evt1

        query_vector = self.embedder.embed_text(query)
        retrieved_evidence = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=filter_metadata["detected_domains"],
            top_k=5,
            query_text=query,
        )
        evt2 = {
            "event": "step_2_vector_retrieval",
            "data": {
                "retrieved_count": len(retrieved_evidence),
                "retrieved_evidence": retrieved_evidence,
            },
        }
        events_recorded.append(evt2)
        yield evt2

        agent_result = None
        graph_seed_context = self.knowledge_graph.graph_rag_context(
            retrieved_evidence, filter_metadata.get("extracted_entities", [])
        )
        for stream_chunk in self.agent.stream_reasoning(
            query, retrieved_evidence, filter_metadata, graph_seed_context
        ):
            if "structured_result" in stream_chunk:
                agent_result = stream_chunk["structured_result"]
            evt3 = {
                "event": "step_3b_rxg_nano_reasoning",
                "data": stream_chunk,
            }
            events_recorded.append(evt3)
            yield evt3

        if agent_result is None:
            agent_result = self.agent.reason_and_synthesize(
                query, retrieved_evidence, filter_metadata, graph_seed_context
            )

        validation_result = self.post_validator.validate(
            agent_result, retrieved_evidence
        )
        evt4 = {"event": "step_3c_post_validation", "data": validation_result}
        events_recorded.append(evt4)
        yield evt4

        abductive_result = self.abductive_engine.perform_abductive_reasoning(
            query, retrieved_evidence, filter_metadata, self.post_validator
        )

        processed_res = self._enrich_result(
            query,
            user_role,
            filter_metadata,
            retrieved_evidence,
            agent_result,
            validation_result,
            0.0,
            0.0,
            abductive_result=abductive_result,
        )

        if settings.DUAL_MEMORY_ENABLED:
            self.dual_memory.record_interaction(
                session_id, query, processed_res, user_role
            )
        else:
            self.memory_service.add_interaction(query, processed_res)

        evt_graph = {
            "event": "step_4_graph_rag",
            "data": {
                "graph_rag": processed_res["graph_rag"],
                "cross_domain_scoring": processed_res["cross_domain_scoring"],
                "confidence_calibration": processed_res["confidence_calibration"],
                "abductive_reasoning": processed_res["abductive_reasoning"],
                "experimental_blueprint": processed_res["experimental_blueprint"],
                "collaboration_recommendations": processed_res[
                    "collaboration_recommendations"
                ],
                "memory_footprint": processed_res["memory_footprint"],
            },
        }
        events_recorded.append(evt_graph)
        yield evt_graph

        evt5 = {
            "event": "completed",
            "data": {
                "status": "success",
                "final_hypothesis": agent_result["output_text"],
                "validation_passed": validation_result["validated"],
                "calibrated_confidence": processed_res["confidence_calibration"][
                    "calibrated_confidence"
                ],
            },
        }
        events_recorded.append(evt5)
        yield evt5

        self._cache[cache_key] = {
            "result": processed_res,
            "events": events_recorded,
        }


_neuro_symbolic_pipeline = None


def get_neuro_symbolic_pipeline() -> NeuroSymbolicPipeline:
    global _neuro_symbolic_pipeline
    if _neuro_symbolic_pipeline is None:
        _neuro_symbolic_pipeline = NeuroSymbolicPipeline()
    return _neuro_symbolic_pipeline
