import time
import logging
from typing import Dict, Any, Generator
from reasoning.symbolic_filter import SymbolicPreFilter, SymbolicPostValidator
from reasoning.rxg_nano_agent import YuukiRxGNanoAgent
from vector_store.qdrant_engine import get_qdrant_engine
from ingestion.embedding import get_embedder

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

    def process_query(self, query: str, user_role: str = "researcher") -> Dict[str, Any]:
        start_total = time.time()

        # Step 3a: Symbolic Pre-Filter
        filter_metadata = self.pre_filter.process(query)

        # Step 2: Vector Retrieval with RBAC
        query_vector = self.embedder.embed_text(query)
        retrieved_evidence = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=filter_metadata["detected_domains"],
            top_k=5
        )

        # Step 3b: Yuuki RxG Nano Reasoning
        start_reasoning = time.time()
        agent_result = self.agent.reason_and_synthesize(query, retrieved_evidence, filter_metadata)
        agent_time_s = round(time.time() - start_reasoning, 2)

        # Step 3c: Symbolic Post-Validation
        validation_result = self.post_validator.validate(agent_result, retrieved_evidence)

        total_time_s = round(time.time() - start_total, 2)

        return {
            "query": query,
            "user_role": user_role,
            "pre_filter": filter_metadata,
            "retrieved_evidence": retrieved_evidence,
            "agent_reasoning": agent_result,
            "post_validation": validation_result,
            "performance_metrics": {
                "total_time_seconds": total_time_s,
                "agent_reasoning_time_seconds": agent_time_s,
                "pre_filter_ms": filter_metadata["execution_time_ms"],
                "post_validation_ms": validation_result["execution_time_ms"],
                "retrieved_chunks_count": len(retrieved_evidence)
            }
        }

    def stream_query(self, query: str, user_role: str = "researcher") -> Generator[Dict[str, Any], None, None]:
        """
        SSE streaming handler for end-to-end execution.
        """
        # Step 3a: Pre-filter
        filter_metadata = self.pre_filter.process(query)
        yield {
            "event": "step_3a_pre_filter",
            "data": filter_metadata
        }

        # Step 2: Vector Search
        query_vector = self.embedder.embed_text(query)
        retrieved_evidence = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=filter_metadata["detected_domains"],
            top_k=5
        )
        yield {
            "event": "step_2_vector_retrieval",
            "data": {
                "retrieved_count": len(retrieved_evidence),
                "retrieved_evidence": retrieved_evidence
            }
        }

        # Step 3b: Yuuki RxG Nano Agent streaming reasoning
        agent_result = None
        for stream_chunk in self.agent.stream_reasoning(query, retrieved_evidence, filter_metadata):
            if "structured_result" in stream_chunk:
                agent_result = stream_chunk["structured_result"]
            yield {
                "event": "step_3b_rxg_nano_reasoning",
                "data": stream_chunk
            }

        if agent_result is None:
            agent_result = self.agent.reason_and_synthesize(query, retrieved_evidence, filter_metadata)

        # Step 3c: Post-Validation
        validation_result = self.post_validator.validate(agent_result, retrieved_evidence)
        yield {
            "event": "step_3c_post_validation",
            "data": validation_result
        }

        yield {
            "event": "completed",
            "data": {
                "status": "success",
                "final_hypothesis": agent_result["output_text"],
                "validation_passed": validation_result["validated"]
            }
        }

_neuro_symbolic_pipeline = None

def get_neuro_symbolic_pipeline() -> NeuroSymbolicPipeline:
    global _neuro_symbolic_pipeline
    if _neuro_symbolic_pipeline is None:
        _neuro_symbolic_pipeline = NeuroSymbolicPipeline()
    return _neuro_symbolic_pipeline
