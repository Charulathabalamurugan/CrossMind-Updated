import uuid
import logging
import time
from typing import List, Dict, Any, Optional
from config import settings
from ingestion.embedding import get_embedder
from vector_store.qdrant_engine import get_qdrant_engine
from reasoning.knowledge_graph import get_knowledge_graph
from ingestion.active_learning import get_active_learning_engine
from ingestion.ingestion_cache import get_ingestion_cache
from ingestion.text_extractor import get_text_extractor
from ingestion.chunker import get_chunker
from ingestion.sparse_vector import get_sparse_vector_engine
from reasoning.query_cache import get_query_cache
from reasoning.sparse_retriever import get_sparse_retriever
from reasoning.query_preprocessor import get_query_preprocessor
from reasoning.hypothesis_generator import get_hypothesis_generator
from reasoning.rule_engine import get_rule_engine
from reasoning.bridge_scorer import get_bridge_scorer
from reasoning.result_formatter import format_pipeline_result
from reasoning.benchmark_collector import get_benchmark_collector
from reasoning.feedback_collector import get_feedback_collector
from reasoning.retrainer import get_model_retrainer
from reasoning.rule_updater import get_rule_updater

logger = logging.getLogger("crossmind.ingestion")

class IngestionPipeline:
    """
    Unified ingestion pipeline integrating all 6 phases:
    Phase 1: Multimodal document ingestion with sparse vector support
    Phase 2: Hybrid retrieval with query caching and preprocessing
    Phase 3: Hypothesis generation and symbolic rule engine validation
    Phase 4: Enhanced bridge strength scoring
    Phase 5: Structured result formatting and benchmark tracking
    Phase 6: Risk-controlled feedback, model retraining, dynamic rule updates
    """
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_engine = get_qdrant_engine()
        self.knowledge_graph = get_knowledge_graph()
        self.cache = get_ingestion_cache()
        self.active_learning = get_active_learning_engine()
        self._initialized = False

        # Phase 1 components
        self.text_extractor = get_text_extractor()
        self.chunker = get_chunker()
        self.sparse_engine = get_sparse_vector_engine()

        # Phase 2 components
        self.query_cache = get_query_cache()
        self.sparse_retriever = get_sparse_retriever()
        self.query_preprocessor = get_query_preprocessor()

        # Phase 3 components
        self.hypothesis_generator = get_hypothesis_generator()
        self.rule_engine = get_rule_engine()

        # Phase 4 components
        self.bridge_scorer = get_bridge_scorer()

        # Phase 5 components
        self.benchmark_collector = get_benchmark_collector()

        # Phase 6 components
        self.feedback_collector = get_feedback_collector()
        self.model_retrainer = get_model_retrainer()
        self.rule_updater = get_rule_updater()

    def auto_init(self):
        if self._initialized:
            return
        self._initialized = True
        if not settings.AUTO_INIT_ON_STARTUP:
            logger.info("Auto-init disabled by settings.")
            return
        from ingestion.dynamic_connectors import get_dynamic_connectors
        manager = get_dynamic_connectors()
        manager.load_from_env(callback=self.ingest_documents)
        manager.start_all(callback=self.ingest_documents)
        self._start_continuous_ingestion()
        logger.info("Auto-init complete: connectors and continuous ingestion started.")

    def _start_continuous_ingestion(self):
        try:
            from ingestion.continuous_ingestion import ContinuousIngestionWorker
            worker = ContinuousIngestionWorker(pipeline=self)
            worker.start()
            self._continuous_worker = worker
        except Exception as exc:
            logger.warning(f"Failed to start continuous ingestion worker: {exc}")

    def _extract_document_content(self, doc: Dict[str, Any]) -> str:
        file_path = doc.get("file_path", "")
        if file_path:
            ext = file_path.lower()
            if ext.endswith(".pdf") or ext.endswith(".docx"):
                extracted = self.text_extractor.extract(file_path)
                if extracted:
                    logger.info(f"Extracted text from {file_path}")
                    return extracted
        content = doc.get("content", "")
        if isinstance(content, str) and (content.endswith(".pdf") or content.endswith(".docx")):
            extracted = self.text_extractor.extract(content)
            if extracted:
                logger.info(f"Extracted text from content path {content}")
                return extracted
        return content

    def _chunk_and_embed(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = self._extract_document_content(doc)
        metadata = {
            "title": doc.get("title", "Untitled Document"),
            "domain": doc.get("domain", "general"),
            "year": doc.get("year", 2024),
            "authors": doc.get("authors", []),
            "allowed_roles": doc.get("allowed_roles", ["public", "researcher"]),
            "tags": doc.get("tags", []),
        }
        chunks = self.chunker.chunk_text(content, metadata=metadata)
        if not chunks:
            return []
        for chunk in chunks:
            chunk["id"] = doc.get("id") or str(uuid.uuid4()) + f"_chunk_{chunk['chunk_index']}"
            chunk["doc_id"] = doc.get("id") or str(uuid.uuid4())
            chunk["content_hash"] = doc.get("content_hash") or str(hash(content))
        texts = [c["text"] for c in chunks]
        emb_dim = settings.BGE_M3_RETRIEVAL_DIM if settings.BGE_M3_MATRYOSHKA_ENABLED else settings.EMBEDDING_DIM
        embeddings = self.embedder.embed_texts(texts, dim=emb_dim)
        for chunk, emb in zip(chunks, embeddings):
            chunk["vector"] = emb
        return chunks

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        if not documents:
            return []

        deduped: List[Dict[str, Any]] = []
        for doc in documents:
            cache_key = doc.get("content_hash") or str(hash(doc.get("content", "")))
            if self.cache.get(cache_key):
                continue
            if not doc.get("content_hash"):
                doc["content_hash"] = cache_key
            deduped.append(doc)
            self.cache.set(cache_key, True)

        self.benchmark_collector.start_phase("ingestion")
        all_inserted_ids = []
        all_sparse_records = []

        for doc in deduped:
            chunks = self._chunk_and_embed(doc)
            if not chunks:
                continue
            records_to_upsert = []
            for chunk in chunks:
                doc_id = chunk["id"]
                content_text = chunk.get("text", "")
                
                # Quality scoring logic
                q_score = 0.1
                if content_text:
                    words = len(content_text.split())
                    if words > 500:
                        q_score += 0.4
                    elif words > 200:
                        q_score += 0.2
                    elif words > 50:
                        q_score += 0.1
                    
                    structure_words = ["abstract", "conclusions", "references", "methods", "results", "discussion", "introduction", "figure", "table"]
                    content_lower = content_text.lower()
                    matches = sum(1 for w in structure_words if w in content_lower)
                    q_score += min(matches * 0.05, 0.3)
                    
                    import re
                    if re.search(r'\b\d+(\.\d+)?%\b', content_lower) or re.search(r'\bp\s*<\s*0\.\d+\b', content_lower):
                        q_score += 0.1
                    if doc.get("authors") and doc.get("year") and doc.get("title"):
                        q_score += 0.1
                q_score = round(min(q_score, 1.0), 3)

                payload = {
                    "id": doc_id,
                    "title": chunk.get("title", doc.get("title", "Untitled Document")),
                    "content": content_text,
                    "domain": chunk.get("domain", doc.get("domain", "general")),
                    "year": chunk.get("year", doc.get("year", 2024)),
                    "authors": chunk.get("authors", doc.get("authors", [])),
                    "allowed_roles": chunk.get("allowed_roles", doc.get("allowed_roles", ["public", "researcher"])),
                    "tags": chunk.get("tags", doc.get("tags", [])),
                    "citation": f"{doc.get('authors', ['CrossMind Research'])[0]} et al. ({doc.get('year', 2024)}) - {doc.get('title', 'Untitled')}",
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content_hash": chunk.get("content_hash", ""),
                    "quality_score": q_score,
                }
                records_to_upsert.append({
                    "id": doc_id,
                    "vector": chunk.get("vector", []),
                    "payload": payload,
                })
                all_sparse_records.append({
                    "id": doc_id,
                    "text": chunk.get("text", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                })

            inserted_ids = self.vector_engine.upsert_vectors(records_to_upsert)
            all_inserted_ids.extend(inserted_ids)
            for record in records_to_upsert:
                cache_key = record["payload"].get("content_hash") or str(hash(record["payload"].get("content", "")))
                self.cache.set(cache_key, record["payload"])

        self.knowledge_graph.index_documents([
            {"id": r["payload"]["id"], "title": r["payload"]["title"], "content": r["payload"]["content"], "domain": r["payload"]["domain"]}
            for r in records_to_upsert if r["payload"]
        ])
        self.sparse_engine.index_documents(all_sparse_records)

        self.benchmark_collector.end_phase("ingestion")
        self.benchmark_collector.record_metric("documents_ingested", len(deduped), {"chunks": len(all_inserted_ids)})
        logger.info(f"Successfully ingested {len(all_inserted_ids)} document chunks into Qdrant vector store with sparse indexing.")
        return all_inserted_ids

    def search(self, query: str, user_role: str = "researcher", top_k: int = 5) -> Dict[str, Any]:
        self.benchmark_collector.start_phase("query")
        processed = self.query_preprocessor.preprocess(query)
        cache_key = f"{query}:{user_role}:{top_k}"
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            self.benchmark_collector.end_phase("query")
            logger.info(f"Query cache hit for: {query[:50]}...")
            return cached

        semantic_cached = self.query_cache.get_similar(query, user_role=user_role)
        if semantic_cached is not None:
            self.benchmark_collector.end_phase("query")
            logger.info(f"Semantic query cache hit for: {query[:50]}...")
            return semantic_cached

        query_vector = self.embedder.embed_text(query, dim=settings.BGE_M3_RETRIEVAL_DIM if settings.BGE_M3_MATRYOSHKA_ENABLED else settings.EMBEDDING_DIM)
        dense_results = self.vector_engine.search_with_rbac(
            query_vector=query_vector,
            user_role=user_role,
            allowed_domains=[],
            top_k=top_k,
            query_text=query,
        )
        hybrid_results = self.sparse_retriever.hybrid_search(query, dense_results, top_k=top_k)
        result = {
            "query": query,
            "tokenized_query": processed,
            "results": hybrid_results,
            "result_count": len(hybrid_results),
            "retrieval_phase": "phase2_hybrid_retrieval",
        }
        self.query_cache.set(cache_key, result, query=query, user_role=user_role)
        self.benchmark_collector.end_phase("query")
        self.benchmark_collector.record_retrieval(
            dense_ms=0.0, sparse_ms=0.0, dense_count=len(dense_results), sparse_count=len(hybrid_results)
        )
        return result

    def generate_and_validate_hypotheses(self, query: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.benchmark_collector.start_phase("hypothesis_generation")
        filter_metadata = {
            "extracted_entities": [],
            "detected_domains": list(set(e.get("domain", "general") for e in evidence)),
            "session_id": "default",
        }
        hypothesis_result = self.hypothesis_generator.generate(filter_metadata, evidence)
        self.benchmark_collector.end_phase("hypothesis_generation")
        self.benchmark_collector.start_phase("rule_validation")
        validation_result = self.rule_engine.evaluate(
            hypothesis_result.get("hypothesis", ""),
            evidence,
            filter_metadata,
        )
        self.benchmark_collector.end_phase("rule_validation")
        self.benchmark_collector.record_validation(
            validation_result.get("validation_score", 0.0),
            len(self.rule_engine.rules),
        )
        return {
            "hypothesis": hypothesis_result,
            "validation": validation_result,
        }

    def score_bridges(self, paths: List[Dict[str, Any]], evidence_count: int, domain_count: int) -> Dict[str, Any]:
        self.benchmark_collector.start_phase("bridge_scoring")
        result = self.bridge_scorer.compute_bridge_strength(paths, evidence_count, domain_count)
        self.benchmark_collector.end_phase("bridge_scoring")
        return result

    def format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return format_pipeline_result(result)

    def submit_feedback(self, query: str, doc_id: str, score: float, user_role: str, risk_level: str = "low") -> Dict[str, Any]:
        self.benchmark_collector.start_phase("feedback")
        entry = self.feedback_collector.submit(
            query=query,
            doc_id=doc_id,
            relevance_score=score,
            user_role=user_role,
            risk_level=risk_level,
        )
        if self.model_retrainer.needs_retraining():
            logger.info("Model retraining triggered by feedback collector.")
            self.model_retrainer.perform_retrain(
                {"feedback_count": self.feedback_collector.count(), "latest_score": score}
            )
        self.benchmark_collector.end_phase("feedback")
        return {"status": "recorded", "entry": entry.to_dict()}

    def record_feedback(self, query: str, doc_id: str, score: float, user_role: str):
        self.active_learning.record_feedback(query, doc_id, score, user_role)
        self.active_learning.retrain(self)
        risk_level = "high" if score < 0.3 else "medium" if score < 0.6 else "low"
        self.submit_feedback(query, doc_id, score, user_role, risk_level=risk_level)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "benchmark_summary": self.benchmark_collector.get_summary(),
            "query_cache_size": self.query_cache.size(),
            "feedback_stats": self.feedback_collector.get_stats(),
            "model_retrainer_status": self.model_retrainer.get_status(),
            "rule_updater_stats": self.rule_updater.get_stats(),
            "rule_engine_log_size": len(self.rule_engine.get_log()),
            "sparse_engine_indexed_docs": self.sparse_engine.total_documents,
        }

_pipeline_instance = None

def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = IngestionPipeline()
    return _pipeline_instance