import numpy as np
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger("crossmind.evaluation")

class RetrievalEvaluator:
    """
    RetrievalEvaluator calculates Precision@K, Recall@K, MRR, and NDCG@K
    to assess vector and hybrid search retrieval quality.
    """
    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        if k <= 0 or not retrieved:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in ground_truth)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        if not ground_truth or not retrieved or k <= 0:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in ground_truth)
        return relevant_retrieved / len(ground_truth)

    @staticmethod
    def mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
        if not ground_truth or not retrieved:
            return 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in ground_truth:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        if k <= 0 or not retrieved or not ground_truth:
            return 0.0
        retrieved_k = retrieved[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_k, 1):
            if doc_id in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
                
        # Calculate IDCG@K (Ideal DCG)
        idcg = 0.0
        ideal_hits = min(len(ground_truth), k)
        for rank in range(1, ideal_hits + 1):
            idcg += 1.0 / np.log2(rank + 1)
            
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate(cls, retrieved: List[str], ground_truth: List[str], k: int = 5) -> Dict[str, float]:
        gt_set = set(ground_truth)
        return {
            f"precision_at_{k}": round(cls.precision_at_k(retrieved, gt_set, k), 4),
            f"recall_at_{k}": round(cls.recall_at_k(retrieved, gt_set, k), 4),
            "mrr": round(cls.mrr(retrieved, gt_set), 4),
            f"ndcg_at_{k}": round(cls.ndcg_at_k(retrieved, gt_set, k), 4),
        }

# Predefined Ground Truth queries and expected doc IDs for system self-evaluation
BENCHMARK_GROUND_TRUTH = {
    "Alzheimer's disease": ["doc_neuro_01", "doc_neuro_02"],
    "nanoparticles": ["doc_nano_01", "doc_nano_02"],
    "drug bioavailability": ["doc_pharm_01", "doc_pharm_02"],
    "lithium-ion battery": ["doc_energy_01", "doc_energy_02"],
    "portfolio risk optimization": ["doc_finance_01", "doc_finance_02"],
}

def evaluate_system_retrieval(pipeline) -> Dict[str, Any]:
    """
    Evaluates current system retrieval using the benchmark set.
    """
    # Seed mock benchmark data into the vector index if empty
    from vector_store.qdrant_engine import get_qdrant_engine
    engine = get_qdrant_engine()
    
    # Check if empty, and upsert mock documents if so
    if not getattr(engine, "_memory_store", []):
        logger.info("Seeding mock document data for evaluation benchmark...")
        mock_records = [
            {"id": "doc_neuro_01", "vector": [0.1] * 1024, "payload": {"title": "Alzheimer's biology", "content": "Alzheimer's disease amyloid plaque and neurodegeneration", "domain": "neuroscience"}},
            {"id": "doc_neuro_02", "vector": [0.12] * 1024, "payload": {"title": "Alzheimer's synaptic loss", "content": "Synaptic loss in Alzheimer's patients cortical neurons", "domain": "neuroscience"}},
            {"id": "doc_nano_01", "vector": [0.2] * 1024, "payload": {"title": "Lipid nanoparticles", "content": "Lipid nanoparticles LNP design and development", "domain": "nanotechnology"}},
            {"id": "doc_nano_02", "vector": [0.22] * 1024, "payload": {"title": "Gold nanoparticles PEG", "content": "PEG functionalized gold nanoparticles for cellular delivery", "domain": "nanotechnology"}},
            {"id": "doc_pharm_01", "vector": [0.3] * 1024, "payload": {"title": "Bioavailability study", "content": "Bioavailability pharmacokinetics oral dosing", "domain": "pharmacology"}},
            {"id": "doc_pharm_02", "vector": [0.32] * 1024, "payload": {"title": "Drug absorption toxicity", "content": "Drug absorption rate and toxicity metrics", "domain": "pharmacology"}},
            {"id": "doc_energy_01", "vector": [0.4] * 1024, "payload": {"title": "Lithium battery design", "content": "Lithium-ion battery anode cathode electrolyte chemistry", "domain": "energy"}},
            {"id": "doc_energy_02", "vector": [0.42] * 1024, "payload": {"title": "Solid-state battery", "content": "Solid-state electrolyte battery technology", "domain": "energy"}},
            {"id": "doc_finance_01", "vector": [0.5] * 1024, "payload": {"title": "Portfolio optimization", "content": "Portfolio risk optimization models Black-Scholes", "domain": "financial"}},
            {"id": "doc_finance_02", "vector": [0.52] * 1024, "payload": {"title": "Risk premium", "content": "Volatility modeling stock finance asset return", "domain": "financial"}},
        ]
        engine.upsert_vectors(mock_records)

    eval_results = []
    for query, gt_ids in BENCHMARK_GROUND_TRUTH.items():
        # Retrieve using pipeline
        res = pipeline.process_query(query, user_role="admin")
        retrieved_evidence = res.get("retrieved_evidence", [])
        retrieved_ids = [str(item.get("id")) for item in retrieved_evidence]
        
        # Calculate metrics
        metrics = RetrievalEvaluator.evaluate(retrieved_ids, gt_ids, k=5)
        eval_results.append({
            "query": query,
            "retrieved": retrieved_ids,
            "ground_truth": gt_ids,
            "metrics": metrics
        })

    # Average metrics
    avg_metrics = {}
    metric_keys = ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"]
    for key in metric_keys:
        avg_metrics[key] = round(float(np.mean([item["metrics"][key] for item in eval_results])), 4)
        
    return {
        "status": "success",
        "benchmark_runs": eval_results,
        "average_metrics": avg_metrics
    }
