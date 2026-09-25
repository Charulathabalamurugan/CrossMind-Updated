import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from config import settings
from reasoning.evaluation import RetrievalEvaluator, BENCHMARK_GROUND_TRUTH

logger = logging.getLogger("crossmind.evaluation_enhancements")


@dataclass
class EvaluationRun:
    run_id: str
    timestamp: float
    query: str
    retrieved_ids: List[str]
    ground_truth: List[str]
    metrics: Dict[str, float]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationQualityMetrics:
    faithfulness: float
    relevance: float
    completeness: float
    conciseness: float
    hallucination_score: float
    overall_score: float


class EvaluationEngine:
    def __init__(self):
        self._runs: List[EvaluationRun] = []
        self._lock = threading.Lock()
        self._enabled = settings.EVALUATION_ENABLED
        self._max_runs = 10000
        self._auto_benchmark = True
        self._benchmark_interval = 3600
        self._last_benchmark = 0.0

    def record_run(
        self,
        query: str,
        retrieved_ids: List[str],
        ground_truth: List[str],
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationRun:
        if not self._enabled:
            return EvaluationRun(
                run_id="disabled",
                timestamp=time.time(),
                query=query,
                retrieved_ids=retrieved_ids,
                ground_truth=ground_truth,
                metrics={},
                latency_ms=latency_ms,
            )

        metrics = RetrievalEvaluator.evaluate(retrieved_ids, ground_truth, k=5)

        run = EvaluationRun(
            run_id=f"eval_{int(time.time() * 1000)}",
            timestamp=time.time(),
            query=query,
            retrieved_ids=retrieved_ids,
            ground_truth=ground_truth,
            metrics=metrics,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        with self._lock:
            self._runs.append(run)
            if len(self._runs) > self._max_runs:
                self._runs = self._runs[-self._max_runs:]

        return run

    def evaluate_generation_quality(
        self,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_contexts: List[str],
    ) -> GenerationQualityMetrics:
        faithfulness = self._compute_faithfulness(generated_answer, retrieved_contexts)
        relevance = self._compute_relevance(generated_answer, ground_truth_answer)
        completeness = self._compute_completeness(generated_answer, ground_truth_answer)
        conciseness = self._compute_conciseness(generated_answer)
        hallucination_score = self._compute_hallucination(generated_answer, retrieved_contexts)

        overall = (faithfulness + relevance + completeness + conciseness + (1.0 - hallucination_score)) / 5.0

        return GenerationQualityMetrics(
            faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4),
            completeness=round(completeness, 4),
            conciseness=round(conciseness, 4),
            hallucination_score=round(hallucination_score, 4),
            overall_score=round(overall, 4),
        )

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not answer or not contexts:
            return 0.0
        answer_lower = answer.lower()
        context_text = " ".join(contexts).lower()
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())
        if not answer_words:
            return 0.0
        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def _compute_relevance(self, answer: str, ground_truth: str) -> float:
        if not answer or not ground_truth:
            return 0.0
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        if not truth_words:
            return 1.0
        overlap = answer_words.intersection(truth_words)
        return len(overlap) / len(truth_words)

    def _compute_completeness(self, answer: str, ground_truth: str) -> float:
        if not ground_truth:
            return 1.0
        if not answer:
            return 0.0
        truth_sentences = [s.strip() for s in ground_truth.split(".") if s.strip()]
        if not truth_sentences:
            return 1.0
        answer_lower = answer.lower()
        covered = sum(1 for s in truth_sentences if s.lower() in answer_lower)
        return covered / len(truth_sentences)

    def _compute_conciseness(self, answer: str) -> float:
        if not answer:
            return 1.0
        word_count = len(answer.split())
        if word_count <= 50:
            return 1.0
        elif word_count <= 200:
            return 0.8
        elif word_count <= 500:
            return 0.6
        else:
            return 0.4

    def _compute_hallucination(self, answer: str, contexts: List[str]) -> float:
        if not answer:
            return 0.0
        if not contexts:
            return 0.5
        answer_entities = self._extract_entities(answer)
        context_entities = set()
        for ctx in contexts:
            context_entities.update(self._extract_entities(ctx))
        if not answer_entities:
            return 0.0
        hallucinated = answer_entities - context_entities
        return len(hallucinated) / len(answer_entities)

    def _extract_entities(self, text: str) -> set:
        import re
        words = re.findall(r"\b[A-Z][a-z]+\b", text)
        return set(words)

    def run_benchmark(self, pipeline) -> Dict[str, Any]:
        if not self._enabled:
            return {"status": "disabled"}

        eval_results = []
        for query, gt_ids in BENCHMARK_GROUND_TRUTH.items():
            start = time.time()
            res = pipeline.process_query(query, user_role="admin")
            latency_ms = (time.time() - start) * 1000

            retrieved_evidence = res.get("retrieved_evidence", [])
            retrieved_ids = [str(item.get("id")) for item in retrieved_evidence]

            metrics = RetrievalEvaluator.evaluate(retrieved_ids, gt_ids, k=5)
            metrics["latency_ms"] = round(latency_ms, 2)

            eval_results.append({
                "query": query,
                "retrieved": retrieved_ids,
                "ground_truth": gt_ids,
                "metrics": metrics
            })

        avg_metrics = {}
        metric_keys = ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5", "latency_ms"]
        for key in metric_keys:
            avg_metrics[key] = round(float(np.mean([item["metrics"][key] for item in eval_results])), 4)

        self._last_benchmark = time.time()

        return {
            "status": "success",
            "benchmark_runs": eval_results,
            "average_metrics": avg_metrics,
            "timestamp": self._last_benchmark,
        }

    def should_run_benchmark(self) -> bool:
        if not self._enabled or not self._auto_benchmark:
            return False
        elapsed = time.time() - self._last_benchmark
        return elapsed >= self._benchmark_interval

    def get_aggregate_metrics(self, window: int = 100) -> Dict[str, Any]:
        with self._lock:
            recent = self._runs[-window:] if self._runs else []

        if not recent:
            return {"status": "no_data"}

        metric_keys = ["precision_at_5", "recall_at_5", "mrr", "ndcg_at_5"]
        aggregates = {}
        for key in metric_keys:
            vals = [r.metrics.get(key, 0.0) for r in recent]
            aggregates[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        latencies = [r.latency_ms for r in recent]
        aggregates["latency_ms"] = {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        }

        return {
            "window_size": len(recent),
            "metrics": aggregates,
            "timestamp": time.time(),
        }

    def get_recent_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            recent = self._runs[-limit:]
        return [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "query": r.query[:100],
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
            }
            for r in recent
        ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_runs = len(self._runs)
        return {
            "enabled": self._enabled,
            "total_runs": total_runs,
            "auto_benchmark": self._auto_benchmark,
            "benchmark_interval": self._benchmark_interval,
            "last_benchmark": self._last_benchmark,
            "should_benchmark": self.should_run_benchmark(),
        }


_evaluation_engine_instance = None


def get_evaluation_engine() -> EvaluationEngine:
    global _evaluation_engine_instance
    if _evaluation_engine_instance is None:
        _evaluation_engine_instance = EvaluationEngine()
    return _evaluation_engine_instance