"""Lightweight, evidence-grounded knowledge graph and GraphRAG utilities."""
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List

from config import settings
from reasoning.graphrag_traversal import GraphTraverser, TraversalConfig

STRUCTURAL_ENTITIES = (
    "alzheimer's", "aβ42", "tau", "apoe4", "microglia", "neuroinflammation",
    "blood-brain barrier", "bbb", "bace1", "nanomaterials", "lipid nanoparticles",
    "lnp", "dendrimers", "plga", "nanocarriers", "transcytosis", "microrna",
)


class KnowledgeGraph:
    """Builds a typed graph from document metadata without external graph storage.

    The graph deliberately stores only document metadata, tags and entities already
    supplied at ingestion time.  This keeps GraphRAG retrieval subject to the same
    RBAC filtering that protects the vector search results.
    """
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_documents: Dict[str, set] = defaultdict(set)

    @staticmethod
    def _terms(document: Dict[str, Any]) -> List[str]:
        tags = document.get("tags", []) or []
        text = " ".join((str(document.get("title", "")), str(document.get("content", "")), " ".join(map(str, tags)))).lower()
        # Curated scientific entities provide bridges even when two papers use
        # different free-form tags for the same mechanism.
        recognized = [entity for entity in STRUCTURAL_ENTITIES if entity in text]
        return list(dict.fromkeys([str(tag).strip() for tag in tags if str(tag).strip()] + recognized))

    def index_documents(self, documents: Iterable[Dict[str, Any]]) -> None:
        for document in documents:
            doc_id = str(document.get("id", ""))
            if not doc_id:
                continue
            old = self.documents.get(doc_id)
            if old:
                for term in self._terms(old):
                    self.entity_documents[term.lower()].discard(doc_id)
            self.documents[doc_id] = dict(document)
            for term in self._terms(document):
                self.entity_documents[term.lower()].add(doc_id)

    def graph_rag_context(self, evidence: List[Dict[str, Any]], query_entities: List[str]) -> Dict[str, Any]:
        """Return multi-hop paths and a compact typed subgraph for retrieved evidence."""
        config = TraversalConfig(
            max_depth=settings.GRAPH_RAG_DEPTH,
            early_stop_threshold=settings.GRAPH_RAG_EARLY_STOP_THRESHOLD,
            max_paths=settings.GRAPH_RAG_MAX_PATHS,
            min_evidence=settings.GRAPH_RAG_MIN_EVIDENCE,
        )
        traverser = GraphTraverser(config=config, knowledge_graph=self)
        return traverser.traverse(evidence, query_entities)


class DiscoveryScorer:
    """Produces an inspectable 0-100 score rather than a model-only assertion."""
    @staticmethod
    def score(evidence: List[Dict[str, Any]], graph_context: Dict[str, Any]) -> Dict[str, Any]:
        domains = {e.get("payload", {}).get("domain", "general") for e in evidence}
        raw_scores = [max(0.0, min(1.0, float(e.get("score", 0.0)))) for e in evidence]
        relevance = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        evidence_coverage = min(len(evidence) / 5.0, 1.0)
        domain_diversity = min(len(domains) / 3.0, 1.0)
        bridge_strength = min(graph_context.get("cross_domain_path_count", 0) / 2.0, 1.0)
        overall = 100 * (0.30 * relevance + 0.25 * evidence_coverage + 0.25 * domain_diversity + 0.20 * bridge_strength)
        return {
            "overall_score": round(overall, 1),
            "rating": "strong" if overall >= 75 else "promising" if overall >= 50 else "preliminary",
            "components": {
                "semantic_relevance": round(relevance * 100, 1),
                "evidence_coverage": round(evidence_coverage * 100, 1),
                "domain_diversity": round(domain_diversity * 100, 1),
                "structural_bridge_strength": round(bridge_strength * 100, 1),
            },
            "evidence_domains": sorted(domains),
            "method": "30% relevance, 25% evidence coverage, 25% domain diversity, 20% graph bridge strength",
        }


class ConfidenceCalibrator:
    """Shrinks model confidence toward independently observable evidence quality."""
    @staticmethod
    def calibrate(raw_confidence: float, discovery: Dict[str, Any], validation: Dict[str, Any], thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        raw = max(0.0, min(1.0, float(raw_confidence)))
        evidence_signal = discovery["overall_score"] / 100.0
        validation_signal = validation["validation_score"] / 100.0
        calibrated = 0.35 * raw + 0.40 * evidence_signal + 0.25 * validation_signal
        spread = max(0.05, (1.0 - evidence_signal) * 0.22)
        thresholds = thresholds or {}
        proceed_threshold = float(thresholds.get("proceed", 0.75))
        investigate_threshold = float(thresholds.get("investigate", 0.50))
        if not 0 <= investigate_threshold <= proceed_threshold <= 1:
            raise ValueError("Confidence thresholds must satisfy 0 <= investigate <= proceed <= 1")
        return {
            "raw_model_confidence": round(raw, 3),
            "calibrated_confidence": round(calibrated, 3),
            "confidence_interval": [round(max(0.0, calibrated - spread), 3), round(min(1.0, calibrated + spread), 3)],
            "decision": "proceed_to_experimental_design" if calibrated >= proceed_threshold else "seek_more_evidence" if calibrated >= investigate_threshold else "do_not_act_without_validation",
            "thresholds": {"proceed": proceed_threshold, "investigate": investigate_threshold},
            "basis": "35% model estimate, 40% discovery strength, 25% symbolic validation",
        }


_knowledge_graph = KnowledgeGraph()


def get_knowledge_graph() -> KnowledgeGraph:
    return _knowledge_graph
