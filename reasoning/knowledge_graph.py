"""Lightweight, evidence-grounded knowledge graph and GraphRAG utilities."""
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List

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
        evidence_ids = {str(item.get("id")) for item in evidence}
        terms_by_doc: Dict[str, set] = {}
        for item in evidence:
            payload = item.get("payload", {})
            terms_by_doc[str(item.get("id"))] = {term.lower() for term in self._terms(payload)}

        # Query entities anchor the graph, then tags shared by different documents
        # form one-hop bridges. Documents from different domains sharing a bridge are
        # useful cross-domain / multi-hop evidence chains.
        anchors = {str(entity).lower() for entity in query_entities}
        shared = defaultdict(list)
        for doc_id, terms in terms_by_doc.items():
            for term in terms:
                shared[term].append(doc_id)

        nodes, edges, paths = [], [], []
        for item in evidence:
            doc_id, payload = str(item.get("id")), item.get("payload", {})
            nodes.append({"id": "doc:" + doc_id, "label": payload.get("title", doc_id), "type": "document", "domain": payload.get("domain", "general")})
            for term in sorted(terms_by_doc.get(doc_id, set())):
                entity_id = "entity:" + term
                nodes.append({"id": entity_id, "label": term, "type": "entity"})
                edges.append({"source": "doc:" + doc_id, "target": entity_id, "relation": "mentions"})

        evidence_scores = {str(item.get("id")): max(0.0, min(1.0, float(item.get("score", 0.0)))) for item in evidence}
        for term, doc_ids in shared.items():
            if len(doc_ids) < 2:
                continue
            for left, right in combinations(sorted(set(doc_ids)), 2):
                left_domain = next((e.get("payload", {}).get("domain") for e in evidence if str(e.get("id")) == left), "general")
                right_domain = next((e.get("payload", {}).get("domain") for e in evidence if str(e.get("id")) == right), "general")
                relevance = (evidence_scores.get(left, 0.0) + evidence_scores.get(right, 0.0)) / 2
                # A bridge that appears in fewer documents is more discriminative;
                # cross-domain paths receive a modest novelty bonus.
                novelty = min(1.0, 1.0 / len(doc_ids) + (0.25 if left_domain != right_domain else 0.0))
                score = 100 * (0.65 * relevance + 0.35 * novelty)
                paths.append({
                    "path": ["doc:" + left, "entity:" + term, "doc:" + right],
                    "bridge_entity": term,
                    "cross_domain": left_domain != right_domain,
                    "relevance_score": round(relevance * 100, 1),
                    "novelty_score": round(novelty * 100, 1),
                    "path_score": round(score, 1),
                })

        unique_nodes = {node["id"]: node for node in nodes}
        cross_domain_paths = [path for path in paths if path["cross_domain"]]
        return {
            "strategy": "vector_seed_then_graph_expansion",
            "seed_document_ids": sorted(evidence_ids),
            "query_anchors": sorted(anchors),
            "nodes": list(unique_nodes.values()),
            "edges": edges,
            "multi_hop_paths": sorted(paths, key=lambda path: path["path_score"], reverse=True)[:10],
            "cross_domain_path_count": len(cross_domain_paths),
            "path_scoring_method": "65% seed-document relevance, 35% bridge novelty (with cross-domain bonus)",
        }


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
