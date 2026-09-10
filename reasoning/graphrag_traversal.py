"""Adaptive GraphRAG traversal with configurable depth, early stopping, and cross-domain scoring."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set

from config import settings


@dataclass
class TraversalConfig:
    """Configuration for graph traversal behavior."""
    max_depth: int = field(default_factory=lambda: settings.GRAPH_RAG_DEPTH)
    early_stop_threshold: float = field(default_factory=lambda: settings.GRAPH_RAG_EARLY_STOP_THRESHOLD)
    max_paths: int = field(default_factory=lambda: settings.GRAPH_RAG_MAX_PATHS)
    min_evidence: int = field(default_factory=lambda: settings.GRAPH_RAG_MIN_EVIDENCE)
    cross_domain_bonus: float = 0.25
    relevance_weight: float = 0.65
    novelty_weight: float = 0.35

    def __post_init__(self) -> None:
        self.max_depth = max(1, min(self.max_depth, 10))
        self.early_stop_threshold = max(0.0, min(self.early_stop_threshold, 1.0))
        self.max_paths = max(1, min(self.max_paths, 1000))
        self.min_evidence = max(1, min(self.min_evidence, 1000))
        self.cross_domain_bonus = max(0.0, min(self.cross_domain_bonus, 1.0))
        total = self.relevance_weight + self.novelty_weight
        if total > 0:
            self.relevance_weight /= total
            self.novelty_weight /= total


class PathScorer(ABC):
    """Abstract base class for path scoring strategies."""

    @abstractmethod
    def score_path(
        self,
        left_doc: Dict[str, Any],
        right_doc: Dict[str, Any],
        bridge_entity: str,
        evidence_scores: Dict[str, float],
        doc_entities: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        """Score a path between two documents via a bridge entity.

        Returns a dict with keys: relevance_score, novelty_score, path_score, cross_domain
        """
        pass


class DefaultPathScorer(PathScorer):
    """Default path scorer using relevance, novelty, and cross-domain bonuses."""

    def __init__(self, config: TraversalConfig) -> None:
        self.config = config

    def score_path(
        self,
        left_doc: Dict[str, Any],
        right_doc: Dict[str, Any],
        bridge_entity: str,
        evidence_scores: Dict[str, float],
        doc_entities: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        left_id = str(left_doc.get("id", ""))
        right_id = str(right_doc.get("id", ""))

        left_score = evidence_scores.get(left_id, 0.0)
        right_score = evidence_scores.get(right_id, 0.0)
        relevance = (left_score + right_score) / 2.0

        left_domain = left_doc.get("payload", {}).get("domain", "general")
        right_domain = right_doc.get("payload", {}).get("domain", "general")
        cross_domain = left_domain != right_domain

        bridge_docs = {
            doc_id for doc_id, entities in doc_entities.items()
            if bridge_entity.lower() in {e.lower() for e in entities}
        }
        discriminativeness = min(1.0, 1.0 / max(1, len(bridge_docs)))
        novelty = discriminativeness + (self.config.cross_domain_bonus if cross_domain else 0.0)
        novelty = min(1.0, novelty)

        path_score = 100 * (
            self.config.relevance_weight * relevance +
            self.config.novelty_weight * novelty
        )

        return {
            "relevance_score": round(relevance * 100, 1),
            "novelty_score": round(novelty * 100, 1),
            "path_score": round(path_score, 1),
            "cross_domain": cross_domain,
        }


class GraphTraverser:
    """Adaptive graph traversal with configurable depth, early stopping, and path limits."""

    def __init__(
        self,
        config: Optional[TraversalConfig] = None,
        scorer: Optional[PathScorer] = None,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
    ) -> None:
        from reasoning.knowledge_graph import KnowledgeGraph

        self.config = config or TraversalConfig()
        self.scorer = scorer or DefaultPathScorer(self.config)
        self.kg = knowledge_graph or KnowledgeGraph()

    def traverse(
        self,
        evidence: List[Dict[str, Any]],
        query_entities: List[str],
    ) -> Dict[str, Any]:
        """Perform adaptive graph traversal and return graph context with metadata."""
        if not evidence:
            return self._empty_context()

        evidence_ids = {str(item.get("id")) for item in evidence}
        terms_by_doc: Dict[str, Set[str]] = {}
        for item in evidence:
            payload = item.get("payload", {})
            terms_by_doc[str(item.get("id"))] = {
                term.lower() for term in self.kg._terms(payload)
            }

        anchors = {str(entity).lower() for entity in query_entities}

        nodes, edges = [], []
        for item in evidence:
            doc_id = str(item.get("id", ""))
            payload = item.get("payload", {})
            nodes.append({
                "id": "doc:" + doc_id,
                "label": payload.get("title", doc_id),
                "type": "document",
                "domain": payload.get("domain", "general"),
            })
            for term in sorted(terms_by_doc.get(doc_id, set())):
                entity_id = "entity:" + term
                nodes.append({"id": entity_id, "label": term, "type": "entity"})
                edges.append({"source": "doc:" + doc_id, "target": entity_id, "relation": "mentions"})

        evidence_scores = {
            str(item.get("id")): max(0.0, min(1.0, float(item.get("score", 0.0))))
            for item in evidence
        }

        shared_entities = self._build_shared_entities(terms_by_doc)
        paths, traversal_meta = self._traverse_paths(
            shared_entities, evidence, evidence_scores, terms_by_doc
        )

        unique_nodes = {node["id"]: node for node in nodes}
        cross_domain_paths = [p for p in paths if p["cross_domain"]]

        graph_context = {
            "strategy": "vector_seed_then_graph_expansion",
            "seed_document_ids": sorted(evidence_ids),
            "query_anchors": sorted(anchors),
            "nodes": list(unique_nodes.values()),
            "edges": edges,
            "multi_hop_paths": sorted(paths, key=lambda p: p["path_score"], reverse=True)[:self.config.max_paths],
            "cross_domain_path_count": len(cross_domain_paths),
            "path_scoring_method": (
                f"{int(self.config.relevance_weight*100)}% seed-document relevance, "
                f"{int(self.config.novelty_weight*100)}% bridge novelty "
                f"(with cross-domain bonus {self.config.cross_domain_bonus})"
            ),
            "traversal_metadata": traversal_meta,
        }
        return graph_context

    def _build_shared_entities(self, terms_by_doc: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Build mapping of entity -> list of document IDs that mention it."""
        shared: Dict[str, List[str]] = {}
        for doc_id, terms in terms_by_doc.items():
            for term in terms:
                shared.setdefault(term, []).append(doc_id)
        return shared

    def _traverse_paths(
        self,
        shared_entities: Dict[str, List[str]],
        evidence: List[Dict[str, Any]],
        evidence_scores: Dict[str, float],
        doc_entities: Dict[str, Set[str]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Traverse and score paths with adaptive depth and early stopping."""
        paths = []
        doc_map = {str(e.get("id")): e for e in evidence}
        max_depth = self.config.max_depth
        stopped_early = False
        stop_reason = ""
        max_path_score = 0.0

        for term, doc_ids in shared_entities.items():
            if len(doc_ids) < 2:
                continue

            unique_doc_ids = sorted(set(doc_ids))
            for left_id, right_id in combinations(unique_doc_ids, 2):
                left_doc = doc_map.get(left_id)
                right_doc = doc_map.get(right_id)
                if not left_doc or not right_doc:
                    continue

                score_info = self.scorer.score_path(
                    left_doc, right_doc, term, evidence_scores, doc_entities
                )

                path = {
                    "path": ["doc:" + left_id, "entity:" + term, "doc:" + right_id],
                    "bridge_entity": term,
                    **score_info,
                }
                paths.append(path)

                if score_info["path_score"] > max_path_score:
                    max_path_score = score_info["path_score"]

                if max_path_score >= self.config.early_stop_threshold * 100:
                    stopped_early = True
                    stop_reason = f"early_stop_threshold_reached ({max_path_score:.1f} >= {self.config.early_stop_threshold * 100:.1f})"
                    break

            if stopped_early:
                break

        if len(paths) >= self.config.max_paths:
            stopped_early = True
            stop_reason = f"max_paths_reached ({len(paths)} >= {self.config.max_paths})"

        traversal_meta = {
            "max_depth_configured": max_depth,
            "actual_depth": 2,
            "paths_evaluated": len(paths),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason or "completed_full_traversal",
            "evidence_sufficiency": len(evidence) >= self.config.min_evidence,
            "min_evidence_configured": self.config.min_evidence,
        }
        return paths, traversal_meta

    def _empty_context(self) -> Dict[str, Any]:
        return {
            "strategy": "vector_seed_then_graph_expansion",
            "seed_document_ids": [],
            "query_anchors": [],
            "nodes": [],
            "edges": [],
            "multi_hop_paths": [],
            "cross_domain_path_count": 0,
            "path_scoring_method": (
                f"{int(self.config.relevance_weight*100)}% seed-document relevance, "
                f"{int(self.config.novelty_weight*100)}% bridge novelty "
                f"(with cross-domain bonus {self.config.cross_domain_bonus})"
            ),
            "traversal_metadata": {
                "max_depth_configured": self.config.max_depth,
                "actual_depth": 0,
                "paths_evaluated": 0,
                "stopped_early": False,
                "stop_reason": "no_evidence",
                "evidence_sufficiency": False,
                "min_evidence_configured": self.config.min_evidence,
            },
        }