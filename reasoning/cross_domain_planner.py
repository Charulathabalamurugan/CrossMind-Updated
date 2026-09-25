import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("crossmind.cross_domain_planner")


class DomainBridge:
    """A typed bridge between two scientific domains via a shared entity."""

    def __init__(
        self,
        bridge_entity: str,
        source_domain: str,
        target_domain: str,
        source_doc_id: str,
        target_doc_id: str,
        path_score: float,
        evidence_scores: List[float],
    ) -> None:
        self.bridge_entity = bridge_entity
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_doc_id = source_doc_id
        self.target_doc_id = target_doc_id
        self.path_score = path_score
        self.evidence_scores = evidence_scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_entity": self.bridge_entity,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "source_doc_id": self.source_doc_id,
            "target_doc_id": self.target_doc_id,
            "path_score": self.path_score,
            "evidence_scores": self.evidence_scores,
            "cross_domain": self.source_domain != self.target_domain,
        }


class CrossDomainPlan:
    """Structured plan consumed by the main neuro-symbolic pipeline."""

    def __init__(
        self,
        query: str,
        detected_domains: List[str],
        extracted_entities: List[str],
        bridges: List[DomainBridge],
        ranked_hypotheses: List[Dict[str, Any]],
        domain_coverage: Dict[str, int],
        plan_score: float,
        plan_rating: str,
        warnings: List[str],
    ) -> None:
        self.query = query
        self.detected_domains = detected_domains
        self.extracted_entities = extracted_entities
        self.bridges = bridges
        self.ranked_hypotheses = ranked_hypotheses
        self.domain_coverage = domain_coverage
        self.plan_score = plan_score
        self.plan_rating = plan_rating
        self.warnings = warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "detected_domains": self.detected_domains,
            "extracted_entities": self.extracted_entities,
            "bridges": [b.to_dict() for b in self.bridges],
            "ranked_hypotheses": self.ranked_hypotheses,
            "domain_coverage": self.domain_coverage,
            "plan_score": self.plan_score,
            "plan_rating": self.plan_rating,
            "warnings": self.warnings,
            "bridge_count": len(self.bridges),
            "cross_domain_bridge_count": sum(
                1 for b in self.bridges if b.source_domain != b.target_domain
            ),
        }