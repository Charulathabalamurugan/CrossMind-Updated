import os
import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger("crossmind.semara")


class SemaraReasoner:
    def __init__(self):
        self.enabled = getattr(settings, "SEMARA_ENABLED", False)
        self.impl = getattr(settings, "SEMARA_IMPL", "tech-mahindra")
        self.open_source_fallback = getattr(settings, "SEMARA_OPEN_SOURCE_FALLBACK", True)
        self.ontology_url = getattr(settings, "SEMARA_ONTOLOGY_URL", "")
        self.reasoning_results: Dict[str, Any] = {}

    def is_available(self) -> bool:
        if self.enabled:
            return True
        if self.open_source_fallback:
            return True
        return False

    def get_impl_name(self) -> str:
        if self.enabled and self.impl == "tech-mahindra":
            return "Tech Mahindra SEMARA"
        if self.open_source_fallback:
            return "Open-source SeMRA (rdflib-based)"
        return "Semara (unavailable)"

    def ground_semantics(
        self,
        entities: List[str],
        domains: List[str],
        query: str,
    ) -> Dict[str, Any]:
        results = {
            "semara_enabled": self.enabled,
            "implementation": self.get_impl_name(),
            "entities_grounded": len(entities),
            "domains_grounded": len(domains),
            "grounding_confidence": 0.0,
            "semantic_relations": [],
            "ontology_matches": [],
        }

        try:
            from rdflib import Graph, Namespace, Literal, URIRef
            from rdflib.namespace import RDF, RDFS, OWL, XSD
        except ImportError:
            results["grounding_confidence"] = 0.0
            results["fallback_reason"] = "rdflib not installed"
            return results

        g = Graph()
        EX = Namespace("http://crossmind.org/ontology/")
        g.bind("ex", EX)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)

        for entity in entities:
            uri = EX[entity.lower().replace(" ", "_")]
            g.add((uri, RDF.type, OWL.Thing))
            g.add((uri, RDFS.label, Literal(entity)))

        for domain in domains:
            domain_uri = EX[f"domain_{domain.lower().replace(' ', '_')}"]
            g.add((domain_uri, RDF.type, EX.Domain))
            g.add((domain_uri, RDFS.label, Literal(domain)))

        for s, p, o in g:
            results["semantic_relations"].append(f"{s} -> {p} -> {o}")

        results["grounding_confidence"] = min(1.0, len(entities) * 0.2 + len(domains) * 0.15)
        results["ontology_matches"] = list(g.subjects(RDF.type, OWL.Thing))

        self.reasoning_results[query] = results
        return results

    def get_semantic_score(self, query: str) -> float:
        if query in self.reasoning_results:
            return self.reasoning_results[query].get("grounding_confidence", 0.0)
        return 0.0