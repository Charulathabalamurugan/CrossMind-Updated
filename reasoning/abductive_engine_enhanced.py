import time
import logging
from typing import List, Dict, Any, Optional
from config import settings
from reasoning.wfa_fast_path import get_wfa_engine

logger = logging.getLogger("crossmind.abductive")

class EnhancedAbductiveEngine:
    def __init__(self):
        self.wfa = get_wfa_engine()
        self.candidates: List[Dict[str, Any]] = []

    def generate_candidates(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        filter_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self.candidates = []
        entities = filter_metadata.get("extracted_entities", [])
        domains = filter_metadata.get("detected_domains", [])
        top_k = settings.ABDUCTIVE_TOP_K

        if not entities:
            return [{
                "id": "no_entities",
                "explanation": "No entities detected, cannot generate causal hypotheses.",
                "causal_score": 0.0,
                "confidence": 0.1,
                "rank": 1,
            }]

        explanations = [
            {
                "id": "causal_bridge",
                "explanation": f"{entities[0]} mediates a causal link between {', '.join(domains)} domains.",
                "causal_score": 0.9,
                "confidence": 0.85,
            },
            {
                "id": "shared_pathway",
                "explanation": f"Shared pathway connects {entities[0]} across detected domains.",
                "causal_score": 0.75,
                "confidence": 0.65,
            },
        ]

        wfa_result, wfa_weight = self.wfa.classify_query(
            "causal" if len(domains) > 1 else "factual",
            max(e["confidence"] for e in explanations),
        )

        for idx, cand in enumerate(explanations):
            wfa_decision = self.wfa.fast_path_decision(
                "causal" if len(domains) > 1 else "factual",
                cand["confidence"],
                len(domains),
            )
            self.candidates.append({
                **cand,
                "rank": idx + 1,
                "wfa_path": wfa_result,
                "wfa_weight": wfa_weight,
                "wfa_decision": wfa_decision,
            })

        self.candidates.sort(key=lambda x: x["causal_score"], reverse=True)
        return self.candidates[:top_k]

    def get_best_explanation(self) -> Dict[str, Any]:
        if not self.candidates:
            return {"explanation": "No candidates generated.", "causal_score": 0.0}
        return self.candidates[0]

abductive_instance = None

def get_enhanced_abductive_engine() -> EnhancedAbductiveEngine:
    global abductive_instance
    if abductive_instance is None:
        abductive_instance = EnhancedAbductiveEngine()
    return abductive_instance