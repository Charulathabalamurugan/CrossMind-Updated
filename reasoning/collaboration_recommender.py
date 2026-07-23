import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("crossmind.collaboration_recommender")


class CollaborationRecommender:
    def __init__(self):
        self.expertise_matrix = {
            "neuroscience": [
                "Neurobiologist",
                "Computational Neuroscientist",
                "Clinical Neurologist",
            ],
            "nanotechnology": [
                "Nanomaterials Scientist",
                "Nanomedicine Engineer",
                "Physical Chemist",
            ],
            "pharmacology": [
                "Pharmacologist",
                "Medicinal Chemist",
                "Toxicologist",
            ],
            "cross_domain": [
                "Translational Scientist",
                "Biomedical Engineer",
                "Systems Biologist",
            ],
        }
        self.role_skills = {
            "researcher": [
                "experimental_design",
                "data_analysis",
                "literature_review",
            ],
            "engineer": [
                "nanofabrication",
                "microfluidics",
                "instrumentation",
            ],
            "clinician": [
                "patient_cohort_selection",
                "ethical_oversight",
                "biomarker_validation",
            ],
        }

    def recommend(
        self,
        discovered_bridges: List[Dict[str, Any]],
        domains: List[str],
        team_size: int = 3,
    ) -> Dict[str, Any]:
        seen_expertise = set()
        recommended = []
        bridge_entities = []
        for bridge in discovered_bridges:
            bridge_entities.append(bridge.get("bridge_entity", "unknown"))
        for domain in domains:
            experts = self.expertise_matrix.get(
                domain, self.expertise_matrix.get("cross_domain", [])
            )
            for expert in experts:
                if (
                    len(recommended) < team_size
                    and expert not in seen_expertise
                ):
                    recommended.append(
                        {
                            "recommended_role": expert,
                            "primary_domain": domain,
                            "rationale": f"Domain expertise in {domain} required to validate {bridge_entities[:2]} bridge entities.",
                            "collaboration_strength": round(
                                min(1.0, 0.5 + 0.1 * len(bridge_entities)), 2
                            ),
                            "suggested_stage": (
                                "early"
                                if domain in ["nanotechnology", "neuroscience"]
                                else "validation"
                            ),
                        }
                    )
                    seen_expertise.add(expert)
        if not recommended:
            recommended.append(
                {
                    "recommended_role": "Computational Scientist",
                    "primary_domain": domains[0]
                    if domains
                    else "cross_domain",
                    "rationale": "Lead integration of cross-domain evidence into testable model.",
                    "collaboration_strength": 0.6,
                    "suggested_stage": "early",
                }
            )
        bridge_summary = {
            "bridge_entities": bridge_entities[:10],
            "cross_domain_links": sum(
                1 for b in discovered_bridges if b.get("cross_domain")
            ),
            "recommended_team_size": len(recommended),
        }
        return {
            "recommendations": recommended,
            "bridge_summary": bridge_summary,
            "generated_at": time.time(),
        }


_recommender_instance: Optional[CollaborationRecommender] = None


def get_collaboration_recommender() -> CollaborationRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = CollaborationRecommender()
    return _recommender_instance
