import time
import logging
from typing import Any, Dict, List, Optional
from reasoning.knowledge_graph import KnowledgeGraph

logger = logging.getLogger("crossmind.experimental_blueprint")


class ExperimentalBlueprintGenerator:
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()

    def generate(
        self,
        hypothesis: Dict[str, Any],
        abductive_result: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypothesis_text = hypothesis.get(
            "hypothesis", hypothesis.get("output_text", "")
        )
        causal_path = (
            abductive_result.get("causal_pathway", "")
            if abductive_result
            else ""
        )
        domains = list(
            {
                ev.get("payload", {}).get("domain", "general")
                for ev in evidence
            }
        )

        blueprint = {
            "generated_at": time.time(),
            "title": f"Experimental Blueprint: {hypothesis_text[:80]}...",
            "primary_objective": self._derive_objective(hypothesis_text),
            "hypothesis_under_test": hypothesis_text,
            "causal_pathway": causal_path,
            "methodology": self._build_methodology(domains, evidence),
            "controls": self._build_controls(domains),
            "materials": self._build_materials(evidence),
            "expected_outcomes": self._build_expected_outcomes(
                hypothesis_text, abductive_result
            ),
            "risk_mitigations": self._build_risks(domains),
            "timeline_estimate": (
                "12-18 months"
                if any(d in domains for d in ["nanotechnology", "pharmacology"])
                else "6-12 months"
            ),
            "readouts": self._build_readouts(domains),
            "confidence": self._estimate_confidence(evidence, abductive_result),
        }
        return blueprint

    def _derive_objective(self, hypothesis_text: str) -> str:
        h = hypothesis_text.lower()
        if (
            "cross-domain" in h
            or "link" in h
            or "bridge" in h
            or "connect" in h
        ):
            return "Validate and characterize the cross-domain mechanistic link identified in hypothesis."
        if "delivery" in h or "targeted" in h:
            return "Demonstrate targeted delivery efficiency and biocompatibility in relevant model systems."
        if "biomarker" in h or "imaging" in h:
            return "Quantify biomarker modulation and imaging contrast enhancement."
        return "Test the hypothesized mechanism with rigorous in vitro and in vivo experiments."

    def _build_methodology(
        self, domains: List[str], evidence: List[Dict[str, Any]]
    ) -> List[str]:
        steps = []
        if "nanotechnology" in domains:
            steps.append(
                "Synthesize and characterize nanoparticles (DLS, TEM, zeta potential)."
            )
            steps.append(
                "Validate surface functionalization and ligand density by XPS or NMR."
            )
        if "neuroscience" in domains:
            steps.append(
                "Primary cortical neuron culture and viability assays (MTT, LDH)."
            )
            steps.append(
                "Transwell BBB model with hCMEC/D3 or primary brain endothelial cells."
            )
        if "pharmacology" in domains:
            steps.append(
                "Dose-response curve and IC50 determination in relevant cell lines."
            )
            steps.append(
                "Stability profiling in serum (PBS, FBS at 37C over 72h)."
            )
        if not steps:
            steps = [
                "Literature replication study with standardized protocols.",
                "Statistical power analysis and effect size estimation.",
            ]
        steps.append(
            "Statistical analysis: n>=3 biological replicates, unpaired t-test or ANOVA with post-hoc correction."
        )
        return steps

    def _build_controls(self, domains: List[str]) -> List[str]:
        controls = [
            "Negative control: non-functionalized vehicle without therapeutic payload."
        ]
        if "nanotechnology" in domains:
            controls.append(
                "Positive control: clinically validated comparator nanocarrier if available."
            )
        if "neuroscience" in domains:
            controls.append(
                "Vehicle-only control on matched cell density and passage number."
            )
        controls.append(
            "Blind quantification by independent scorer to reduce operator bias."
        )
        return controls

    def _build_materials(self, evidence: List[Dict[str, Any]]) -> List[str]:
        materials = [
            "Synthesized nanocarrier batch with batch-to-batch QC documentation."
        ]
        seen = set()
        for ev in evidence:
            payload = ev.get("payload", {})
            tags = payload.get("tags", [])
            for tag in tags:
                t = str(tag).lower()
                if t not in seen and len(t) > 3:
                    seen.add(t)
                    materials.append(tag)
            if payload.get("domain") == "neuroscience":
                materials.append(
                    "Human induced pluripotent stem cell (iPSC)-derived neurons"
                )
        return materials[:12]

    def _build_expected_outcomes(
        self,
        hypothesis_text: str,
        abductive_result: Optional[Dict[str, Any]],
    ) -> List[str]:
        outcomes = [
            "Primary endpoint significantly outperforms vehicle control (p < 0.05)."
        ]
        if abductive_result:
            outcomes.append(
                f"Mechanistic validation of pathway: {abductive_result.get('causal_pathway', 'TBD')}."
            )
        outcomes.append(
            "Comprehensive biocompatibility profile: >85% viability across tested concentrations."
        )
        outcomes.append(
            "Replicable across at least two independent experimental batches."
        )
        return outcomes

    def _build_risks(self, domains: List[str]) -> List[str]:
        risks = [
            "Batch variability in nanomaterial synthesis; implement GMP-like SOPs and QC gates."
        ]
        if "neuroscience" in domains:
            risks.append(
                "Off-target effects on non-CNS cell populations; confirm cell-type specificity."
            )
        if "pharmacology" in domains:
            risks.append(
                "Unintended immunogenicity; include complement activation ELISA screening."
            )
        risks.append(
            "Interpretation bias from small sample sizes; pre-register analysis plan."
        )
        return risks

    def _build_readouts(self, domains: List[str]) -> List[str]:
        readouts = [
            "Primary efficacy metric (dose-response curve, EC50/IC50)."
        ]
        if "neuroscience" in domains:
            readouts.extend(
                [
                    "Neurite outgrowth quantification (Sholl analysis).",
                    "Cytokine panel (IL-1b, TNF-a, IL-6) by Luminex.",
                ]
            )
        if "nanotechnology" in domains:
            readouts.extend(
                [
                    "Particle size distribution by DLS (PDI < 0.2).",
                    "Encapsulation efficiency and loading capacity (%w/w).",
                ]
            )
        readouts.append(
            "Biocompatibility: cell viability >85% at projected therapeutic concentration."
        )
        return readouts

    def _estimate_confidence(
        self,
        evidence: List[Dict[str, Any]],
        abductive_result: Optional[Dict[str, Any]],
    ) -> str:
        if not evidence:
            return "low"
        avg_score = (
            sum(float(ev.get("score", 0.0)) for ev in evidence) / len(evidence)
        )
        if (
            abductive_result
            and abductive_result.get("abductive_score", 0) > 75
            and avg_score > 0.6
        ):
            return "high"
        if avg_score > 0.4:
            return "moderate"
        return "low"


_blueprint_instance: Optional[ExperimentalBlueprintGenerator] = None


def get_experimental_blueprint_generator() -> ExperimentalBlueprintGenerator:
    global _blueprint_instance
    if _blueprint_instance is None:
        _blueprint_instance = ExperimentalBlueprintGenerator()
    return _blueprint_instance
