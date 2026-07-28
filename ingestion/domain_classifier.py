import re
import logging
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger("crossmind.domain_classifier")

DOMAIN_KEYWORDS = {
    "neuroscience": [
        "neuron", "neuronal", "synapse", "synaptic", "neurotransmitter", "dopamine",
        "serotonin", "acetylcholine", "glutamate", "gaba", "cortex", "hippocampus",
        "amygdala", "cerebellum", "prefrontal", "Alzheimer", "dementia", "Parkinson",
        "schizophrenia", "depression", "anxiety", "BDNF", "CREB", "LTP", "long-term",
        "potentiation", "neuroplasticity", "amyloid", "tau", "alpha-synuclein",
        "prion", "prion disease", "prion-like", "amyotrophic", "lateral", "sclerosis",
        "multiple", "sclerosis", "myelin", "oligodendrocyte", "astrocyte", "microglia",
        "blood-brain barrier", "blood brain barrier", "BBB", "blood cerebral",
        "neuroinflammation", "neurodegenerative", "neuroprotective", "neurotoxicity",
        "cognitive", "hippocampal", "cortical", "subcortical", "dopaminergic",
        "serotonergic", "cholinergic", "GABAergic", "glutamatergic", "opioid",
        "receptor", "receptors", "ligand", "ligands", "transporter", "transporters",
        "gene", "genomic", "epigenetic", "methylation", "phosphorylation",
    ],
    "nanotechnology": [
        "nanoparticle", "nanoparticles", "nanomaterial", "nanomaterials", "nanocarrier",
        "nanocarriers", "lipid nanoparticle", "LNP", "lipid nanoparticle", "polymeric",
        "nanocarrier", "dendrimer", "dendrimers", "PLGA", "PEG", "pegylated",
        "functionalized", "surface modification", "zeta potential", "DLS", "TEM",
        "NMR", "XPS", "FTIR", "hydrodynamic", "dispersity", "colloidal", "stability",
        "cellular uptake", "endocytosis", "phagocytosis", "lysosome", "cytotoxicity",
        "intracellular", "extracellular", "transferrin", "receptor-mediated",
        "transcytosis", "endothelial", "hCMEC", "astrocyte", "pericyte", "tight junction",
        "blood brain barrier", "BBB", "bioavailability", "bioavailability",
        "encapsulation", "loading", "payload", "controlled release", "sustained release",
        "crosslinking", "self-assembly", "amphiphilic", "hydrophobic", "hydrophilic",
        "core-shell", "shell", "core", "micelle", "vesicle", "liposome", "liposomes",
    ],
    "pharmacology": [
        "drug", "drugs", "pharmacokinetics", "pharmacodynamic", "bioavailability",
        "bioavailability", "metabolism", "metabolite", "metabolites", "excretion",
        "absorption", "distribution", "pharmacogenomics", "therapeutic", "therapy",
        "therapeutic", "dose", "dosage", "administration", "oral", "intravenous", "IV",
        "intramuscular", "IM", "subcutaneous", "SC", "topical", "inhalation",
        "bioequivalence", "bioequivalence", "FDA", "IND", "NDA", "clinical trial",
        "clinical", "toxicity", "toxicology", "adverse", "side effect", "contraindication",
        "interaction", "formulation", "excipient", "active pharmaceutical ingredient",
        "API", "pharmaceutical", "compound", "compounds", "molecule", "molecules",
        "binding", "affinity", "potency", "efficacy", "potency", "IC50", "EC50",
        "LD50", " therapeutic index", "bioavailability", "first-pass metabolism",
        "half-life", "half life", "clearance", "volume of distribution", "Vd",
        "protein binding", "plasma protein", "albumin", "alpha-1 acid glycoprotein",
    ],
    "energy": [
        "battery", "batteries", "lithium", "sodium", "ion", "solid-state", "fuel cell",
        "solar", "photovoltaic", "thermoelectric", "supercapacitor", "capacitor",
        "energy storage", "hydrogen", "electrolyte", "catalyst", "catalysis",
        "photocatalytic", "electrocatalyst", "anode", "cathode", "separator",
        "electrode", "graphene", "carbon nanotube", "CNT", "transition metal",
        "oxide", "sulfide", "phosphate", "sulfate", "nitride", "carbide",
    ],
    "financial": [
        "finance", "financial", "market", "trading", "investment", "portfolio",
        "stock", "bond", "derivative", "option", "futures", "hedge", "hedging",
        "risk", "liquidity", "volatility", "return", "alpha", "beta", "Sharpe",
        "capital", "asset", "dividend", "yield", "interest", "credit", "default",
        "mortgage", "insurance", "banking", "cryptocurrency", "bitcoin", "blockchain",
    ],
}

class DomainClassifier:
    def __init__(self):
        self.domains = list(DOMAIN_KEYWORDS.keys())
        self.compiled_patterns = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            pattern = re.compile(
                r"\b(" + "|".join(re.escape(kw) for kw in keywords)
                + r")\b", re.IGNORECASE
            )
            self.compiled_patterns[domain] = pattern
        self.fallback_domain = settings.FALLBACK_DOMAIN if hasattr(settings, "FALLBACK_DOMAIN") else "general"

    def classify(self, text: str) -> Dict[str, Any]:
        scores = {}
        for domain, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            scores[domain] = len(matches)
        total = sum(scores.values())
        if total == 0:
            return {"domain": self.fallback_domain, "confidence": 0.0, "scores": scores}
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain = sorted_domains[0][0]
        top_score = sorted_domains[0][1]
        confidence = round(top_score / max(total, 1), 3) if total > 0 else 0.0
        detected_domains = [d for d, s in sorted_domains if s > 0]
        return {
            "domain": top_domain,
            "confidence": confidence,
            "scores": {d: s for d, s in sorted_domains if s > 0},
            "detected_domains": detected_domains,
        }

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.classify(t) for t in texts]

_classifier_instance = None

def get_domain_classifier() -> DomainClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DomainClassifier()
    return _classifier_instance
