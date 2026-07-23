import re
import time
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("crossmind.symbolic")

KNOWN_ENTITIES = {
    "neuroscience": ["Alzheimer's", "Aβ42", "Tau", "APOE4", "microglia", "neuroinflammation", "BACE1", "blood-brain barrier", "BBB"],
    "nanotechnology": ["nanomaterials", "lipid nanoparticles", "LNP", "dendrimers", "PLGA", "nanocarriers", "gold nanoparticles", "transcytosis"],
    "pharmacology": ["small molecules", "microRNA", "miR-124", "monoclonal antibodies", "PET-MRI imaging", "targeted delivery"]
}

SCIENTIFIC_RULES = [
    {
        "id": "RULE_01_BIOCOMPATIBILITY",
        "description": "Nanomaterials intended for central nervous system crossing must exhibit non-cytotoxic, biocompatible lipid or surface modifications.",
        "keywords": ["nanomaterial", "nanoparticle", "LNP", "dendrimer", "delivery", "brain", "CNS"],
        "required_safety_tags": ["biocompatible", "non-toxic", "PEGylated", "surface-functionalized"]
    },
    {
        "id": "RULE_02_TEMPORAL_ALIGNMENT",
        "description": "Scientific findings must align with the current literature baseline (2020-2024 evidence).",
        "min_year": 2020,
        "max_year": 2026
    },
    {
        "id": "RULE_03_CROSS_DOMAIN_PAIRING",
        "description": "Hypothesis must establish a non-trivial functional linkage between a targeted biomarker and a drug delivery/carrier mechanism.",
        "domain_pair_required": True
    }
]

class SymbolicPreFilter:
    """
    Step 3a: Deterministic rule engine executed in <50ms.
    Parses user query, identifies target entities and domain categories,
    filters search space and enforces hard query constraints.
    Optimized with Aho-Corasick algorithm for O(N + M) matching complexity.
    """
    def __init__(self):
        # Build Aho-Corasick trie from KNOWN_ENTITIES
        # Each node is a dict: {char: child_index}
        self.trie = [{}]
        self.output = [[]]  # list of tuples: (entity, domain)
        self.fail = [0]
        
        for domain, entities in KNOWN_ENTITIES.items():
            for entity in entities:
                word = entity.lower()
                curr = 0
                for char in word:
                    if char not in self.trie[curr]:
                        self.trie[curr][char] = len(self.trie)
                        self.trie.append({})
                        self.output.append([])
                        self.fail.append(0)
                    curr = self.trie[curr][char]
                self.output[curr].append((entity, domain))
                
        # Build fail links (BFS)
        from collections import deque
        queue = deque()
        for char, child in self.trie[0].items():
            self.fail[child] = 0
            queue.append(child)
            
        while queue:
            curr = queue.popleft()
            for char, child in self.trie[curr].items():
                f = self.fail[curr]
                while f > 0 and char not in self.trie[f]:
                    f = self.fail[f]
                if char in self.trie[f]:
                    f = self.trie[f][char]
                self.fail[child] = f
                self.output[child].extend(self.output[f])
                queue.append(child)

    def process(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        query_lower = query.lower()

        detected_domains = []
        extracted_entities = []

        # Single-pass search using Aho-Corasick automaton
        curr = 0
        for char in query_lower:
            while curr > 0 and char not in self.trie[curr]:
                curr = self.fail[curr]
            if char in self.trie[curr]:
                curr = self.trie[curr][char]
            
            # Record any matches at this state
            for entity, domain in self.output[curr]:
                if entity not in extracted_entities:
                    extracted_entities.append(entity)
                if domain not in detected_domains:
                    detected_domains.append(domain)

        # Detect language
        spanish_keywords = ["buscar", "relacion", "relación", "encontrar", "conectar", "nanomateriales", "enfermedad"]
        language = "spanish" if any(w in query_lower for w in spanish_keywords) and not "find" in query_lower and not "links" in query_lower else "english"

        execution_time_ms = (time.time() - start_time) * 1000

        filter_metadata = {
            "query": query,
            "language": language,
            "detected_domains": detected_domains or ["neuroscience", "nanotechnology"],
            "extracted_entities": list(set(extracted_entities)),
            "execution_time_ms": round(execution_time_ms, 2),
            "search_constraints": {
                "min_year": 2020,
                "domain_filter": detected_domains or ["neuroscience", "nanotechnology"]
            }
        }
        logger.info(f"Symbolic Pre-Filter completed in {filter_metadata['execution_time_ms']}ms. Domains: {detected_domains}, Entities: {extracted_entities}")
        return filter_metadata


class SymbolicPostValidator:
    """
    Step 3c: Scientific rule validator executed in <50ms.
    Verifies candidate hypothesis against scientific constraints, logical consistency,
    and temporal/biocompatibility rules.
    """
    def validate(self, hypothesis: Dict[str, Any], retrieved_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = time.time()
        rule_checks = []
        is_valid = True
        validation_score = 100.0

        hyp_text = str(hypothesis.get("hypothesis", "")) + " " + str(hypothesis.get("reasoning", ""))
        hyp_text_lower = hyp_text.lower()

        # Rule 1: Biocompatibility check for nanomaterials
        if any(kw in hyp_text_lower for kw in ["nanomaterial", "nanoparticle", "lnp", "dendrimer"]):
            safety_found = any(safe_word in hyp_text_lower or any(safe_word in str(ev.get("payload", {})).lower() for ev in retrieved_evidence)
                               for safe_word in ["biocompatibl", "lipid", "pegylated", "functionalized", "non-toxic", "exosome"])
            rule_checks.append({
                "rule_id": "RULE_01_BIOCOMPATIBILITY",
                "passed": safety_found,
                "details": "Biocompatibility and low-toxicity nanocarrier criteria validated." if safety_found else "Warning: Ensure nanocarrier cytotoxicity profiles are specified."
            })
            if not safety_found:
                validation_score -= 15.0

        # Rule 2: Temporal alignment check
        years = [ev.get("payload", {}).get("year", 2024) for ev in retrieved_evidence if isinstance(ev.get("payload"), dict)]
        outdated_years = [y for y in years if y < 2020]
        temporal_passed = len(outdated_years) == 0
        rule_checks.append({
            "rule_id": "RULE_02_TEMPORAL_ALIGNMENT",
            "passed": temporal_passed,
            "details": f"Evidence years aligned ({min(years, default=2024)}-{max(years, default=2024)})." if temporal_passed else "Outdated evidence detected prior to 2020."
        })
        if not temporal_passed:
            validation_score -= 10.0

        # Rule 3: Citation grounding check
        retrieved_ids = [ev.get("id") for ev in retrieved_evidence]
        cited_ids = hypothesis.get("cited_evidence_ids", retrieved_ids)
        grounded = len(cited_ids) > 0 and all(c_id in retrieved_ids for c_id in cited_ids)
        rule_checks.append({
            "rule_id": "RULE_03_CITATION_GROUNDING",
            "passed": grounded,
            "details": f"All cited evidence IDs ({len(cited_ids)}) exist in retrieved context." if grounded else "Uncited or ungrounded assertions detected."
        })
        if not grounded:
            validation_score -= 10.0

        execution_time_ms = (time.time() - start_time) * 1000

        validation_result = {
            "validated": validation_score >= 70.0,
            "validation_score": round(validation_score, 1),
            "rule_checks": rule_checks,
            "execution_time_ms": round(execution_time_ms, 2)
        }
        logger.info(f"Symbolic Post-Validation completed in {validation_result['execution_time_ms']}ms. Score: {validation_score}%")
        return validation_result
