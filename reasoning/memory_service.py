import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger("crossmind.memory")

class MemoryService:
    """
    Modular Memory Service separate from execution pipeline.
    Implements a three-level memory framework:
    1. Individual Level (Episodic, Semantic, Persona)
    2. Domain Level
    3. Interdisciplinary Level
    """
    def __init__(self):
        # Individual Memory
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, int] = {}  # term -> frequency of interest
        self.persona_memory: Dict[str, Any] = {
            "cognitive_style": "Analytical & Exploratory",
            "preferred_domains": [],
            "safety_focus_level": "High",  # based on rule compliance focus
            "interaction_count": 0
        }
        
        # Domain Memory
        self.domain_memory: Dict[str, int] = {
            "neuroscience": 0,
            "nanotechnology": 0,
            "pharmacology": 0,
            "general": 0
        }
        
        # Interdisciplinary Memory
        self.interdisciplinary_memory: Dict[str, int] = {}  # "domain_A <-> domain_B" -> link frequency
        
        logger.info("MirrorMind memory service initialized.")

    def add_interaction(self, query: str, result: Dict[str, Any]):
        """
        Distills past queries, query parameters, validation results, and entities
        into Episodic, Semantic, Persona, Domain, and Interdisciplinary memories.
        """
        self.persona_memory["interaction_count"] += 1
        
        # 1. Episodic Memory (Research interaction record)
        episode = {
            "timestamp": time.time(),
            "query": query,
            "hypothesis_summary": result.get("agent_reasoning", {}).get("output_text", "")[:120] + "...",
            "validation_passed": result.get("post_validation", {}).get("validated", False),
            "domains": result.get("pre_filter", {}).get("detected_domains", [])
        }
        self.episodic_memory.append(episode)
        
        # Keep episodic memory bounded to last 50 queries
        if len(self.episodic_memory) > 50:
            self.episodic_memory.pop(0)

        # 2. Semantic Memory (Accumulating entities of interest)
        entities = result.get("pre_filter", {}).get("extracted_entities", [])
        for entity in entities:
            self.semantic_memory[entity] = self.semantic_memory.get(entity, 0) + 1

        # 3. Domain Memory
        detected_domains = result.get("pre_filter", {}).get("detected_domains", [])
        for domain in detected_domains:
            dom = domain.lower()
            self.domain_memory[dom] = self.domain_memory.get(dom, 0) + 1
            if dom not in self.persona_memory["preferred_domains"]:
                self.persona_memory["preferred_domains"].append(dom)

        # 4. Interdisciplinary Memory (Links made between distinct domains)
        if len(detected_domains) >= 2:
            sorted_domains = sorted(list(set(detected_domains)))
            for i in range(len(sorted_domains)):
                for j in range(i + 1, len(sorted_domains)):
                    link_key = f"{sorted_domains[i]} <-> {sorted_domains[j]}"
                    self.interdisciplinary_memory[link_key] = self.interdisciplinary_memory.get(link_key, 0) + 1

        # 5. Persona Memory Adaptation
        # If user queries has failed validations, increase safety focus profile
        failed_count = sum(1 for ep in self.episodic_memory if not ep["validation_passed"])
        if failed_count > 2:
            self.persona_memory["safety_focus_level"] = "Rigorous"
        else:
            self.persona_memory["safety_focus_level"] = "Balanced"

    def get_relevant_context(self, query: str) -> str:
        """
        Scans episodic and semantic memories to retrieve relevant past contexts
        and cognitive style details to augment reasoning prompts.
        """
        # Find similar past queries in episodic memory (simple token matching)
        query_words = set(query.lower().split())
        similar_episodes = []
        for ep in self.episodic_memory:
            past_words = set(ep["query"].lower().split())
            intersection = query_words.intersection(past_words)
            if len(intersection) >= 2:
                similar_episodes.append(ep["query"])
                
        context_parts = []
        if similar_episodes:
            context_parts.append(f"Researcher previously explored related topics: {'; '.join(similar_episodes[:2])}.")
            
        # Add persona details to shape the prompt response style
        context_parts.append(f"Researcher preference profile: Style: {self.persona_memory['cognitive_style']}, Safety Threshold: {self.persona_memory['safety_focus_level']}.")
        
        return " ".join(context_parts)

    def get_summary(self) -> Dict[str, Any]:
        """Returns the full memory footprint for system dashboards."""
        return {
            "individual": {
                "episodic_count": len(self.episodic_memory),
                "episodes": self.episodic_memory[-5:], # return last 5 interactions
                "top_semantic_interests": sorted(self.semantic_memory.items(), key=lambda x: x[1], reverse=True)[:5],
                "persona": self.persona_memory
            },
            "domain": self.domain_memory,
            "interdisciplinary": self.interdisciplinary_memory
        }

_memory_service_instance = MemoryService()

def get_memory_service() -> MemoryService:
    return _memory_service_instance
