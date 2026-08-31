from typing import Any, Dict, List


class GLiNERExtractor:
    """Simple compatibility implementation for entity extraction."""

    def __init__(self):
        self.entity_types = ["disease", "gene", "drug", "battery", "market", "signal"]

    def extract(self, text: str) -> List[Dict[str, Any]]:
        normalized = text.lower()
        extracted = []
        for entity in self.entity_types:
            if entity in normalized:
                extracted.append({"text": entity, "label": entity})
        return extracted


_extractor_instance = GLiNERExtractor()


def get_gliner_extractor() -> GLiNERExtractor:
    return _extractor_instance


__all__ = ["GLiNERExtractor", "get_gliner_extractor"]
