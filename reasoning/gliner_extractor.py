from typing import Any, Dict, List


class GLiNERExtractor:
    def __init__(self):
        self.name = "gliner_extractor"

    def extract(self, text: str) -> List[Dict[str, Any]]:
        terms = []
        for chunk in text.split():
            if len(chunk) > 3:
                terms.append({"text": chunk, "label": "entity"})
        return terms


def get_gliner_extractor() -> GLiNERExtractor:
    return GLiNERExtractor()


__all__ = ["GLiNERExtractor", "get_gliner_extractor"]
