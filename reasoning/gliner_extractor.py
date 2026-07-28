import logging
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.gliner")

class GLiNERExtractor:
    def __init__(self):
        self.enabled = settings.GLiNER_ENABLED
        self.model = None
        if self.enabled:
            self._load_model()

    def _load_model(self):
        try:
            from gliner import GLiNERModel
            self.model = GLiNERModel()
            logger.info("GLiNER model loaded for entity extraction.")
        except ImportError:
            logger.info("GLiNER not installed. Entity extraction disabled.")
            self.enabled = False
        except Exception as exc:
            logger.warning(f"GLiNER load failed: {exc}. Entity extraction disabled.")
            self.enabled = False

    def extract_entities(self, text: str, ontology: str = "UMLS") -> List[Dict[str, Any]]:
        if not self.enabled or not self.model or not text.strip():
            return []
        try:
            entities = self.model.extract(text, ontology=ontology)
            return [
                {
                    "entity": e.get("text", ""),
                    "type": e.get("type", "unknown"),
                    "confidence": e.get("confidence", 0.0),
                    "ontology": ontology,
                    "start_pos": e.get("start_pos", 0),
                    "end_pos": e.get("end_pos", len(e.get("text", ""))),
                }
                for e in entities
                if e.get("confidence", 0.0) > 0.5
            ]
        except Exception as exc:
            logger.error(f"GLiNER extraction failed: {exc}")
            return []

    def link_to_ontology(self, entities: List[Dict[str, Any]], primary: str = "UMLS", fallbacks: List[str] = None) -> List[Dict[str, Any]]:
        fallbacks = fallbacks or ["MGI", "ChEBI"]
        linked = []
        for entity in entities:
            linked_entity = dict(entity)
            linked_entity["linked_id"] = None
            linked_entity["linking_confidence"] = 0.0
            linked_entity["source_ontology"] = None
            for ontology in [primary] + (fallbacks or []):
                match_id, match_conf = self._attempt_linking(entity["entity"], ontology)
                if match_conf > linked_entity["linking_confidence"]:
                    linked_entity["linked_id"] = match_id
                    linked_entity["linking_confidence"] = match_conf
                    linked_entity["source_ontology"] = ontology
                    if match_conf >= 0.9:
                        break
            if linked_entity["linking_confidence"] < 0.5:
                linked_entity["active_learning_queue"] = True
            linked.append(linked_entity)
        return linked

    def _attempt_linking(self, entity_text: str, ontology: str) -> tuple:
        return (f"{ontology}:{entity_text[:20].upper().replace(' ', '_')}", 0.5)

_gliner_instance = None

def get_gliner_extractor() -> GLiNERExtractor:
    global _gliner_instance
    if _gliner_instance is None:
        _gliner_instance = GLiNERExtractor()
    return _gliner_instance