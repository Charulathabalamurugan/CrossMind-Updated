import logging
import re
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger("crossmind.query_preprocessor")

class QueryPreprocessor:
    def __init__(self):
        self.max_query_length = settings.MAX_QUERY_LENGTH
        self.token_pattern = re.compile(r"[\w'-]+")

    def tokenize(self, query: str) -> List[str]:
        if not query:
            return []
        tokens = self.token_pattern.findall(query.lower())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "because", "but", "and", "or", "if", "while",
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def preprocess(self, query: str) -> Dict[str, Any]:
        tokens = self.tokenize(query)
        return {
            "original_query": query,
            "tokens": tokens,
            "token_count": len(tokens),
            "normalized": " ".join(tokens),
        }

_preprocessor_instance = None

def get_query_preprocessor() -> QueryPreprocessor:
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = QueryPreprocessor()
    return _preprocessor_instance