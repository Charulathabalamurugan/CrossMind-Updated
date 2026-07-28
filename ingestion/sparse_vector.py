import math
import logging
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger("crossmind.sparse_vector")

class SparseVectorEngine:
    def __init__(self):
        self.document_vectors: Dict[str, Dict[str, float]] = {}
        self.idf_weights: Dict[str, float] = {}
        self.total_documents = 0
        self._built = False

    def tokenize(self, text: str) -> List[str]:
        import re
        text = text.lower()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "but", "and", "or", "if", "while",
        }
        words = re.findall(r"[\w'-]+", text)
        return [w for w in words if w not in stop_words and len(w) > 2]

    def index_documents(self, documents: List[Dict[str, Any]]):
        doc_freq = Counter()
        for doc in documents:
            doc_id = doc.get("id", doc.get("chunk_index", str(hash(str(doc)))))
            text = doc.get("text", "")
            tokens = self.tokenize(text)
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)
            self.document_vectors[doc_id] = Counter(tokens)
            self.total_documents += 1
        for term, count in doc_freq.items():
            self.idf_weights[term] = math.log(
                (self.total_documents + 1) / (count + 1)
            ) + 1
        self._built = True
        logger.info(f"Indexed {self.total_documents} documents for sparse retrieval. Vocabulary size: {len(self.idf_weights)}")

    def compute_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        if not self._built or self.total_documents == 0:
            return {}
        tf = Counter(tokens)
        tfidf = {}
        for term, count in tf.items():
            idf = self.idf_weights.get(term, 1.0)
            tfidf[term] = (1 + math.log(count)) * idf if count > 0 else 0.0
        norm = math.sqrt(sum(v ** 2 for v in tfidf.values()))
        if norm > 0:
            tfidf = {k: v / norm for k, v in tfidf.items()}
        return tfidf

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._built or not self.document_vectors:
            return []
        query_tokens = self.tokenize(query)
        query_vec = self.compute_tfidf(query_tokens)
        if not query_vec:
            return []
        results = []
        for doc_id, doc_vec in self.document_vectors.items():
            dot = sum(query_vec.get(term, 0.0) * weight for term, weight in doc_vec.items())
            if dot > 0:
                results.append({"doc_id": doc_id, "score": dot})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

_sparse_instance = None

def get_sparse_vector_engine() -> SparseVectorEngine:
    global _sparse_instance
    if _sparse_instance is None:
        _sparse_instance = SparseVectorEngine()
    return _sparse_instance