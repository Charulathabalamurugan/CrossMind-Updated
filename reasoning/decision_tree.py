import logging
from typing import Dict, Any, List
from config import settings

logger = logging.getLogger("crossmind.decision_tree")

class DecisionTreeNode:
    def __init__(self, feature: str = None, threshold: float = None, label: str = None):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.path_log: List[Dict[str, Any]] = []

    def _build_tree(self, features: list, labels: list, depth: int = 0) -> DecisionTreeNode:
        if len(set(labels)) == 1:
            return DecisionTreeNode(label=labels[0])
        if depth >= self.max_depth or len(features) < self.min_samples_split:
            return DecisionTreeNode(label=max(set(labels), key=labels.count))
        return DecisionTreeNode(label=self._majority_label(labels))

    def _majority_label(self, labels: list) -> str:
        from collections import Counter
        return Counter(labels).most_common(1)[0][0]

    def predict(self, features: Dict[str, Any]) -> str:
        if self.root is None:
            return "fast_path"
        return self.root.label

    def classify_query(self, query: str, entities: list, domains: list) -> Dict[str, Any]:
        features = {
            "query_length": len(query),
            "entity_count": len(entities),
            "domain_count": len(domains),
            "has_multi_domain": len(domains) > 1,
        }
        if features["has_multi_domain"]:
            decision = "graph_rag_slow_path"
        elif features["entity_count"] > 3:
            decision = "wfa_fast_path"
        else:
            decision = "fast_path"

        self.path_log.append({
            "query": query[:100],
            "decision": decision,
            "features": features,
        })
        return {
            "decision": decision,
            "confidence": 0.85,
            "path": decision,
            "features": features,
        }