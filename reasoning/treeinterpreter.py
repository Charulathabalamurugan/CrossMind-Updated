import logging
from typing import Dict, Any, List
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger("crossmind.treeinterpreter")

class TreeInterpreter:
    def __init__(self):
        self.enabled = True
        self.model: DecisionTreeClassifier = None
        self.vectorizer: TfidfVectorizer = None
        self.feature_names: List[str] = []
        self.trained = False

    def train(self, X: List[str], y: List[int], feature_names: List[str] = None):
        if not X or not y:
            logger.warning("Empty training data for TreeInterpreter.")
            return
        self.vectorizer = TfidfVectorizer(max_features=100)
        X_vec = self.vectorizer.fit_transform(X)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_vec.shape[1])]

        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.model.fit(X_vec.toarray(), y)
        self.trained = True
        logger.info("TreeInterpreter model trained.")

    def explain(self, text: str, prediction: float, evidence: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.trained or not self.model:
            return {"explanation": "Model not trained.", "feature_importance": {}}

        try:
            X_vec = self.vectorizer.transform([text])
            features = X_vec.toarray()[0]
            importance = {}
            for i, name in enumerate(self.feature_names):
                if features[i] > 0 and i < len(self.model.feature_importances_):
                    importance[name] = round(float(self.model.feature_importances_[i]), 4)

            tree_rules = export_text(self.model, feature_names=self.feature_names[:20])
            return {
                "prediction": prediction,
                "feature_importance": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]),
                "decision_tree_rules": tree_rules[:2000],
            }
        except Exception as exc:
            logger.error(f"TreeInterpreter explain failed: {exc}")
            return {"explanation": f"Error: {str(exc)}", "feature_importance": {}}

    def get_model_summary(self) -> Dict[str, Any]:
        if not self.trained or not self.model:
            return {"trained": False}
        return {
            "trained": True,
            "tree_depth": self.model.get_depth(),
            "leaf_count": self.model.get_n_leaves(),
            "feature_importance": {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(self.feature_names, self.model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            },
        }

tree_interpreter_instance = None

def get_tree_interpreter() -> TreeInterpreter:
    global tree_interpreter_instance
    if tree_interpreter_instance is None:
        tree_interpreter_instance = TreeInterpreter()
    return tree_interpreter_instance
