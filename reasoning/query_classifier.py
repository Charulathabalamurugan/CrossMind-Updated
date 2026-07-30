import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger("crossmind.query_classifier")

# Predefined training dataset for scientific query classification
TRAINING_DATA = [
    # Neuroscience
    ("Alzheimer's disease amyloid plaque formation and synaptic loss", "neuroscience", "complex", "high"),
    ("How does dopamine regulate motor function in Parkinson's disease?", "neuroscience", "causal", "high"),
    ("Synaptic plasticity and long term potentiation in hippocampus", "neuroscience", "factual", "low"),
    ("What neurotransmitters are involved in sleep regulation?", "neuroscience", "factual", "low"),
    ("Neurodegeneration mediated by microglia activation", "neuroscience", "causal", "medium"),

    # Nanotechnology
    ("Synthesis of lipid nanoparticles for mRNA delivery", "nanotechnology", "complex", "medium"),
    ("Gold nanoparticle PEG functionalization surface chemistry", "nanotechnology", "factual", "low"),
    ("How do carbon nanotubes improve conductivity in polymer composites?", "nanotechnology", "causal", "high"),
    ("Characterization of polymeric micelles using dynamic light scattering", "nanotechnology", "factual", "low"),
    ("Nanoparticle cellular uptake via receptor-mediated endocytosis", "nanotechnology", "complex", "high"),

    # Pharmacology
    ("Pharmacokinetics and bioavailability of oral drug formulations", "pharmacology", "factual", "low"),
    ("How does first pass metabolism affect drug bioavailability?", "pharmacology", "causal", "medium"),
    ("Toxicity profile of novel small molecule kinase inhibitors", "pharmacology", "complex", "high"),
    ("Cytochrome P450 enzyme inhibition and drug drug interactions", "pharmacology", "complex", "high"),
    ("In vitro IC50 measurement for oncology drug candidates", "pharmacology", "factual", "low"),

    # Energy
    ("Lithium-ion battery anode degradation and solid electrolyte interphase", "energy", "complex", "high"),
    ("Efficiency optimization in perovskite solar cells", "energy", "factual", "low"),
    ("How does electrolyte composition affect supercapacitor cycle life?", "energy", "causal", "high"),
    ("Catalytic activity of transition metal oxides in fuel cells", "energy", "factual", "medium"),
    ("Solid-state battery electrolyte conductivity at room temperature", "energy", "complex", "high"),

    # Financial
    ("Portfolio risk optimization using black-scholes options pricing", "financial", "complex", "high"),
    ("Stock market volatility forecasting using GARCH models", "financial", "factual", "low"),
    ("How does central bank interest rate policy affect bond yields?", "financial", "causal", "medium"),
    ("Algorithmic trading strategies using momentum indicators", "financial", "factual", "low"),
    ("Credit default swap pricing and liquidity risk modeling", "financial", "complex", "high"),

    # Cross-Domain
    ("Nanoparticle drug delivery systems for targeting Alzheimer's in the brain", "neuroscience", "cross_domain", "high"),
    ("Using energy storage investments to hedge stock market volatility", "energy", "cross_domain", "high"),
    ("Toxicology of nanocarriers used in central nervous system pharmacotherapy", "pharmacology", "cross_domain", "high"),
    ("Financial viability of hydrogen fuel cell integration in national power grids", "energy", "cross_domain", "high"),
]

class QueryClassifier:
    """
    Query Classifier classifying queries into scientific domains, types, and complexity levels.
    Employs LightGBM (via sklearn GradientBoostingClassifier fallback) and TinyBERT (via sklearn MLPClassifier/Sentence-Transformers).
    """
    def __init__(self, model_type: str = "LightGBM"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        
        # Prepare labels
        self.texts = [item[0] for item in TRAINING_DATA]
        self.domains = [item[1] for item in TRAINING_DATA]
        self.types = [item[2] for item in TRAINING_DATA]
        self.complexities = [item[3] for item in TRAINING_DATA]
        
        # Train vectorized features
        self.X = self.vectorizer.fit_transform(self.texts).toarray()
        
        # We will use either GradientBoostingClassifier (LightGBM representation)
        # or MLPClassifier (TinyBERT representation)
        if self.model_type == "LightGBM":
            self.domain_classifier = GradientBoostingClassifier(n_estimators=30, random_state=42)
            self.type_classifier = GradientBoostingClassifier(n_estimators=30, random_state=42)
            self.complexity_classifier = GradientBoostingClassifier(n_estimators=30, random_state=42)
        else: # TinyBERT / Neural Representation
            self.domain_classifier = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
            self.type_classifier = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
            self.complexity_classifier = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
            
        self._fit()

    def _fit(self):
        try:
            self.domain_classifier.fit(self.X, self.domains)
            self.type_classifier.fit(self.X, self.types)
            self.complexity_classifier.fit(self.X, self.complexities)
            logger.info(f"QueryClassifier fitted successfully using model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error training QueryClassifier: {e}")

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies the incoming query text.
        """
        if not query.strip():
            return {
                "predicted_domain": "general",
                "query_type": "factual",
                "complexity": "low",
                "confidence": 1.0,
                "model_used": self.model_type
            }
            
        x_query = self.vectorizer.transform([query]).toarray()
        
        # Predictions
        domain_pred = self.domain_classifier.predict(x_query)[0]
        type_pred = self.type_classifier.predict(x_query)[0]
        complexity_pred = self.complexity_classifier.predict(x_query)[0]
        
        # Probabilities for confidence
        domain_probs = self.domain_classifier.predict_proba(x_query)[0]
        type_probs = self.type_classifier.predict_proba(x_query)[0]
        
        # Compute avg confidence
        confidence = float(np.mean([np.max(domain_probs), np.max(type_probs)]))
        
        # Rule overrides: if query mentions keywords from a specific domain, prioritize that
        lower_q = query.lower()
        matched_domains = []
        domain_keywords = {
            "neuroscience": ["alzheimer", "brain", "synap", "neuron", "neuro", "parkinson", "cortex", "hippocampus"],
            "nanotechnology": ["nanoparticle", "nanomaterial", "nanocarrier", "lnp", "micelle", "nanotube"],
            "pharmacology": ["pharmacokinetic", "bioavailability", "drug", "toxic", "ic50"],
            "energy": ["battery", "batteries", "solar", "anode", "cathode", "electrolyte", "fuel cell"],
            "financial": ["finance", "market", "portfolio", "stock", "bond", "yield", "volatility"]
        }
        for dom, keywords in domain_keywords.items():
            if any(kw in lower_q for kw in keywords):
                matched_domains.append(dom)
                
        # Convert classification predictions to standard python string types
        domain_pred = str(domain_pred)
        type_pred = str(type_pred)
        complexity_pred = str(complexity_pred)

        if len(matched_domains) == 1:
            domain_pred = matched_domains[0]
        elif len(matched_domains) > 1:
            type_pred = "cross_domain"
            complexity_pred = "high"
            if domain_pred not in matched_domains:
                domain_pred = matched_domains[0]
                
        return {
            "predicted_domain": str(domain_pred),
            "query_type": str(type_pred),
            "complexity": str(complexity_pred),
            "confidence": round(confidence, 3),
            "model_used": self.model_type
        }

_classifier_instance = None

def get_query_classifier(model_type: str = "LightGBM") -> QueryClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier(model_type=model_type)
    return _classifier_instance
