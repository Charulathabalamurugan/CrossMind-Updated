import sys
import os

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reasoning.query_classifier import get_query_classifier
from reasoning.evaluation import RetrievalEvaluator
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline

def test_query_classifier():
    print("Testing Query Classifier...")
    # Test LightGBM
    classifier_lgb = get_query_classifier("LightGBM")
    q1 = "peptides and lipid nanoparticles targeting Alzheimer's in the brain"
    res1 = classifier_lgb.classify(q1)
    print(f"Query: {q1}")
    print(f"Classification: {res1}")
    assert res1["query_type"] == "cross_domain"
    
    # Test TinyBERT (MLP)
    classifier_tb = get_query_classifier("TinyBERT")
    q2 = "lithium battery anode degradation chemistry"
    res2 = classifier_tb.classify(q2)
    print(f"Query: {q2}")
    print(f"Classification: {res2}")
    assert res2["predicted_domain"] == "energy"
    print("[PASS] Query Classifier tests passed!\n")

def test_evaluation_metrics():
    print("Testing Evaluation Metrics...")
    # Precision@5, Recall@5, MRR, NDCG@5
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    ground_truth = {"doc2", "doc5", "doc7"}
    
    p = RetrievalEvaluator.precision_at_k(retrieved, ground_truth, k=5)
    r = RetrievalEvaluator.recall_at_k(retrieved, ground_truth, k=5)
    mrr = RetrievalEvaluator.mrr(retrieved, ground_truth)
    ndcg = RetrievalEvaluator.ndcg_at_k(retrieved, ground_truth, k=5)
    
    print(f"Precision@5: {p}")
    print(f"Recall@5: {r}")
    print(f"MRR: {mrr}")
    print(f"NDCG@5: {ndcg}")
    
    assert p == 0.4  # 2 out of 5
    assert r == 2/3  # 2 out of 3
    assert mrr == 0.5  # doc2 is at index 1 (rank 2) -> 1/2
    assert ndcg > 0.0
    print("[PASS] Evaluation Metrics tests passed!\n")

def test_pipeline_integration():
    print("Testing Pipeline Integration...")
    pipeline = get_neuro_symbolic_pipeline()
    
    # Test a simple query (should trigger optimized standard retrieval)
    print("Running simple query...")
    res_simple = pipeline.process_query("What are lithium-ion batteries?", user_role="admin")
    strategy_simple = res_simple["pre_filter"]["retrieval_strategy"]
    print(f"Simple Query Strategy: {strategy_simple}")
    assert "optimized" in strategy_simple.lower() or "standard" in strategy_simple.lower()
    
    # Test a complex cross-domain query
    print("Running complex query...")
    res_complex = pipeline.process_query("Peptides and lipid nanoparticles targeting Alzheimer's in the brain", user_role="admin")
    strategy_complex = res_complex["pre_filter"]["retrieval_strategy"]
    print(f"Complex Query Strategy: {strategy_complex}")
    print("[PASS] Pipeline Integration tests passed!\n")

if __name__ == "__main__":
    try:
        test_query_classifier()
        test_evaluation_metrics()
        test_pipeline_integration()
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    except AssertionError as ae:
        print(f"[FAIL] Assertion error during verification: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
