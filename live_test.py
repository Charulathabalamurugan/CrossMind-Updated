"""Live test demonstrating the CrossMind pipeline end-to-end."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import get_ingestion_pipeline
from vector_store.qdrant_engine import get_qdrant_engine

# 1. Ingest sample documents
print("=" * 60)
print("STEP 1: Ingest documents")
print("=" * 60)
pipeline = get_ingestion_pipeline()
docs = [
    {
        "id": "doc1",
        "title": "Lithium-Ion Battery Anode Degradation",
        "domain": "energy",
        "year": 2024,
        "authors": ["Smith et al."],
        "content": "Abstract: We study lithium-ion battery anode degradation. Introduction: Graphite anodes degrade over many charge cycles. Methods: Electrochemical impedance spectroscopy and XRD. Results: SEI layer growth increases resistance. Conclusions: Silicon-based anodes offer improved cycle life.",
        "tags": ["battery", "lithium", "anode", "energy"],
    },
    {
        "id": "doc2",
        "title": "Solid-State Battery Electrolytes",
        "domain": "energy",
        "year": 2025,
        "authors": ["Chen et al."],
        "content": "Abstract: Solid-state electrolytes improve lithium battery safety. Methods: Sulfide-based electrolytes tested at room temperature. Results: Ionic conductivity reaches 10 mS/cm. Conclusions: Solid-state batteries are commercially viable by 2027.",
        "tags": ["battery", "solid-state", "energy"],
    },
    {
        "id": "doc3",
        "title": "Sodium-Ion Battery Alternatives",
        "domain": "energy",
        "year": 2023,
        "authors": ["Park et al."],
        "content": "Abstract: Sodium-ion batteries offer a cheaper alternative to lithium-ion. Results: Energy density of 160 Wh/kg. Conclusions: Suitable for grid storage applications.",
        "tags": ["battery", "sodium", "energy"],
    },
    {
        "id": "doc4",
        "title": "General Materials Science Survey",
        "domain": "general",
        "year": 2024,
        "authors": ["Doe et al."],
        "content": "Abstract: A general review of materials science covering metals, ceramics, polymers, and composites for engineering applications.",
        "tags": ["materials", "general"],
    },
]
ids = pipeline.ingest_documents(docs)
print(f"Ingested chunk IDs: {ids}")

# 2. Check the in-memory store
print()
print("=" * 60)
print("STEP 2: Inspect vector store state")
print("=" * 60)
engine = get_qdrant_engine()
print(f"Qdrant client available: {engine.client is not None}")
print(f"In-memory store size: {len(engine._memory_store)}")
print(f"BM25 indexed docs: {engine.bm25.doc_count}")

# 3. BM25 search
print()
print("=" * 60)
print("STEP 3: BM25 keyword search")
print("=" * 60)
results = engine.bm25.search("lithium battery anode", user_role="admin")
print(f"BM25 results: {len(results)}")
for r in results:
    title = r["payload"].get("title", "?")
    score = r["score"]
    print(f"  {r['id']}: {title} (score={score:.3f})")

# 4. Run the full pipeline
print()
print("=" * 60)
print("STEP 4: Full neuro-symbolic pipeline")
print("=" * 60)
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline
nsp = get_neuro_symbolic_pipeline()
res = nsp.process_query(
    "How does silicon improve lithium battery anodes?",
    user_role="admin",
    session_id="demo"
)

print(f"Strategy: {res['pre_filter'].get('retrieval_strategy')}")
print(f"Model: {res['pre_filter'].get('reasoning_model')}")
print(f"Execution mode: {res['pre_filter'].get('execution_mode')}")
print(f"Detected domains: {res['pre_filter'].get('detected_domains')}")
print(f"Extracted entities: {res['pre_filter'].get('extracted_entities')}")
print(f"Evidence count: {res['performance_metrics']['retrieved_chunks_count']}")
print(f"Graph nodes: {res['performance_metrics']['graph_nodes_count']}")
print(f"Multi-hop paths: {res['performance_metrics']['multi_hop_paths_count']}")
print(f"Calibrated confidence: {res['confidence_calibration']['calibrated_confidence']:.3f}")
print(f"Decision: {res['confidence_calibration']['decision']}")
print(f"Quality gate: {res['quality_gate']['decision']}")
print(f"Total time: {res['performance_metrics']['total_time_seconds']}s")
print()
print("RETRIEVED EVIDENCE TITLES:")
for ev in res.get("retrieved_evidence", []):
    print(f"  - {ev.get('payload', {}).get('title', '?')}")
print()
print("Z3 VALIDATION:", res.get("z3_formal_validation", {}).get("validation_score", "N/A"))
print("VALIDATION RULES:")
for rc in res.get("post_validation", {}).get("rule_checks", []):
    print(f"  - {rc.get('rule_id')}: {'PASS' if rc.get('passed') else 'FAIL'}")
print()
print("HYPOTHESIS:")
print("-" * 60)
print(res["agent_reasoning"]["output_text"])
