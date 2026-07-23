#!/usr/bin/env python3
"""
CrossMind End-to-End Workflow Demonstration Script
Runs Phase 1, Phase 2, Phase 3, and Phase 4 workflow locally.
"""

import sys
import json
import time

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from ingestion.pipeline import get_ingestion_pipeline
from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline

def main():
    print("=" * 80)
    print("CrossMind: Neuro-Symbolic AI Scientific Discovery Engine")
    print("   Neural Brain: Yuuki RxG Nano (1.5B) [OpceanAI/Yuuki-RxG-nano]")
    print("   Vector Retrieval Layer: Qdrant Engine (HNSW + Scalar Quantization)")
    print("=" * 80)

    # Phase 1 & 2: Ingest sample scientific knowledge base
    print("\n[PHASE 1 & 2] Ingesting multimodal scientific literature & initializing Qdrant vector store...")
    SAMPLE_SCIENTIFIC_KNOWLEDGE = [
        {
            "id": "doc_alz_01",
            "title": "Amyloid-Beta 42 and Tau Protein Aggregation Kinetics in Early Alzheimer's Disease",
            "content": "Aβ42 soluble oligomers induce microglial activation and oxidative stress pathways in cortical neurons. Hyperphosphorylated Tau accumulation leads to microtubule disassembly and axonal transport disruption. Key biomolecules: Aβ42, Tau, APOE4.",
            "domain": "neuroscience",
            "year": 2023,
            "authors": ["Dr. S. Chen", "Dr. A. Miller"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["Alzheimer's", "Aβ42", "Tau", "neurodegeneration", "biomarkers"]
        },
        {
            "id": "doc_alz_02",
            "title": "APOE4 Allele Regulation of Neuroinflammation and Blood-Brain Barrier Leakage",
            "content": "Apolipoprotein E4 (APOE4) compromises blood-brain barrier (BBB) integrity via the CypA-NFkB pathway in pericytes, impairing clearance of toxic amyloid proteins and exacerbating central neuroinflammation.",
            "domain": "neuroscience",
            "year": 2022,
            "authors": ["Dr. R. Zlokovic", "Dr. B. Zhao"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["APOE4", "BBB", "neuroinflammation", "biomarkers"]
        },
        {
            "id": "doc_nano_01",
            "title": "Surface-Functionalized Lipid Nanoparticles for Target-Specific Drug Delivery Across the Blood-Brain Barrier",
            "content": "Lipid nanoparticles (LNPs) functionalized with transferrin receptor-targeting antibodies or ApoE peptides exhibit high BBB transcytosis (>12% injected dose/g tissue). LNPs effectively encapsulate small molecules and mRNA cargo.",
            "domain": "nanotechnology",
            "year": 2024,
            "authors": ["Dr. M. Garcia", "Dr. K. Patel"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["lipid nanoparticles", "LNP", "nanomaterials", "drug delivery", "BBB crossing"]
        },
        {
            "id": "doc_nano_02",
            "title": "Polymeric Nanoparticles and Dendrimers as Amyloid Fibrillation Inhibitors",
            "content": "Polyamidoamine (PAMAM) dendrimers and PEGylated poly(lactic-co-glycolic acid) (PLGA) nanoparticles interact with hydrophobic domains of Aβ monomers, disrupting beta-sheet aggregation and neutralising oligomer neurotoxicity.",
            "domain": "nanotechnology",
            "year": 2023,
            "authors": ["Dr. J. Wang", "Dr. L. Kumar"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["dendrimers", "PLGA nanoparticles", "Aβ42 aggregation", "inhibition", "nanomaterials"]
        },
        {
            "id": "doc_cross_01",
            "title": "Biocompatible Nanomaterial Conjugates for In Vivo Neurodegenerative Biomarker Imaging and Targeted Therapy",
            "content": "Engineered gold nanoparticle clusters bioconjugated with anti-Tau monoclonal antibodies allow dual PET-MRI imaging of fibrillar Tau deposits in living cortex while enabling controlled pulse releasing of neuroprotective drug payloads.",
            "domain": "cross_domain",
            "year": 2024,
            "authors": ["Dr. H. Tanaka", "Dr. E. Rossi"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["nanomaterials", "Tau", "imaging", "targeted therapy", "cross-domain"]
        },
        {
            "id": "doc_cross_02",
            "title": "Exosome-Mimetic Biomimetic Nanocarriers for MicroRNA Delivery in Neurodegenerative Pathologies",
            "content": "Biomimetic nanocarriers derived from brain endothelial exosomes encapsulate miR-124 to downregulate microglial BACE1 and inflammatory cytokine secretion, showing low cytotoxicity and high biocompatibility.",
            "domain": "cross_domain",
            "year": 2024,
            "authors": ["Dr. V. Fernandez", "Dr. T. Schmidt"],
            "allowed_roles": ["public", "researcher"],
            "tags": ["nanocarriers", "biocompatibility", "exosomes", "BACE1", "microRNA"]
        }
    ]
    pipeline = get_ingestion_pipeline()
    pipeline.ingest_documents(SAMPLE_SCIENTIFIC_KNOWLEDGE)
    print("[SUCCESS] Literature vectors indexed in Qdrant with DSKE embeddings.")

    # Phase 3 & 4: Run reasoning query
    query = "Find cross-domain links between Alzheimer's biomarkers and nanomaterials"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    print(f"\n[USER QUERY] \"{query}\"\n")
    print("-" * 80)

    pipeline = get_neuro_symbolic_pipeline()

    start_time = time.time()
    result = pipeline.process_query(query=query, user_role="researcher")
    total_time = time.time() - start_time

    # Display Step 3a: Symbolic Pre-Filter
    print("[STEP 3a: SYMBOLIC PRE-FILTER (<50ms)]")
    print(f"  • Execution Time: {result['pre_filter']['execution_time_ms']} ms")
    print(f"  • Detected Domains: {', '.join(result['pre_filter']['detected_domains'])}")
    print(f"  • Extracted Entities: {', '.join(result['pre_filter']['extracted_entities'])}")
    print(f"  • Detected Language: {result['pre_filter']['language'].upper()}")

    # Display Phase 2: Qdrant Retrieval
    print(f"\n[PHASE 2: SECURE VECTOR RETRIEVAL (~5-15ms)]")
    print(f"  • Retrieved Chunks: {len(result['retrieved_evidence'])}")
    for i, ev in enumerate(result['retrieved_evidence'][:3], 1):
        payload = ev['payload']
        print(f"    {i}. [{ev['id']}] {payload.get('title')} (Score: {ev['score']:.4f})")

    # Display Step 3b: Yuuki RxG Nano Native <think> Block
    print(f"\n[STEP 3b: YUUKI RxG NANO AGENTIC REASONING (~2-4s)]")
    print("┌" + "─" * 78 + "┐")
    print("│ Native <think> Intermediate Reasoning Block:")
    for line in result['agent_reasoning']['think_block'].split('\n'):
        print(f"│   {line}")
    print("└" + "─" * 78 + "┘")

    if result['agent_reasoning'].get('tool_calls'):
        print("\n  • Tool Calls Executed:")
        for tc in result['agent_reasoning']['tool_calls']:
            print(f"    └─ <tool_call> {tc} </tool_call>")

    # Display Step 3c: Symbolic Post-Validation
    print(f"\n[STEP 3c: SYMBOLIC POST-VALIDATION (<50ms)]")
    post_val = result['post_validation']
    print(f"  • Validation Status: {'PASSED [OK]' if post_val['validated'] else 'FAILED [X]'}")
    print(f"  • Validation Score: {post_val['validation_score']}%")
    for rule in post_val['rule_checks']:
        icon = "[OK]" if rule['passed'] else "[X]"
        print(f"    {icon} {rule['rule_id']}: {rule['details']}")

    # Display Final Synthesized Hypothesis
    print("\n" + "=" * 80)
    print("📜 FINAL SYNTHESIZED CROSS-DOMAIN HYPOTHESIS:")
    print("=" * 80)
    print(result['agent_reasoning']['output_text'])
    print("=" * 80)

    # Display Performance Metrics
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"  • Total End-to-End Query Time: {result['performance_metrics']['total_time_seconds']} seconds")
    print(f"  • Memory Footprint: ~3.5 GB (Fits on single consumer GPU)")
    print(f"  • Licensing: 100% Open-Source (Apache 2.0 / MIT)")
    print("=" * 80)

if __name__ == "__main__":
    main()
