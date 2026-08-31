import sys
import os

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import IngestionPipeline
from reasoning.conflict_detector import ConflictDetector
from reasoning.rxg_nano_agent import ZAYA1_8BAgent

def test_quality_scoring():
    print("Testing Quality Ingestion Scoring...")
    pipeline = IngestionPipeline()
    # Simple document
    doc_simple = {
        "title": "Short Doc",
        "content": "This is a short sample paper on nanotechnology.",
        "domain": "nanotechnology"
    }
    chunks = pipeline._chunk_and_embed(doc_simple)
    assert len(chunks) > 0
    q1 = chunks[0]["payload"]["quality_score"]
    print(f"Simple doc quality: {q1}")
    
    # Rich structured document
    doc_rich = {
        "title": "Extended Biomarker Research",
        "authors": ["Dr. Smith"],
        "year": 2026,
        "content": "Abstract: We study Alzheimer's disease pathology in hippocampus. Introduction: Amyloid plaques build up in neurons. Methods: Cellular imaging and Western blot. Results: Synaptic density decreases with plaque counts. Discussion: This reveals a key pathway. Conclusions: Suppressing amyloid slows neurodegeneration. References: Dr. Smith et al. 2026.",
        "domain": "neuroscience"
    }
    chunks_rich = pipeline._chunk_and_embed(doc_rich)
    assert len(chunks_rich) > 0
    q2 = chunks_rich[0]["payload"]["quality_score"]
    print(f"Rich doc quality: {q2}")
    assert q2 > q1
    print("[PASS] Quality Scoring tests passed!\n")

def test_conflict_detection():
    print("Testing Conflict Detection...")
    evidence = [
        {
            "id": "doc_a",
            "payload": {
                "title": "Nanoparticle Activation Study",
                "content": "Nanoparticles activate immune cells and stimulate cytokine secretion, improving immune response.",
                "domain": "nanotechnology"
            }
        },
        {
            "id": "doc_b",
            "payload": {
                "title": "Nanoparticle Toxicity Study",
                "content": "Nanoparticles inhibit immune cells and suppress cytokine pathways, inducing severe toxicity.",
                "domain": "nanotechnology"
            }
        }
    ]
    conflicts = ConflictDetector.detect_conflicts(evidence)
    print(f"Detected Conflicts: {conflicts}")
    assert len(conflicts) > 0
    assert conflicts[0]["source_id_1"] == "doc_a"
    assert conflicts[0]["source_id_2"] == "doc_b"
    print("[PASS] Conflict Detection tests passed!\n")

def test_zaya_prompt_conflict():
    print("Testing Conflict integration in ZAYA Prompt...")
    agent = ZAYA1_8BAgent()
    evidence = [
        {
            "id": "doc_a",
            "payload": {
                "title": "Study A",
                "content": "Amyloid plaques increase synaptic degeneration.",
                "domain": "neuroscience"
            }
        },
        {
            "id": "doc_b",
            "payload": {
                "title": "Study B",
                "content": "Amyloid plaques decrease synaptic degeneration in mutant strains.",
                "domain": "neuroscience"
            }
        }
    ]
    prompt = agent._build_prompt(
        query="Amyloid synaptic links",
        evidence=evidence,
        filter_meta={"language": "english", "extracted_entities": ["amyloid"]}
    )
    print(f"ZAYA Prompt:\n{prompt}")
    assert "CRITICAL CONFLICTS DETECTED" in prompt
    assert "doc_a" in prompt
    assert "doc_b" in prompt
    print("[PASS] ZAYA Prompt Conflict integration passed!\n")

if __name__ == "__main__":
    try:
        test_quality_scoring()
        test_conflict_detection()
        test_zaya_prompt_conflict()
        print("ALL NEW FEATURES VERIFIED SUCCESSFULLY!")
        sys.exit(0)
    except AssertionError as ae:
        print(f"[FAIL] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
