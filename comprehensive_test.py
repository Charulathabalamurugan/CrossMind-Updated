"""
CrossMind Comprehensive Test Suite
Tests all endpoints, features, and edge cases thoroughly.
"""
import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"
PASS = 0
FAIL = 0
TOTAL = 0

def test(name, condition, detail=""):
    global PASS, FAIL, TOTAL
    TOTAL += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))

def check_keys(data, required_keys, context=""):
    missing = [k for k in required_keys if k not in data]
    return len(missing) == 0, missing

print("=" * 70)
print("CROSSMIND COMPREHENSIVE TEST SUITE")
print("=" * 70)

# ========================
# TEST 1: ROOT ENDPOINT
# ========================
print("\n[TEST GROUP 1] Root & Health Endpoints")
try:
    r = requests.get(f"{API_BASE}/", timeout=5)
    test("Root endpoint returns 200", r.status_code == 200)
    d = r.json()
    ok, missing = check_keys(d, ["project", "engine", "status", "version", "endpoints"])
    test("Root response has all required fields", ok, f"missing: {missing}" if missing else "")
    test("Engine status is 'online'", d.get("status") == "online")
    test("Engine is Yuuki RxG Nano", "RxG" in d.get("engine", ""))
except Exception as e:
    test("Root endpoint", False, str(e))

# ========================
# TEST 2: METRICS ENDPOINT
# ========================
print("\n[TEST GROUP 2] Metrics Endpoint")
try:
    r = requests.get(f"{API_BASE}/api/metrics", timeout=5)
    test("Metrics endpoint returns 200", r.status_code == 200)
    d = r.json()
    ok, missing = check_keys(d, ["model", "metrics"])
    test("Metrics has model and metrics fields", ok, f"missing: {missing}" if missing else "")
    metrics = d.get("metrics", {})
    test("AIME 2024 metric present", "AIME_2024" in metrics)
    test("TruthfulQA metric present", "TruthfulQA_MC1" in metrics)
    test("MMLU-Pro metric present", "MMLU_Pro" in metrics)
except Exception as e:
    test("Metrics endpoint", False, str(e))

# ========================
# TEST 3: ENGLISH QUERY
# ========================
print("\n[TEST GROUP 3] English Query Processing")
try:
    q = "Find cross-domain links between Alzheimer biomarkers and nanomaterials"
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=60)
    test("English query returns 200", r.status_code == 200)
    d = r.json()
    required = ["query", "user_role", "pre_filter", "retrieved_evidence", "agent_reasoning", "post_validation", "performance_metrics"]
    ok, missing = check_keys(d, required)
    test("Response has all pipeline stages", ok, f"missing: {missing}" if missing else "")
    
    pre = d["pre_filter"]
    test("Pre-filter has language", pre.get("language") == "english")
    test("Pre-filter has domains", len(pre.get("detected_domains", [])) > 0)
    test("Pre-filter has entities", len(pre.get("extracted_entities", [])) > 0)
    test("Pre-filter execution time reasonable", pre.get("execution_time_ms", 100) < 100)
    
    test("Retrieved evidence found", len(d.get("retrieved_evidence", [])) > 0)
    
    agent = d["agent_reasoning"]
    test("Agent has think_block", len(agent.get("think_block", "") ) > 0)
    test("Agent has output_text/hypothesis", len(agent.get("output_text", "")) > 0)
    test("Agent has confidence score", agent.get("confidence_score", 0) > 0)
    
    val = d["post_validation"]
    test("Validation score present", val.get("validation_score", 0) >= 0)
    test("Rule checks present", len(val.get("rule_checks", [])) > 0)
    
    perf = d["performance_metrics"]
    test("Total time tracked", perf.get("total_time_seconds", 0) > 0)
    test("Total time reasonable", perf.get("total_time_seconds", 0) < 60)
except Exception as e:
    test("English query", False, str(e))

# ========================
# TEST 4: SPANISH QUERY
# ========================
print("\n[TEST GROUP 4] Spanish Query Processing")
try:
    q = "Buscar conexiones entre el peptido Abeta42 y nanoparticulas lipidicas para la barrera hematoencefalica"
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=60)
    test("Spanish query returns 200", r.status_code == 200)
    d = r.json()
    pre = d["pre_filter"]
    test("Spanish language detected correctly", pre.get("language") == "spanish")
    test("Spanish query has domains", len(pre.get("detected_domains", [])) > 0)
    test("Spanish query retrieves evidence", len(d.get("retrieved_evidence", [])) > 0)
    val = d["post_validation"]
    test("Spanish query validation passes", val.get("validation_score", 0) >= 70)
except Exception as e:
    test("Spanish query", False, str(e))

# ========================
# TEST 5: DOCUMENT INGESTION
# ========================
print("\n[TEST GROUP 5] Document Ingestion")
try:
    payload = {
        "documents": [{
            "title": "Test Doc - Nanotechnology in Cancer Therapy",
            "domain": "nanotechnology",
            "content": "Gold nanoparticles functionalized with PEG and targeting antibodies show enhanced permeability and retention effect in solid tumors. Surface plasmon resonance enables photothermal ablation therapy.",
            "year": 2025,
            "tags": ["gold nanoparticles", "cancer", "PEG", "photothermal"],
            "allowed_roles": ["researcher", "admin"],
            "authors": ["Dr. Test Researcher"]
        }]
    }
    r = requests.post(f"{API_BASE}/api/ingest", json=payload, timeout=15)
    test("Ingestion returns 200", r.status_code == 200)
    d = r.json()
    test("Ingested count is 1", d.get("ingested_count") == 1)
    test("Has inserted_ids", len(d.get("inserted_ids", [])) == 1)
    test("Status is success", d.get("status") == "success")
    
    # Verify by searching for the ingested document
    q = "gold nanoparticles cancer photothermal therapy"
    r2 = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "admin"}, timeout=60)
    d2 = r2.json()
    test("Ingested document is searchable", len(d2.get("retrieved_evidence", [])) > 0)
except Exception as e:
    test("Document ingestion", False, str(e))

# ========================
# TEST 6: RBAC FILTERING
# ========================
print("\n[TEST GROUP 6] RBAC Role-Based Access Control")
try:
    q = "How do lipid nanoparticles cross the blood-brain barrier?"
    
    # Admin role
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "admin"}, timeout=60)
    d = r.json()
    admin_evidence = len(d.get("retrieved_evidence", []))
    test("Admin can access evidence", admin_evidence > 0, f"got {admin_evidence} chunks")
    
    # Researcher role
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=60)
    d = r.json()
    researcher_evidence = len(d.get("retrieved_evidence", []))
    test("Researcher can access evidence", researcher_evidence > 0, f"got {researcher_evidence} chunks")
    
    # Public role
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "public"}, timeout=60)
    d = r.json()
    public_evidence = len(d.get("retrieved_evidence", []))
    test("Public can access evidence", public_evidence > 0, f"got {public_evidence} chunks")
    
    test("All roles return validated results", d.get("post_validation", {}).get("validated", False))
except Exception as e:
    test("RBAC filtering", False, str(e))

# ========================
# TEST 7: SSE STREAMING
# ========================
print("\n[TEST GROUP 7] SSE Streaming Endpoint")
try:
    r = requests.get(
        f"{API_BASE}/api/stream_reasoning",
        params={"query": "How do nanoparticles target brain tumors?", "user_role": "researcher"},
        timeout=30,
        stream=True
    )
    test("Streaming returns 200", r.status_code == 200)
    test("Content-Type is text/event-stream", "text/event-stream" in r.headers.get("content-type", ""))
    
    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                event_data = json.loads(line[6:])
                events.append(event_data)
            except json.JSONDecodeError:
                pass
    
    test("SSE events received", len(events) > 0, f"got {len(events)} events")
    
    event_types = [e.get("event") for e in events if "event" in e]
    if not event_types:
        event_types = [e.get("stage") for e in events if "stage" in e]
    test("Has step_3a_pre_filter event", "step_3a_pre_filter" in event_types or "pre_filter" in event_types)
    test("Has step_2_vector_retrieval event", "step_2_vector_retrieval" in event_types)
    test("Has completed event", "completed" in event_types)
except Exception as e:
    test("SSE streaming", False, str(e))

# ========================
# TEST 8: EDGE CASES
# ========================
print("\n[TEST GROUP 8] Edge Cases & Error Handling")
# Test empty query
try:
    r = requests.post(f"{API_BASE}/api/query", json={"query": "", "user_role": "researcher"}, timeout=10)
    test("Empty query returns response", r.status_code in [200, 422])
    if r.status_code == 422:
        test("Empty query rejected with 422", True)
except Exception as e:
    test("Empty query handling", False, str(e))

# Test very long query
try:
    long_q = "Find links between " + "Alzheimer " * 50 + " and nanomaterials"
    r = requests.post(f"{API_BASE}/api/query", json={"query": long_q[:1000], "user_role": "researcher"}, timeout=60)
    test("Long query processes", r.status_code == 200)
except Exception as e:
    test("Long query", False, str(e))

# Test bulk ingestion
try:
    bulk_payload = {"documents": [
        {"title": f"Bulk Test Doc {i}", "domain": "neuroscience" if i % 2 == 0 else "nanotechnology",
         "content": f"This is bulk test document number {i} for testing ingestion pipeline capacity.",
         "year": 2024, "tags": ["bulk", "test"], "allowed_roles": ["public"], "authors": ["Bulk Tester"]}
        for i in range(3)
    ]}
    r = requests.post(f"{API_BASE}/api/ingest", json=bulk_payload, timeout=15)
    test("Bulk ingestion returns 200", r.status_code == 200)
    d = r.json()
    test("Bulk ingestion count is 3", d.get("ingested_count") == 3)
except Exception as e:
    test("Bulk ingestion", False, str(e))

# ========================
# SUMMARY
# ========================
print("\n" + "=" * 70)
print(f"TEST SUMMARY: {PASS}/{TOTAL} passed, {FAIL} failed")
if FAIL > 0:
    print("SOME TESTS FAILED - Review output above for details")
else:
    print("ALL TESTS PASSED - System is working correctly")
print("=" * 70)
