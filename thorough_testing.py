"""
CrossMind Thorough Testing Suite
Covers: CORS headers, response times, concurrent requests, error paths, Docker config, full UI simulation
"""
import requests
import json
import time
import asyncio
import sys

API_BASE = "http://localhost:8000"
PASS = 0
FAIL = 0
TOTAL = 0
FAILURES = []

def test(name, condition, detail=""):
    global PASS, FAIL, TOTAL, FAILURES
    TOTAL += 1
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} - {detail}")

def section(title):
    print(f"\n{'='*70}")
    print(f"[SECTION] {title}")
    print(f"{'='*70}")

# =============================================
# SECTION 1: CORS HEADERS VALIDATION
# =============================================
section("CORS Headers Validation")

try:
    r = requests.get(f"{API_BASE}/", timeout=5, headers={"Origin": "http://localhost:8501"})
    headers = r.headers
    test("Access-Control-Allow-Origin matches origin", 
         headers.get("Access-Control-Allow-Origin") == "http://localhost:8501",
         f"got: {headers.get('Access-Control-Allow-Origin')}")
    test("Access-Control-Allow-Methods exists", 
         "Access-Control-Allow-Methods" in headers,
         f"got: {headers.get('Access-Control-Allow-Methods')}")
    test("Access-Control-Allow-Headers exists", 
         "Access-Control-Allow-Headers" in headers,
         f"got: {headers.get('Access-Control-Allow-Headers')}")
except Exception as e:
    test("CORS headers with Origin", False, str(e))

# Test OPTIONS preflight
try:
    r = requests.options(f"{API_BASE}/", timeout=5, headers={"Origin": "http://localhost:8501"})
    test("OPTIONS preflight returns 200", r.status_code == 200, f"got: {r.status_code}")
    test("OPTIONS has CORS origin", 
         r.headers.get("Access-Control-Allow-Origin") is not None,
         f"got: {r.headers.get('Access-Control-Allow-Origin')}")
except Exception as e:
    test("OPTIONS preflight", False, str(e))

# =============================================
# SECTION 2: API RESPONSE TIME BENCHMARKS
# =============================================
section("API Response Time Benchmarks")

# Metrics endpoint (should be instant)
try:
    times = []
    for i in range(5):
        start = time.time()
        r = requests.get(f"{API_BASE}/api/metrics", timeout=5)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    avg = sum(times) / len(times)
    test("Metrics endpoint avg < 300ms", avg < 300, f"avg: {avg:.1f}ms")
    test("Metrics endpoint max < 500ms", max(times) < 500, f"max: {max(times):.1f}ms")
except Exception as e:
    test("Metrics response time", False, str(e))

# Root endpoint
try:
    times = []
    for i in range(5):
        start = time.time()
        r = requests.get(f"{API_BASE}/", timeout=5)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    avg = sum(times) / len(times)
    test("Root endpoint avg < 100ms", avg < 100, f"avg: {avg:.1f}ms")
except Exception as e:
    test("Root response time", False, str(e))

# Full query endpoint
try:
    q = "Alzheimer biomarkers nanomaterials"
    start = time.time()
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=60)
    elapsed = time.time() - start
    test("Full query completes < 30s", elapsed < 30, f"took: {elapsed:.1f}s")
    test("Full query returns 200", r.status_code == 200)
except Exception as e:
    test("Full query response time", False, str(e))

# =============================================
# SECTION 3: CONCURRENT REQUEST HANDLING
# =============================================
section("Concurrent Request Handling")

try:
    import threading
    results = []
    errors = []
    
    def make_request(idx):
        try:
            q = f"Test query number {idx} about nanoparticles and drug delivery"
            r = requests.post(f"{API_BASE}/api/query", 
                            json={"query": q, "user_role": "researcher"}, 
                            timeout=30)
            results.append((idx, r.status_code, len(r.json().get("retrieved_evidence", []))))
        except Exception as e:
            errors.append((idx, str(e)))
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=make_request, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    test("All concurrent requests completed", len(results) == 3, f"completed: {len(results)}, errors: {len(errors)}")
    test("All concurrent requests returned 200", all(r[1] == 200 for r in results))
    test("All concurrent requests returned evidence", all(r[2] > 0 for r in results))
    if errors:
        for idx, err in errors:
            print(f"  [INFO] Request {idx} error: {err}")
except Exception as e:
    test("Concurrent requests", False, str(e))

# =============================================
# SECTION 4: ERROR PATH TESTING
# =============================================
section("Error Path Testing")

# Test 1: Invalid JSON body
try:
    r = requests.post(f"{API_BASE}/api/query", data="not-json-at-all", 
                      headers={"Content-Type": "application/json"}, timeout=5)
    test("Invalid JSON returns 422", r.status_code == 422)
except Exception as e:
    test("Invalid JSON handling", False, str(e))

# Test 2: Missing required fields
try:
    r = requests.post(f"{API_BASE}/api/query", json={"user_role": "researcher"}, timeout=5)
    test("Missing query field returns 422", r.status_code == 422)
except Exception as e:
    test("Missing query field", False, str(e))

# Test 3: Empty documents array
try:
    r = requests.post(f"{API_BASE}/api/ingest", json={"documents": []}, timeout=5)
    test("Empty documents returns 200 with count 0", r.status_code == 200)
    if r.status_code == 200:
        test("Empty documents count is 0", r.json().get("ingested_count") == 0)
except Exception as e:
    test("Empty documents", False, str(e))

# Test 4: Invalid role edge case
try:
    q = "test"
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "invalid_role"}, timeout=10)
    test("Invalid role returns 422 or 200", r.status_code in [200, 422],
         f"got: {r.status_code}")
except Exception as e:
    test("Invalid role handling", False, str(e))

# Test 5: Special characters in query
try:
    q = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./"
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=30)
    test("Special characters query returns 200", r.status_code == 200)
except Exception as e:
    test("Special characters query", False, str(e))

# Test 6: Unicode / Emoji in query
try:
    q = "Alzheimer's disease 🧠 biomarkers and nanoparticles 💊 cross-domain analysis"
    r = requests.post(f"{API_BASE}/api/query", json={"query": q, "user_role": "researcher"}, timeout=30)
    test("Unicode/emoji query returns 200", r.status_code == 200)
except Exception as e:
    test("Unicode/emoji query", False, str(e))

# Test 7: Method not allowed
try:
    r = requests.put(f"{API_BASE}/api/query", json={"query": "test", "user_role": "researcher"}, timeout=5)
    test("PUT to query returns 405", r.status_code == 405)
except Exception as e:
    test("Method not allowed", False, str(e))

# Test 8: Missing content-type header
try:
    r = requests.post(f"{API_BASE}/api/ingest", data="raw string", timeout=5)
    test("Missing content-type returns 422", r.status_code in [400, 415, 422])
except Exception as e:
    test("Missing content-type", False, str(e))

# =============================================
# SECTION 5: STREAMLIT DASHBOARD API DEPENDENCIES
# =============================================
section("Streamlit Dashboard API Dependencies")

# Verify all API endpoints used by dashboard/app.py
try:
    # Dashboard tabs: Cross-Domain Reasoning, Benchmark, Document Ingestion
    
    # Tab 1: Query endpoint (POST /api/query)
    r = requests.post(f"{API_BASE}/api/query", 
                      json={"query": "test", "user_role": "researcher"}, timeout=30)
    test("Dashboard Tab1 query endpoint works", r.status_code == 200)
    d = r.json()
    test("Dashboard Tab1 requires: pre_filter.detected_domains", 
         len(d.get("pre_filter", {}).get("detected_domains", [])) > 0)
    test("Dashboard Tab1 requires: retrieved_evidence", 
         len(d.get("retrieved_evidence", [])) > 0)
    test("Dashboard Tab1 requires: agent_reasoning.think_block",
         len(d.get("agent_reasoning", {}).get("think_block", "")) > 0)
    test("Dashboard Tab1 requires: agent_reasoning.output_text",
         len(d.get("agent_reasoning", {}).get("output_text", "")) > 0)
    test("Dashboard Tab1 requires: post_validation.validation_score",
         d.get("post_validation", {}).get("validation_score", -1) >= 0)
    test("Dashboard Tab1 requires: performance_metrics",
         "performance_metrics" in d)
except Exception as e:
    test("Dashboard Tab1 dependencies", False, str(e))

try:
    # Tab 2: Metrics endpoint (GET /api/metrics) - for benchmark charts
    r = requests.get(f"{API_BASE}/api/metrics", timeout=5)
    test("Dashboard Tab2 metrics endpoint works", r.status_code == 200)
    d = r.json()
    metrics = d.get("metrics", {})
    test("Dashboard Tab2 requires: AIME_2024 for bar chart", "AIME_2024" in metrics)
    test("Dashboard Tab2 requires: TruthfulQA_MC1 for bar chart", "TruthfulQA_MC1" in metrics)
    test("Dashboard Tab2 requires: Total_End_To_End_Latency for pie chart", 
         "Total_End_To_End_Latency" in metrics)
except Exception as e:
    test("Dashboard Tab2 dependencies", False, str(e))

try:
    # Tab 3: Ingest endpoint (POST /api/ingest)
    r = requests.post(f"{API_BASE}/api/ingest", 
                      json={"documents": [{
                          "title": "Dashboard Test Doc",
                          "domain": "nanotechnology",
                          "content": "Test content for dashboard verification.",
                          "year": 2024,
                          "tags": ["test"],
                          "allowed_roles": ["public"],
                          "authors": ["Dashboard Tester"]
                      }]}, timeout=15)
    test("Dashboard Tab3 ingest endpoint works", r.status_code == 200)
    d = r.json()
    test("Dashboard Tab3 requires: status=success", d.get("status") == "success")
    test("Dashboard Tab3 requires: inserted_ids", len(d.get("inserted_ids", [])) > 0)
except Exception as e:
    test("Dashboard Tab3 dependencies", False, str(e))

# =============================================
# SECTION 6: DOCKER CONFIGURATION VALIDATION
# =============================================
section("Docker Configuration Validation")

try:
    # Check Dockerfile exists and has valid structure
    with open("e:/CrossMind-Updated/Dockerfile", "r") as f:
        content = f.read()
    test("Dockerfile exists", True)
    test("Dockerfile has FROM instruction", "FROM" in content)
    test("Dockerfile has EXPOSE instruction", "EXPOSE" in content)
    test("Dockerfile has CMD or ENTRYPOINT", "CMD" in content or "ENTRYPOINT" in content)
except FileNotFoundError:
    test("Dockerfile exists", False, "Dockerfile not found")
except Exception as e:
    test("Dockerfile validation", False, str(e))

try:
    with open("e:/CrossMind-Updated/docker-compose.yml", "r") as f:
        content = f.read()
    test("docker-compose.yml exists", True)
    test("docker-compose has version", "version" in content.lower() or "services" in content)
    test("docker-compose has services definition", "services" in content)
    test("docker-compose has ports mapping", "ports" in content)
    test("docker-compose references Dockerfile", "build" in content or "image" in content)
except FileNotFoundError:
    test("docker-compose.yml exists", False, "docker-compose.yml not found")
except Exception as e:
    test("docker-compose validation", False, str(e))

# =============================================
# SECTION 7: SSE STREAMING EDGE CASES
# =============================================
section("SSE Streaming Edge Cases")

try:
    # Test with very short query
    r = requests.get(f"{API_BASE}/api/stream_reasoning",
                     params={"query": "test", "user_role": "researcher"},
                     timeout=30, stream=True)
    test("SSE with short query returns 200", r.status_code == 200)
    events = 0
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            events += 1
    test("SSE with short query has events", events > 0, f"got {events} events")
except Exception as e:
    test("SSE short query", False, str(e))

try:
    # Test SSE with Spanish query
    r = requests.get(f"{API_BASE}/api/stream_reasoning",
                     params={"query": "Buscar conexiones entre biomarcadores y nanomateriales",
                             "user_role": "researcher"},
                     timeout=30, stream=True)
    test("SSE Spanish query returns 200", r.status_code == 200)
    events = 0
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            events += 1
    test("SSE Spanish query has events", events > 0, f"got {events} events")
except Exception as e:
    test("SSE Spanish query", False, str(e))

# =============================================
# SECTION 8: SERVER RESILIENCE
# =============================================
section("Server Resilience")

# Test server handles rapid sequential requests
try:
    for i in range(10):
        r = requests.get(f"{API_BASE}/", timeout=5)
        if r.status_code != 200:
            test(f"Rapid request {i} returns 200", False)
            break
    else:
        test("10 rapid sequential root requests", True)
except Exception as e:
    test("Rapid sequential requests", False, str(e))

# Test large payload rejection
try:
    large_payload = {"documents": [
        {"title": f"Large Doc {i}", "domain": "neuroscience",
         "content": "X" * 10000,  # 10KB content
         "year": 2024, "tags": ["large"], "allowed_roles": ["public"],
         "authors": ["Tester"]}
        for i in range(50)
    ]}
    r = requests.post(f"{API_BASE}/api/ingest", json=large_payload, timeout=30)
    test("Large payload ingestion returns 200", r.status_code == 200)
    d = r.json()
    test("Large payload ingested 50 docs", d.get("ingested_count") == 50)
except Exception as e:
    test("Large payload handling", False, str(e))

# =============================================
# SUMMARY
# =============================================
print(f"\n{'='*70}")
print(f"THOROUGH TEST RESULTS SUMMARY")
print(f"{'='*70}")
print(f"Total Tests: {TOTAL}")
print(f"Passed:      {PASS}")
print(f"Failed:      {FAIL}")
if FAILURES:
    print(f"\nFailed Tests:")
    for f in FAILURES:
        print(f"  - {f}")
pass_rate = (PASS / TOTAL) * 100 if TOTAL > 0 else 0
print(f"\nPass Rate: {pass_rate:.1f}%")
print(f"{'='*70}")
