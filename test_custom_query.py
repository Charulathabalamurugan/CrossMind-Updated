import requests
import json

queries = [
    "How do PEGylated PLGA nanoparticles target brain tumors?",
    "Buscar conexiones entre nanoparticulas de oro y el cancer cerebral",
    "Explain the mechanism of ApoE-targeted lipid nanoparticles for Alzheimer therapy"
]

for q in queries:
    print("=" * 60)
    print("QUERY: " + q)
    print("=" * 60)
    try:
        r = requests.post('http://localhost:8000/api/query', json={'query': q, 'user_role': 'researcher'}, timeout=30)
        if r.status_code == 200:
            d = r.json()
            print("Status: " + str(r.status_code))
            print("Domains: " + str(d['pre_filter']['detected_domains']))
            print("Evidence chunks: " + str(len(d['retrieved_evidence'])))
            print("Validation score: " + str(d['post_validation']['validation_score']) + "%")
            print("Validated: " + str(d['post_validation']['validated']))
            print("Hypothesis: " + d['agent_reasoning']['output_text'][:200] + "...")
        else:
            print("Error " + str(r.status_code) + ": " + r.text[:200])
    except Exception as e:
        print("Exception: " + str(e))

print("\n\nALL CUSTOM QUERIES TESTED!")
