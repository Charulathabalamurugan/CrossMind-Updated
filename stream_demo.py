"""Show streaming SSE-style events the dashboard would receive."""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline

nsp = get_neuro_symbolic_pipeline()
print("=" * 60)
print("STREAMING OUTPUT (SSE events to dashboard)")
print("=" * 60)
print()

events = list(nsp.stream_query(
    "What are lithium-ion batteries?",
    user_role="admin",
    session_id="stream-demo"
))
print(f"Total events emitted: {len(events)}")
print()
for i, e in enumerate(events[:20]):
    evt = e.get("event", "?")
    data = e.get("data", {})
    if isinstance(data, dict):
        delta = data.get("delta", "")
        if delta:
            print(f"[{i+1}] {evt}: {str(delta)[:120]}")
        else:
            summary = {k: v for k, v in data.items() if k != "structured_result" and k != "retrieved_evidence"}
            print(f"[{i+1}] {evt}: {json.dumps({k: (v if not isinstance(v, list) or len(v) < 5 else f'<{len(v)} items>') for k, v in summary.items()})[:160]}")
    else:
        print(f"[{i+1}] {evt}: {str(data)[:160]}")
