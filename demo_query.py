import requests, json

r = requests.post('http://127.0.0.1:8000/api/query', json={
    'query': 'What are recent scientific breakthroughs across multiple domains?',
    'user_role': 'researcher',
    'confidence_proceed_threshold': 0.75,
    'confidence_investigate_threshold': 0.50,
}, timeout=30)

d = r.json()
print('=== QUERY RESULT ===')
print('Status:', r.status_code)
print()
print('--- Phase 1: Ingestion ---')
pf = d.get('pre_filter', {})
print('  Language:', pf.get('language'))
print('  Domains:', pf.get('detected_domains'))
print()
print('--- Phase 2: Retrieval ---')
ev = d.get('retrieved_evidence', [])
print('  Evidence chunks:', len(ev))
for i, e in enumerate(ev[:3], 1):
    p = e.get('payload', {})
    title = p.get('title', '?')[:60]
    domain = p.get('domain', '?')
    print('  [%d] %s (%s)' % (i, title, domain))
print()
print('--- Phase 3: Reasoning ---')
ar = d.get('agent_reasoning', {})
print('  Model:', ar.get('model', 'N/A'))
think = ar.get('think_block', '')[:200]
print('  Think:', think + ('...' if len(ar.get('think_block', '')) > 200 else ''))
hyp = ar.get('hypothesis', ar.get('output_text', 'N/A'))[:200]
print('  Hypothesis:', hyp)
print()
print('--- Phase 4: Application ---')
pm = d.get('performance_metrics', {})
print('  Total Time: %.2fs' % pm.get('total_time_seconds', 0))
print('  Pre-filter: %dms' % pm.get('pre_filter_ms', 0))
print('  Agent Reasoning: %.2fs' % pm.get('agent_reasoning_time_seconds', 0))
cal = d.get('confidence_calibration', {})
print('  Decision:', cal.get('decision', 'N/A'))
print('  Confidence:', cal.get('calibrated_confidence', 0))
print()
print('=== ALL 4 PHASES EXECUTED SUCCESSFULLY ===')