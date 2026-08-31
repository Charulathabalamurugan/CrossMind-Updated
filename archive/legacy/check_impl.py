import os, sys

base = 'E:\\CrossMind-Updated'
checks = [
    ('FastAPI', 'app/main.py'),
    ('MinerU', 'ingestion/mineru_extractor.py'),
    ('Apache Tika', 'ingestion/text_extractor.py'),
    ('BGE-M3 embedding', 'ingestion/embedding.py'),
    ('Qdrant', 'vector_store/qdrant_engine.py'),
    ('Redis cache', 'ingestion/redis_cache.py'),
    ('BM25 engine', 'vector_store/qdrant_engine.py'),
    ('RRF fusion', 'reasoning/sparse_retriever.py'),
    ('RBAC middleware', 'app/main.py'),
    ('ZAYA1-8B agent', 'reasoning/rxg_nano_agent.py'),
    ('Semara reasoner', 'reasoning/semara_reasoner.py'),
    ('GraphRAG', 'reasoning/hybrid_rag_kg.py'),
    ('WFA fast path', 'reasoning/wfa_fast_path.py'),
    ('DLDB', 'reasoning/dldb.py'),
    ('SSE endpoint', 'app/main.py'),
    ('OpenTelemetry', 'app/observability.py'),
    ('Prometheus monitor', 'reasoning/prometheus_monitor.py'),
]
for tech, path in checks:
    full = os.path.join(base, path)
    exists = os.path.exists(full)
    # Check if it actually uses the tech (not just exists)
    uses_tech = False
    if exists and path.endswith('.py'):
        try:
            with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Check for actual key terms
            key_terms = {
                'FastAPI': 'fastapi',
                'MinerU': 'mineru',
                'Apache Tika': 'tika',
                'BGE-M3 embedding': 'bge' if 'bge' in path else '',
                'Qdrant': 'qdrant',
                'Redis cache': 'redis',
                'BM25 engine': 'BM25',
                'RRF fusion': 'reciprocal',
                'RabAC': 'rbac',
                'ZAYA1-8B agent': 'zaya' if 'zaya' in path else 'ZAYA1',
                'Semara reasoner': 'semara',
                'GraphRAG': 'graph_rag' if 'graph_rag' in path else 'graph',
                'WFA fast path': 'wfa',
                'DLDB': 'dldb',
                'SSE endpoint': 'sse' if 'sse' in path.lower() else 'EventSource',
                'OpenTelemetry': 'opentelemetry',
                'Prometheus monitor': 'prometheus',
            }
            term = key_terms.get(tech, tech.lower().replace(' ', ''))
            uses_tech = term.lower() in content.lower()
        except:
            pass
    status = 'USES TECH' if uses_tech else ('FILE EXISTS' if exists else 'NOT FOUND')
    print(f'{status}: {tech}')