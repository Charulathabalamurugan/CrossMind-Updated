import importlib.util as u
mods = [
    'fastapi', 'pydantic', 'pydantic_settings', 'numpy', 'diskcache', 'z3',
    'rank_bm25', 'networkx', 'requests', 'httpx', 'sentence_transformers',
    'redis', 'celery', 'nltk', 'sklearn', 'rdflib', 'colbert', 'scipy',
    'pyvis', 'plotly', 'streamlit', 'opentelemetry', 'prometheus_client',
    'qdrant_client', 'mineru', 'tika', 'semara', 'scallop', 'deforestvis',
    'vllm', 'pytest', 'unittest',
]
for m in mods:
    print(f"{m}: {'yes' if u.find_spec(m) else 'NO'}")