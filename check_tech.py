import os, re

base = 'E:\\CrossMind-Updated'
files_to_check = {
    'MinerU': ['ingestion/mineru_extractor.py', 'ingestion/pipeline.py'],
    'Tika': ['ingestion/text_extractor.py', 'ingestion/pipeline.py'],
    'BGE-M3': ['ingestion/embedding.py'],
    'BM25': ['reasoning/sparse_retriever.py', 'reasoning/query_preprocessor.py'],
    'RRF': ['reasoning/sparse_retriever.py', 'reasoning/query_preprocessor.py'],
    'ColBERT': ['vector_store/qdrant_engine.py'],
    'Scallop': ['reasoning/neuro_symbolic_pipeline.py'],
    'Semara': ['reasoning/semara_reasoner.py'],
    'DeforestVIS': ['reasoning/neuro_symbolic_pipeline.py', 'dashboard/app.py'],
    'WFA': ['reasoning/wfa_fast_path.py'],
    'DecisionTree': ['reasoning/neuro_symbolic_pipeline.py'],
    'GraphRAG': ['reasoning/neuro_symbolic_pipeline.py', 'reasoning/hybrid_rag_kg.py'],
    'OpenTelemetry': ['app/main.py', 'app/observability.py'],
    'Prometheus': ['app/main.py', 'reasoning/prometheus_monitor.py'],
    'DLDB': ['reasoning/dldb.py'],
    'DiskCache': ['reasoning/neuro_symbolic_pipeline.py'],
    'SSE': ['app/main.py'],
}

for tech, paths in files_to_check.items():
    implementations = []
    for p in paths:
        full = os.path.join(base, p)
        if os.path.exists(full):
            try:
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                search = tech.lower().replace('-', '').replace(' ', '')
                # Special handling for different names
                if tech == 'DecisionTree':
                    search = 'decision_tree'
                elif tech == 'DeforestVIS':
                    search = 'deforestvis'
                elif tech == 'WFA':
                    search = 'wfa'
                elif tech == 'BM25':
                    search = 'bm25'
                elif tech == 'RRF':
                    search = 'rrf'
                elif tech == 'ColBERT':
                    search = 'colbert'
                if search in content:
                    implementations.append(p)
            except:
                pass
    status = 'FOUND' if implementations else 'NOT IMPLEMENTED'
    impl_str = ', '.join(implementations) if implementations else 'none'
    print(f'{tech}: {status} in [{impl_str}]')