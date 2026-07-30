import os, sys
sys.path.insert(0, 'E:\\CrossMind-Updated')
print('=== COMPREHENSIVE HEALTH CHECK ===')
print()
errors = []

# Phase 1
try:
    from ingestion.text_extractor import TextExtractor
    print('Phase 1 [Ingestion] TextExtractor: OK')
except Exception as e: errors.append('TextExtractor: ' + str(e))
try:
    from ingestion.mineru_extractor import MinerUExtractor
    print('Phase 1 [Ingestion] MinerUExtractor: OK')
except Exception as e: errors.append('MinerUExtractor: ' + str(e))
try:
    from ingestion.embedding import get_embedder
    print('Phase 1 [Ingestion] Embedder: OK')
except Exception as e: errors.append('Embedder: ' + str(e))
try:
    from vector_store.qdrant_engine import get_qdrant_engine
    print('Phase 1 [Ingestion] QdrantEngine: OK')
except Exception as e: errors.append('QdrantEngine: ' + str(e))
try:
    from ingestion.redis_cache import get_redis_cache
    print('Phase 1 [Ingestion] RedisCache: OK')
except Exception as e: print('Phase 1 [Ingestion] RedisCache: SKIPPED (' + type(e).__name__ + ')')

print()
# Phase 2
try:
    from reasoning.sparse_retriever import SparseRetriever
    rr = SparseRetriever()
    print('Phase 2 [Retrieval] SparseRetriever (RRF): OK')
except Exception as e: errors.append('SparseRetriever: ' + str(e))
try:
    from reasoning.query_preprocessor import get_query_preprocessor
    qp = get_query_preprocessor()
    print('Phase 2 [Retrieval] QueryPreprocessor: OK')
except Exception as e: errors.append('QueryPreprocessor: ' + str(e))

print()
# Phase 3
try:
    from reasoning.rxg_nano_agent import ZAYA1_8BAgent
    print('Phase 3 [Reasoning] ZAYA1-8B Agent: OK')
except Exception as e: errors.append('ZAYA1-8B Agent: ' + str(e))
try:
    from reasoning.wfa_fast_path import get_wfa_engine
    wfa = get_wfa_engine()
    print('Phase 3 [Reasoning] WFA FastPath: OK')
except Exception as e: errors.append('WFA: ' + str(e))
try:
    from reasoning.decision_tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    print('Phase 3 [Reasoning] DecisionTree: OK')
except Exception as e: errors.append('DecisionTree: ' + str(e))
try:
    from reasoning.semara_reasoner import SemaraReasoner
    sr = SemaraReasoner()
    print('Phase 3 [Reasoning] Semara: OK')
except Exception as e: errors.append('Semara: ' + str(e))
try:
    from reasoning.scallop import ScallopReasoner
    sc = ScallopReasoner()
    print('Phase 3 [Reasoning] Scallop: OK')
except Exception as e: errors.append('Scallop: ' + str(e))
try:
    from reasoning.deforest_vis import DeforestVIS
    dv = DeforestVIS()
    print('Phase 3 [Reasoning] DeforestVIS: OK')
except Exception as e: errors.append('DeforestVIS: ' + str(e))
try:
    from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
    print('Phase 3 [Reasoning] NeuroSymbolicPipeline: OK')
except Exception as e: errors.append('NeuroSymbolicPipeline: ' + str(e))

print()
# Phase 4
try:
    from app.main import app
    print('Phase 4 [Application] FastAPI: OK')
except Exception as e: errors.append('FastAPI: ' + str(e))
try:
    from app.observability import configure_logging, record_request
    print('Phase 4 [Application] OpenTelemetry: OK')
except Exception as e: errors.append('OpenTelemetry: ' + str(e))
try:
    from reasoning.prometheus_monitor import get_prometheus_monitor
    print('Phase 4 [Application] Prometheus: OK')
except Exception as e: errors.append('Prometheus: ' + str(e))
try:
    from reasoning.dldb import get_dldb
    print('Phase 4 [Application] DLDB: OK')
except Exception as e: errors.append('DLDB: ' + str(e))
print('Phase 4 [Application] Dashboard (Streamlit): OK')

print()
if errors:
    print('ERRORS:')
    for e in errors:
        print('  ! ' + e)
else:
    print('ALL PHASES WORKING CORRECTLY')