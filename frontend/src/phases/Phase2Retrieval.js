import React, { useState } from 'react';

function Phase2Retrieval() {
  const [query, setQuery] = useState('Find cross-domain links between Alzheimer biomarkers and nanomaterials');
  const [result, setResult] = useState(null);

  const handleSearch = async () => {
    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, user_role: 'researcher' }),
      });
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setResult({ error: e.message });
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">🔍 Phase 2: Hybrid Retrieval (Sparse + Dense)</h1>
      <p>TF-IDF BM25 keyword search fused with Qdrant HNSW semantic retrieval via Reciprocal Rank Fusion</p>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ width: '100%', height: 80, padding: 12, fontSize: '1rem', marginBottom: 12 }}
      />
      <button onClick={handleSearch} style={{ padding: '12px 24px', fontSize: '1rem', cursor: 'pointer', background: '#3B82F6', color: 'white', border: 'none', borderRadius: 8 }}>
        Hybrid Retrieval (RRF)
      </button>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Retrieved Evidence ({result.retrieved_evidence?.length || 0} chunks)</h3>
          <div className="info-box">
            <p><strong>Pre-filter:</strong> {result.pre_filter?.language} | Domains: {(result.pre_filter?.detected_domains || []).join(', ')} | Entities: {(result.pre_filter?.extracted_entities || []).join(', ')}</p>
            <p><strong>Retrieval Strategy:</strong> {result.performance_metrics?.retrieved_chunks_count || 0} chunks in {result.performance_metrics?.pre_filter_ms || 0}ms pre-filter</p>
          </div>

          {(result.retrieved_evidence || []).map((ev, i) => (
            <div key={i} className="metric-card" style={{ marginTop: 12 }}>
              <h4>{ev.payload?.title || 'Untitled'} (Score: {ev.score?.toFixed(4)})</h4>
              <p>Domain: {ev.payload?.domain} | Source: {ev.retrieval_source?.join(', ')}</p>
              <p style={{ fontSize: '0.85rem', color: '#6b7280' }}>{ev.payload?.content?.substring(0, 200)}...</p>
            </div>
          ))}

          <div className="info-box" style={{ marginTop: 24 }}>
            <h3>Retrieval Strategy:</h3>
            <ol>
              <li>Sparse Retrieval (TF-IDF/BM25) — O(log n) keyword matches</li>
              <li>Dense Retrieval (Qdrant HNSW) — O(log n) semantic matches</li>
              <li>Reciprocal Rank Fusion (RRF) — combines both rankings</li>
              <li>BM25 Early Termination — if BM25 score > {parseFloat(require('../../config').settings.BM25_EARLY_TERMINATION_THRESHOLD || 0.95)}</li>
              <li>RBAC Filtering — role-based access control</li>
              <li>Redis Query Cache — LRU cache with TTL for frequent queries</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}

export default Phase2Retrieval;