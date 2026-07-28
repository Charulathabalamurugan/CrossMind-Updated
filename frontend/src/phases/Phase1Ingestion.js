import React, { useState } from 'react';

function Phase1Ingestion() {
  const [result, setResult] = useState(null);
  const [queueStats, setQueueStats] = useState(null);

  const handleIngest = async () => {
    try {
      const resp = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documents: [{
            title: 'Ingested via React UI',
            content: 'Test document for Phase 1 ingestion pipeline.',
            domain: 'neuroscience',
            tags: ['test', 'react-ui'],
            allowed_roles: ['public', 'researcher'],
          }]
        })
      });
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setResult({ error: e.message });
    }
  };

  const fetchQueueStats = async () => {
    try {
      const resp = await fetch('/api/queue/stats');
      const data = await resp.json();
      setQueueStats(data);
    } catch (e) {
      setQueueStats({ error: e.message });
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">📥 Phase 1: Async Document Ingestion Pipeline</h1>
      <p>BGE-M3 embeddings + MinerU extraction + domain Qdrant collections + Redis caching + WFA fast-path</p>
      
      <button onClick={handleIngest} style={{ padding: '12px 24px', fontSize: '1rem', cursor: 'pointer', background: '#3B82F6', color: 'white', border: 'none', borderRadius: 8 }}>
        Ingest Test Document
      </button>
      <button onClick={fetchQueueStats} style={{ padding: '12px 24px', fontSize: '1rem', cursor: 'pointer', marginLeft: 16, background: '#10B981', color: 'white', border: 'none', borderRadius: 8 }}>
        Queue Stats
      </button>

      {result && (
        <div className="success-box" style={{ marginTop: 16 }}>
          <h3>Ingestion Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}

      {queueStats && (
        <div className="info-box" style={{ marginTop: 16 }}>
          <h3>Queue Statistics</h3>
          <pre>{JSON.stringify(queueStats, null, 2)}</pre>
        </div>
      )}

      <div className="info-box" style={{ marginTop: 24 }}>
        <h3>Phase 1 Components:</h3>
        <ul>
          <li><strong>MinerU Extractor:</strong> Primary extraction for scientific PDFs</li>
          <li><strong>Apache Tika:</strong> Fallback for office docs and emails</li>
          <li><strong>BGE-M3:</strong> Hybrid dense + sparse embeddings (ONNX/TensorRT INT8)</li>
          <li><strong>Domain Classifier:</strong> Auto-detect biomedical, materials, energy, financial</li>
          <li><strong>Domain Qdrant Collections:</strong> Separate collections per domain + Product Quantization</li>
          <li><strong>Redis Cache:</strong> TTL 1hr, LRU eviction, query result caching</li>
          <li><strong>BM25 Early Termination:</strong> Skip retrieval if BM25 score > threshold</li>
          <li><strong>WFA Fast-Path:</strong> Weighted Finite Automata for 80% of queries</li>
          <li><strong>Celery Background Tasks:</strong> 4hr incremental reindex, 168hr full reindex</li>
        </ul>
      </div>
    </div>
  );
}

export default Phase1Ingestion;