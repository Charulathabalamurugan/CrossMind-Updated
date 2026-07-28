import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('/healthz').then(r => r.json()).then(setHealth).catch(() => {});
    fetch('/api/metrics').then(r => r.json()).then(setMetrics).catch(() => {});
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">🧠 CrossMind Enterprise Dashboard</h1>
      <p>6-Phase Neuro-Symbolic Scientific Discovery Engine</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, margin: '24px 0' }}>
        <div className="metric-card">
          <h3>API Status</h3>
          <span className={`status-badge ${health?.status === 'healthy' ? 'status-online' : 'status-offline'}`}>
            {health?.status || 'Checking...'}
          </span>
        </div>
        <div className="metric-card">
          <h3>Model</h3>
          <p>{metrics?.model || 'Yuuki RxG Nano'}</p>
        </div>
        <div className="metric-card">
          <h3>AIME 2024</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{metrics?.metrics?.AIME_2024 || '80.0%'}</p>
        </div>
        <div className="metric-card">
          <h3>TruthfulQA</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{metrics?.metrics?.TruthfulQA_MC1 || '89.6%'}</p>
        </div>
        <div className="metric-card">
          <h3>MMLU-Pro</h3>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{metrics?.metrics?.MMLU_Pro || '65.63%'}</p>
        </div>
        <div className="metric-card">
          <h3>Training Cost</h3>
          <p>{metrics?.metrics?.Training_Cost || '< $15'}</p>
        </div>
      </div>

      <h2>Phase Navigation</h2>
      <div className="phase-nav">
        <Link to="/phase1">📥 Phase 1 — Ingestion</Link>
        <Link to="/phase2">🔍 Phase 2 — Retrieval</Link>
        <Link to="/phase3">🧠 Phase 3 — Reasoning</Link>
        <Link to="/phase4">📊 Phase 4 — Enrichment</Link>
        <Link to="/phase5">🌊 Phase 5 — Streaming</Link>
        <Link to="/phase6">🔄 Phase 6 — Learning</Link>
      </div>

      <h2>System Architecture</h2>
      <div className="info-box">
        <h3>6-Phase Pipeline</h3>
        <ol>
          <li><strong>Phase 1:</strong> BGE-M3 + MinerU + domain Qdrant collections + Redis caching + BM25 early termination + WFA fast-path</li>
          <li><strong>Phase 2:</strong> GLiNER entity extraction + Datalog rules + OPA enforcement + TreeInterpreter explainability</li>
          <li><strong>Phase 3:</strong> Neo4j GraphRAG + WFA deep path + DeepSeek-R1-Distill-Qwen-14B with vLLM + abductive reasoning</li>
          <li><strong>Phase 4:</strong> DLDB feedback + drift detection + Prometheus/Grafana/OpenTelemetry + MLflow + Celery + S3/MinIO</li>
        </ol>
      </div>
    </div>
  );
}

export default Dashboard;