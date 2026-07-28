import React, { useState } from 'react';

function Phase4Enrichment() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'Find cross-domain links between Alzheimer biomarkers and nanomaterials', user_role: 'researcher' }),
      });
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setResult({ error: e.message });
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">📊 Phase 4: Enrichment & Memory</h1>
      <p>GraphRAG construction, bridge scoring, evidence attribution, abductive reasoning, experimental blueprints, dual-memory profiles</p>

      <button onClick={handleRun} disabled={loading} style={{ padding: '12px 24px', fontSize: '1rem', cursor: loading ? 'wait' : 'pointer', background: '#3B82F6', color: 'white', border: 'none', borderRadius: 8 }}>
        {loading ? 'Enriching...' : '📊 Enrich Results'}
      </button>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Bridge Scoring</h3>
          <div className="metric-card">
            <p>Cross-Domain Paths: {result.graph_rag?.cross_domain_path_count || 0}</p>
            <p>Total Paths: {(result.graph_rag?.multi_hop_paths || []).length}</p>
            <p>Graph Nodes: {result.performance_metrics?.graph_nodes_count || 0}</p>
            <p>Discovery Score: {result.cross_domain_scoring?.overall_score}% ({result.cross_domain_scoring?.rating})</p>
          </div>

          <h3>Evidence Attribution</h3>
          <div className="info-box">
            <p>Coverage: {result.evidence_attribution?.overall_attribution_coverage}%</p>
            <p>Supported: {result.evidence_attribution?.supported_claims}/{result.evidence_attribution?.total_claims} claims</p>
          </div>

          <h3>Experimental Blueprint</h3>
          <div className="metric-card">
            {result.experimental_blueprint?.status !== 'disabled' ? (
              <>
                <p><strong>{result.experimental_blueprint?.title}</strong></p>
                <p>Objective: {result.experimental_blueprint?.primary_objective}</p>
                <p>Timeline: {result.experimental_blueprint?.timeline_estimate} | Confidence: {result.experimental_blueprint?.confidence}</p>
              </>
            ) : (
              <p>Blueprint not generated for this query.</p>
            )}
          </div>

          <h3>Collaboration Recommendations</h3>
          <div className="info-box">
            {(result.collaboration_recommendations?.recommendations || []).map((rec, i) => (
              <div key={i}>
                <strong>{rec.recommended_role}</strong> — {rec.primary_domain} (strength {rec.collaboration_strength})
                <p style={{ fontSize: '0.85rem', color: '#6b7280' }}>{rec.rationale}</p>
              </div>
            ))}
          </div>

          <h3>Dual-Memory Profile</h3>
          <div className="metric-card">
            <p>Cognitive Persona: {result.memory_footprint?.individual?.persona?.cognitive_style || 'N/A'}</p>
            <p>Safety Focus: {result.memory_footprint?.individual?.persona?.safety_focus_level || 'N/A'}</p>
            <p>Interactions: {result.memory_footprint?.individual?.persona?.interaction_count || 0}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default Phase4Enrichment;