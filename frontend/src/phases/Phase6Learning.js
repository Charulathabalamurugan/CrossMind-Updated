import React, { useState, useEffect } from 'react';

function Phase6Learning() {
  const [feedback, setFeedback] = useState({ total: 0, applied: 0, avgRelevance: 0, shouldRetrain: false });
  const [ruleEngine, setRuleEngine] = useState(null);
  const [retrainer, setRetrainer] = useState(null);
  const [monitor, setMonitor] = useState(null);
  const [registry, setRegistry] = useState(null);
  const [dldb, setDldb] = useState(null);
  const [drift, setDrift] = useState(null);

  useEffect(() => {
    fetch('/api/feedback/stats').then(r => r.json()).then(setFeedback).catch(() => {});
    fetch('/api/rule-engine/status').then(r => r.json()).then(setRuleEngine).catch(() => {});
    fetch('/api/model/retrain/status').then(r => r.json()).then(setRetrainer).catch(() => {});
    fetch('/api/monitor/summary').then(r => r.json()).then(setMonitor).catch(() => {});
    fetch('/api/mlflow/stats').then(r => r.json()).then(setRegistry).catch(() => {});
    fetch('/api/dldb/stats').then(r => r.json()).then(setDldb).catch(() => {});
    fetch('/api/drift/status').then(r => r.json()).then(setDrift).catch(() => {});
  }, []);

  const submitFeedback = async (score) => {
    try {
      await fetch('/api/feedback/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test', doc_id: 'doc1', score, user_role: 'researcher', evidence_domains: ['neuroscience'] }),
      });
      const resp = await fetch('/api/feedback/stats');
      const data = await resp.json();
      setFeedback(data.feedback_stats || {});
    } catch (e) {
      // Silently handle
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">🔄 Phase 6: Continuous Learning & Feedback Loop</h1>
      <p>User feedback collection, drift detection, risk-controlled retraining, dynamic rule engine</p>

      <h2>DLDB & Feedback</h2>
      <div className="metric-card">
        <p>Total Feedback: {feedback.total || 0}</p>
        <p>Applied: {feedback.applied || 0}</p>
        <p>Average Relevance: {((feedback.average_relevance || 0) * 100).toFixed(1)}%</p>
        <p>Should Retrain: {feedback.should_retrain ? '🔴 Yes' : '🟢 No'}</p>
      </div>
      <div style={{ marginTop: 8 }}>
        <button onClick={() => submitFeedback(0.9)} style={{ padding: '8px 16px', margin: 4, background: '#10B981', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>👍 High Relevance</button>
        <button onClick={() => submitFeedback(0.5)} style={{ padding: '8px 16px', margin: 4, background: '#F59E0B', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>👌 Medium</button>
        <button onClick={() => submitFeedback(0.1)} style={{ padding: '8px 16px', margin: 4, background: '#EF4444', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>👎 Low Relevance</button>
      </div>

      <h2>Rule Engine</h2>
      <div className="metric-card">
        <p>Total Rules: {ruleEngine?.rule_count || 0}</p>
        <p>Log Entries: {ruleEngine?.log_entries || 0}</p>
        <p>Rules: {(ruleEngine?.rules || []).join(', ')}</p>
      </div>

      <h2>Retrainer Status</h2>
      <div className="metric-card">
        <p>Enabled: {retrainer?.enabled ? 'Yes' : 'No'}</p>
        <p>Last Retrain: {retrainer?.last_retrain ? new Date(retrainer.last_retrain * 1000).toLocaleString() : 'Never'}</p>
        <p>Retrain Count: {retrainer?.retrain_count || 0}</p>
        <p>Needs Retraining: {retrainer?.needs_retraining ? '🔴 Yes' : '🟢 No'}</p>
      </div>

      <h2>Drift Detection</h2>
      <div className="metric-card">
        <p>Drift Detected: {drift?.drift_detected ? '🔴 Yes' : '🟢 No'}</p>
        <p>Next Check: {drift?.next_check_in_hours ? `${drift.next_check_in_hours}h` : 'N/A'}</p>
      </div>

      <h2>Monitoring Stack</h2>
      <div className="info-box">
        <ul>
          <li><strong>Prometheus:</strong> {monitor?.prometheus?.enabled ? '✅' : '❌'} Metrics collection and alerting</li>
          <li><strong>Grafana:</strong> {monitor?.grafana?.enabled ? '✅' : '❌'} Dashboard visualization</li>
          <li><strong>OpenTelemetry:</strong> {monitor?.opentelemetry?.enabled ? '✅' : '❌'} Distributed tracing</li>
          <li><strong>DLDB:</strong> {dldb?.total_feedback !== undefined ? '✅ Active' : '❌'} Persistent feedback store</li>
          <li><strong>MLflow:</strong> {registry?.registered_models !== undefined ? '✅ Enabled' : '❌'} Model/rule/ontology registry</li>
          <li><strong>Celery:</strong> Background orchestration for ingestion, validation, index, retrain</li>
          <li><strong>S3/MinIO:</strong> Cold storage for old reports</li>
          <li><strong>PostgreSQL:</strong> Document metadata store</li>
          <li><strong>Slinky Operator:</strong> HPC GPU pool sharing on OpenShift/Slurm</li>
          <li><strong>OPA:</strong> Corporate SSO integration (Okta, Azure AD)</li>
          <li><strong>MLflow Registry:</strong> Versioned models, prompts, ontologies, Datalog rules</li>
        </ul>
      </div>
    </div>
  );
}

export default Phase6Learning;