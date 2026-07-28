import React, { useState } from 'react';

function Phase3Reasoning() {
  const [query, setQuery] = useState('Find cross-domain links between Alzheimer biomarkers and nanomaterials');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
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
    setLoading(false);
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">🧠 Phase 3: Neuro-Symbolic Reasoning Core</h1>
      <p>Decision Tree fast-path → GraphRAG → Abductive deep path → DeepSeek-R1-Distill-Qwen-14B via vLLM</p>

      <button onClick={handleRun} disabled={loading} style={{ padding: '12px 24px', fontSize: '1rem', cursor: loading ? 'wait' : 'pointer', background: '#3B82F6', color: 'white', border: 'none', borderRadius: 8 }}>
        {loading ? 'Running Pipeline...' : '🧠 Run Full Reasoning Pipeline'}
      </button>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Sub-Phase 3a: Pre-Filter</h3>
          <div className="info-box">
            <p>Language: {result.pre_filter?.language} | Domains: {(result.pre_filter?.detected_domains || []).join(', ')}</p>
            <p>Entities: {(result.pre_filter?.extracted_entities || []).join(', ')} | Time: {result.performance_metrics?.pre_filter_ms}ms</p>
          </div>

          <h3>Sub-Phase 3b: Hypothesis + Rule Engine</h3>
          <div className="metric-card">
            <h4>Generated Hypothesis</h4>
            <p>{result.agent_reasoning?.hypothesis || result.agent_reasoning?.output_text}</p>
            <p>Confidence: {(result.confidence_calibration?.calibrated_confidence || 0) * 100}%</p>
            <p>Model: {result.agent_reasoning?.model || 'Yuuki RxG Nano'}</p>
          </div>

          <h3>Sub-Phase 3c: Post-Validation + Z3</h3>
          <div className="success-box">
            <p>Symbolic Validation: {result.post_validation?.validation_score}% | Passed: {result.post_validation?.validated ? '✅' : '⚠️'}</p>
            <p>Z3 Validation: {result.z3_formal_validation?.execution_mode} | Score: {result.z3_formal_validation?.validation_score}% | Passed: {result.z3_formal_validation?.validated ? '✅' : '⚠️'}</p>
            <p>Total Time: {result.performance_metrics?.total_time_seconds}s</p>
          </div>

          <div className="info-box" style={{ marginTop: 24 }}>
            <h3>Reasoning Architecture:</h3>
            <ul>
              <li><strong>WFA Fast-Path (80%):</strong> Decision tree + Weighted Finite Automata for high-confidence, single-domain queries → instant return</li>
              <li><strong>GraphRAG Slow Path (15%):</strong> Neo4j multi-hop graph traversal for complex queries</li>
              <li><strong>Abductive Deep Path (5%):</strong> Competing hypothesis generation for causal queries</li>
              <li><strong>DeepSeek-R1-Distill-Qwen-14B:</strong> 4-bit quantized via vLLM for final hypothesis generation</li>
              <li><strong>TreeInterpreter:</strong> Feature-level explainability for decision tree outputs</li>
              <li><strong>Datalog Rule Traces:</strong> Logical explainability for rule-based validation</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default Phase3Reasoning;