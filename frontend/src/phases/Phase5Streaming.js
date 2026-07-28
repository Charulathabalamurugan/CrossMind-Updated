import React, { useState } from 'react';

function Phase5Streaming() {
  const [query, setQuery] = useState('How do nanoparticles cross the blood-brain barrier?');
  const [streaming, setStreaming] = useState(false);
  const [events, setEvents] = useState([]);

  const handleStream = async () => {
    setStreaming(true);
    setEvents([]);
    try {
      const resp = await fetch(`/api/stream_reasoning?query=${encodeURIComponent(query)}&user_role=researcher`);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n').filter(l => l.trim());
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const evt = JSON.parse(line.substring(6));
              setEvents(prev => [...prev, evt]);
            } catch (e) {
              // Skip non-JSON lines
            }
          }
        }
      }
    } catch (e) {
      setEvents(prev => [...prev, { event: 'error', data: { error: e.message } }]);
    }
    setStreaming(false);
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 className="phase-header">🌊 Phase 5: Structured Streaming & Output</h1>
      <p>Real-time SSE streaming with confidence scores, citations, and structured JSON for frontend consumption</p>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ width: '100%', height: 80, padding: 12, fontSize: '1rem', marginBottom: 12 }}
      />
      <button onClick={handleStream} disabled={streaming} style={{ padding: '12px 24px', fontSize: '1rem', cursor: streaming ? 'wait' : 'pointer', background: '#8B5CF6', color: 'white', border: 'none', borderRadius: 8 }}>
        {streaming ? 'Streaming...' : '📡 Start Streaming'}
      </button>

      {events.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3>Live Stream Events ({events.length})</h3>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            {events.map((evt, i) => (
              <div key={i} className="metric-card" style={{ marginBottom: 8 }}>
                <strong>Event {i + 1}:</strong> <code>{evt.type || evt.event || 'unknown'}</code>
                <pre style={{ fontSize: '0.8rem', maxHeight: 200, overflow: 'auto' }}>
                  {typeof evt.data === 'string' ? evt.data : JSON.stringify(evt.data, null, 2).substring(0, 1000)}
                </pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Phase5Streaming;