import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Phase1Ingestion from './phases/Phase1Ingestion';
import Phase2Retrieval from './phases/Phase2Retrieval';
import Phase3Reasoning from './phases/Phase3Reasoning';
import Phase4Enrichment from './phases/Phase4Enrichment';
import Phase5Streaming from './phases/Phase5Streaming';
import Phase6Learning from './phases/Phase6Learning';
import Dashboard from './Dashboard';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/phase1" element={<Phase1Ingestion />} />
      <Route path="/phase2" element={<Phase2Retrieval />} />
      <Route path="/phase3" element={<Phase3Reasoning />} />
      <Route path="/phase4" element={<Phase4Enrichment />} />
      <Route path="/phase5" element={<Phase5Streaming />} />
      <Route path="/phase6" element={<Phase6Learning />} />
    </Routes>
  );
}

export default App;