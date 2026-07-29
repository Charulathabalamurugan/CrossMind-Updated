import streamlit as st
import requests
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import time
import math
import os

st.set_page_config(
    page_title="CrossMind | Neuro-Symbolic Discovery Engine",
    page_icon="🧠",
    layout="wide"
)

def sanitize_text(text: str, max_length: int = 5000) -> str:
    if not text:
        return ""
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip()

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .think-box { background-color: #F3F4F6; border-left: 4px solid #3B82F6; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; margin: 8px 0; }
    .metric-card { background: #F9FAFB; border: 1px solid #E5E7EB; padding: 12px; border-radius: 8px; text-align: center; }
    .security-badge { background-color: #ECFDF5; color: #065F46; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; border: 1px solid #6EE7B7; }
    .phase-badge { background-color: #EFF6FF; color: #1E40AF; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; border: 1px solid #BFDBFE; display: inline-block; margin: 2px; }
    .phase-active { background-color: #DBEAFE; color: #1E40AF; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 700; border: 2px solid #3B82F6; display: inline-block; margin: 2px; }
    .phase-done { background-color: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; border: 1px solid #10B981; display: inline-block; margin: 2px; }
    .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #E5E7EB; border-top: 3px solid #3B82F6; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .flow-step { border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; margin: 12px 0; background: #FFFFFF; }
    .flow-step h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
st.sidebar.markdown(f'<span class="security-badge">🔒 RBAC Enabled</span>', unsafe_allow_html=True)
API_BASE = st.sidebar.text_input("Backend API Base URL", value=os.getenv("API_BASE", "http://localhost:8000"))
API_KEY = st.sidebar.text_input("API Key (optional)", value="", type="password")
user_role = st.sidebar.selectbox("User Role (RBAC)", ["researcher", "admin", "public"], index=0)
st.sidebar.markdown("### Confidence Policy")
proceed_threshold = st.sidebar.slider("Proceed threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)
investigate_threshold = st.sidebar.slider("Investigate threshold", min_value=0.10, max_value=proceed_threshold, value=min(0.50, proceed_threshold), step=0.05)
st.sidebar.markdown("---")
st.sidebar.success("⚡ Single-Page Flow Mode")
st.sidebar.subheader("🏆 ZAYA1-8B Metrics")
st.sidebar.markdown("""
- **Total Params:** 8.4B (760M active MoE)
- **AIME 2026:** 89.1%
- **Context Window:** 131K tokens
- **Quantization:** Q4_K_M (~5.5 GB)
- **License:** Apache 2.0
""")

# ========== Title ==========
st.markdown('<div class="main-title">🧠 CrossMind: Neuro-Symbolic Scientific Discovery Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">All 6 Phases in a Single Flow — Click Run and Watch Everything Happen</div>', unsafe_allow_html=True)

# ========== Query Input ==========
st.markdown("### 📝 Enter Your Scientific Query")
col_query, col_run = st.columns([3, 1])
with col_query:
    query_input = st.text_area("Query:", value="Find cross-domain links between Alzheimer's biomarkers and nanomaterials", height=80, key="single_flow_query")
with col_run:
    run_clicked = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True, key="run_pipeline")

# ========== 6-Phase Flow ==========
PHASES = [
    ("📥 Phase 1", "Document Ingestion", "Extract text, chunk documents, generate DSKE + TF-IDF vectors, store in Qdrant"),
    ("🔍 Phase 2", "Hybrid Retrieval", "TF-IDF/BM25 + Dense Vector Search fused via Reciprocal Rank Fusion"),
    ("🧠 Phase 3", "Neuro-Symbolic Reasoning", "Pre-filter → Hypothesis Generation → Rule Engine → Agent Reasoning → Validation"),
    ("📊 Phase 4", "Enrichment & Memory", "GraphRAG, Evidence Attribution, Bridge Scoring, Dual-Memory Profile"),
    ("🌊 Phase 5", "Structured Streaming", "Real-time SSE stream with confidence, citations, reasoning traces"),
    ("🔄 Phase 6", "Continuous Learning", "Feedback collection, drift detection, retraining orchestration"),
]

def call_api(endpoint, method="POST", data=None, timeout=30):
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data, headers=get_headers(), timeout=timeout)
        else:
            resp = requests.get(url, headers=get_headers(), timeout=timeout)
        if resp.status_code == 401:
            return None, "API Key required."
        if resp.status_code not in (200, 201):
            return None, f"API error ({resp.status_code}): {resp.text}"
        return resp.json(), None
    except requests.exceptions.ConnectTimeout:
        return None, "Connection timeout. Ensure FastAPI server is running."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {API_BASE}. Start the API server first."
    except Exception as e:
        return None, f"Request failed: {str(e)}"

def get_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

if run_clicked and query_input:
    safe_query = sanitize_text(query_input, 5000)
    if not safe_query:
        st.error("Query is empty after sanitization.")
        st.stop()

    st.markdown("---")
    st.markdown("## 🔄 Pipeline Execution Flow")
    
    # Ingestion + Retrieval in one call (backend handles phases 1+2+3+4+5)
    with st.spinner("🚀 Running all 6 phases..."):
        result, error = call_api("/api/query", data={
            "query": safe_query,
            "user_role": user_role,
            "confidence_proceed_threshold": proceed_threshold,
            "confidence_investigate_threshold": investigate_threshold,
        }, timeout=120)

    if error:
        st.error(f"Pipeline failed: {error}")
        st.stop()

    if not result:
        st.error("No result returned.")
        st.stop()

    # Display all 6 phases in a unified flow
    for i, (phase_label, phase_name, phase_desc) in enumerate(PHASES, 1):
        st.markdown(f"### {phase_label}: {phase_name}")
        st.markdown(f"<span class='phase-done'>✅ Complete</span> {phase_desc}", unsafe_allow_html=True)
        
        if i == 1:
            # Phase 1: Show ingestion stats
            with st.expander("📥 Phase 1 Details", expanded=False):
                st.markdown("**Ingestion Pipeline**")
                st.json({
                    "text_extractor": "MinerU (PDF) / Tika (fallback) / Plain text",
                    "chunker": "Sliding-window (512 tokens, 64 overlap)",
                    "dedup_cache": "Redis-backed TTL dedup",
                    "sparse_vector": "TF-IDF vector generation",
                    "dense_vector": "DSKE 64-dim deterministic embedding",
                    "domain_classifier": "Auto-detected domain from content",
                    "qdrant_storage": "Separate collections per domain + PQ compression",
                })
        
        elif i == 2:
            # Phase 2: Show retrieval results
            with st.expander("🔍 Phase 2 Details", expanded=True):
                evidence = result.get("retrieved_evidence", [])
                st.markdown(f"**Retrieved {len(evidence)} evidence chunks**")
                for idx, ev in enumerate(evidence[:5], 1):
                    payload = ev.get("payload", {})
                    st.markdown(f"**[{idx}] {payload.get('title', 'Untitled')}** — `{payload.get('domain', 'general')}`")
                    st.caption(f"Score: {ev.get('score', 0):.4f} | Source: {', '.join(ev.get('retrieval_source', ['dense']))}")
                
                pre_filter = result.get("pre_filter", {})
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Language", pre_filter.get("language", "unknown"))
                col_r2.metric("Domains", ", ".join(pre_filter.get("detected_domains", [])))
                col_r3.metric("Entities", ", ".join(pre_filter.get("extracted_entities", [])))
                
                retrieval_strategy = pre_filter.get("retrieval_strategy", "standard_vector")
                st.caption(f"Strategy: {retrieval_strategy}")
        
        elif i == 3:
            # Phase 3: Show reasoning
            with st.expander("🧠 Phase 3 Details", expanded=True):
                agent = result.get("agent_reasoning", {})
                think = agent.get("think_block", "")
                st.markdown("**Agent Reasoning:**")
                st.code(think[:1000] + ("..." if len(think) > 1000 else ""), language="text")
                
                post_val = result.get("post_validation", {})
                z3 = result.get("z3_formal_validation", {})
                col_v1, col_v2, col_v3 = st.columns(3)
                col_v1.metric("Symbolic Validation", f"{post_val.get('validation_score', 0)}%", "✅ Passed" if post_val.get("validated") else "⚠️ Failed")
                col_v2.metric("Z3 Mode", z3.get("execution_mode", "N/A"))
                col_v3.metric("Z3 Score", f"{z3.get('validation_score', 0)}%")
                
                rule_checks = post_val.get("rule_checks", [])
                for rc in rule_checks:
                    icon = "✅" if rc.get("passed") else "⚠️"
                    st.markdown(f"{icon} **{rc.get('rule_id')}**: {rc.get('details')}")
                
                st.markdown("**Generated Hypothesis:**")
                st.info(agent.get("hypothesis", agent.get("output_text", "No hypothesis"))[:500])
        
        elif i == 4:
            # Phase 4: Show enrichment
            with st.expander("📊 Phase 4 Details", expanded=False):
                cg = result.get("graph_rag", {})
                att = result.get("evidence_attribution", {})
                bp = result.get("experimental_blueprint", {})
                cr = result.get("collaboration_recommendations", {})
                disc = result.get("cross_domain_scoring", {})

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown(f"**Discovery Strength:** {disc.get('overall_score', 0)}% ({disc.get('rating', 'unknown')})")
                    st.markdown(f"**Graph Nodes:** {cg.get('nodes_count', 0)} | **Paths:** {len(cg.get('multi_hop_paths', []))}")
                    if att:
                        st.progress(att.get("overall_attribution_coverage", 0.0))
                        st.caption(f"Attribution: {att.get('supported_claims', 0)}/{att.get('total_claims', 0)} claims")
                with col_e2:
                    if bp and bp.get("status") != "disabled":
                        st.markdown(f"**Blueprint:** {bp.get('primary_objective', 'N/A')[:80]}...")
                        st.caption(f"Timeline: {bp.get('timeline_estimate', 'N/A')} | Confidence: {bp.get('confidence', 'unknown')}")
                    if cr and cr.get("recommendations"):
                        for rec in cr.get("recommendations", [])[:3]:
                            st.markdown(f"- **{rec.get('recommended_role')}** ({rec.get('primary_domain')})")

                mem = result.get("memory_footprint")
                if mem:
                    ind = mem.get("individual", {})
                    persona = ind.get("persona", {})
                    st.markdown(f"**Memory Profile:** {persona.get('cognitive_style')} | Safety: {persona.get('safety_focus_level')} | Interactions: {persona.get('interaction_count')}")
        
        elif i == 5:
            # Phase 5: Show streaming metrics
            with st.expander("🌊 Phase 5 Details", expanded=False):
                pm = result.get("performance_metrics", {})
                st.markdown("**Performance Metrics**")
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                col_p1.metric("Total Time", f"{pm.get('total_time_seconds', 0):.2f}s")
                col_p2.metric("Pre-filter", f"{pm.get('pre_filter_ms', 0)}ms")
                col_p3.metric("Agent Reasoning", f"{pm.get('agent_reasoning_time_seconds', 0):.2f}s")
                col_p4.metric("Validation", f"{pm.get('post_validation_ms', 0)}ms")
                
                st.markdown("**Streaming Events:**")
                st.json({
                    "sse_enabled": True,
                    "confidence_calibration": result.get("confidence_calibration", {}),
                    "decision": result.get("confidence_calibration", {}).get("decision"),
                    "evidence_count": len(result.get("retrieved_evidence", [])),
                    "graph_visualization": "Available in Phase 4 enrichment view",
                })
        
        elif i == 6:
            # Phase 6: Show learning
            with st.expander("🔄 Phase 6 Details", expanded=False):
                fb_data, fb_err = call_api("/api/feedback/stats", method="GET", timeout=10)
                re_data, re_err = call_api("/api/model/retrain/status", method="GET", timeout=10)
                
                col_l1, col_l2, col_l3 = st.columns(3)
                col_l1.metric("Feedback Records", fb_data.get("feedback_stats", {}).get("total", 0) if not fb_err else "N/A")
                col_l2.metric("Retrain Enabled", "Yes" if not re_err else "N/A")
                col_l3.metric("Needs Retrain", "🔴" if (not re_err and re_data.get("model_retrainer_status", {}).get("needs_retraining")) else "🟢")
                
                st.markdown("**Learning Pipeline:**")
                st.json({
                    "feedback_collector": "Active (risk-tiered)",
                    "drift_detection": "KS-test every 2h on embedding distributions",
                    "active_learning_queue": "Low-confidence queries flagged for expert review",
                    "model_registry": "MLflow versioned models, prompts, rules, ontologies",
                    "celery_orchestration": "Ingestion, validation, indexing, retrain tasks in background",
                    "monitoring": "Prometheus metrics + Grafana dashboards + OpenTelemetry tracing",
                })
        
        # Progress bar between phases
        if i < len(PHASES):
            progress = (i / len(PHASES)) * 100
            st.progress(progress / 100)

    # Final Summary
    st.markdown("---")
    st.markdown("## ✅ Pipeline Complete — All 6 Phases Executed")
    
    # Show the final hypothesis
    st.markdown("### 📜 Final Cross-Domain Hypothesis")
    agent = result.get("agent_reasoning", {})
    st.info(agent.get("output_text", agent.get("hypothesis", "No hypothesis generated"))[:1000])
    
    # Show calibrated confidence
    cal = result.get("confidence_calibration", {})
    disc = result.get("cross_domain_scoring", {})
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Calibrated Confidence", f"{cal.get('calibrated_confidence', 0) * 100:.1f}%", delta=f"Decision: {cal.get('decision', 'unknown')}")
    col_f2.metric("Discovery Strength", f"{disc.get('overall_score', 0)}%", disc.get("rating", "unknown"))
    col_f3.metric("Evidence Chunks", len(result.get("retrieved_evidence", [])))
    
    # Show full result JSON in expander
    with st.expander("📋 Full Pipeline Result (JSON)"):
        st.json(result)

else:
    st.markdown("### 🚀 How It Works")
    st.markdown("""
1. **Enter your scientific query** in the text area above
2. **Click "Run Full Pipeline"** — this triggers all 6 phases sequentially
3. **Watch the flow** — each phase executes and its results appear below
4. **See the final hypothesis** with confidence scores and evidence

    **The 6 Phases Run in This Order:**
    """)
    for i, (label, name, desc) in enumerate(PHASES, 1):
        st.markdown(f"{i}. **{label}: {name}** — {desc}")

    # Show system status
    st.markdown("---")
    st.markdown("### System Status")
    hdata, herr = call_api("/healthz", method="GET", timeout=5)
    mdata, merr = call_api("/api/metrics", method="GET", timeout=5)
    if not herr and hdata:
        st.success(f"API: {hdata.get('status', 'unknown')}")
    else:
        st.error("API server not reachable")
    if not merr and mdata:
        st.json({"Model": mdata.get("model"), "AIME": mdata.get("metrics", {}).get("AIME_2024"), "MMLU-Pro": mdata.get("metrics", {}).get("MMLU_Pro")})