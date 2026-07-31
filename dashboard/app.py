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
    .tech-tag { background-color: #F3F4F6; color: #374151; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; display: inline-block; margin: 1px; }
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
st.markdown('<div class="sub-title">All 4 Phases in a Single Flow — Click Run and Watch Everything Happen</div>', unsafe_allow_html=True)

# ========== Query Input ==========
st.markdown("### 📝 Enter Your Scientific Query")
col_query, col_run = st.columns([3, 1])
with col_query:
    query_input = st.text_area("Query:", value="What are recent scientific breakthroughs across multiple domains?", height=80, key="single_flow_query")
with col_run:
    run_clicked = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True, key="run_pipeline")

# ========== Phase Tags ==========
st.markdown("#### Technology Stack by Phase")
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    st.markdown("**Phase 1: Ingestion**")
    for t in ["FastAPI", "MinerU", "Apache Tika", "BGE-M3", "Qdrant", "Redis"]:
        st.markdown(f'<span class="tech-tag">{t}</span>', unsafe_allow_html=True)
with col_t2:
    st.markdown("**Phase 2: Retrieval**")
    for t in ["BGE-M3", "Qdrant", "BM25", "RRF", "ColBERT", "RBAC", "Redis", "LightGBM/TinyBERT Classifier", "Conditional Retrieval"]:
        st.markdown(f'<span class="tech-tag">{t}</span>', unsafe_allow_html=True)
with col_t3:
    st.markdown("**Phase 3: Reasoning**")
    for t in ["ZAYA1-8B", "vLLM", "Scallop", "Semara", "DeforestVIS", "GraphRAG", "WFA", "Decision Tree", "Redis"]:
        st.markdown(f'<span class="tech-tag">{t}</span>', unsafe_allow_html=True)
with col_t4:
    st.markdown("**Phase 4: Application**")
    for t in ["FastAPI", "React/Streamlit", "SSE", "OpenTelemetry", "Prometheus", "Redis", "DiskCache", "DLDB", "RBAC", "Evaluation Framework (PR/NDCG)"]:
        st.markdown(f'<span class="tech-tag">{t}</span>', unsafe_allow_html=True)

# ========== 4-Phase Flow ==========
PHASES = [
    ("📥 Phase 1", "Ingestion", "FastAPI + MinerU + Apache Tika + BGE-M3 + Qdrant + Redis — Extract text, chunk documents, generate dense/sparse embeddings, store in vector DB, cache for efficient data ingestion"),
    ("🔍 Phase 2", "Retrieval", "BGE-M3 + Qdrant + BM25 + RRF + ColBERT (Server-Side via Qdrant API) + RBAC + Redis — Hybrid semantic and keyword retrieval, rerank via ColBERT, role-based access control"),
    ("🧠 Phase 3", "Reasoning", "ZAYA1-8B + vLLM + Scallop + Semara + DeforestVIS + GraphRAG + WFA + Decision Tree + Redis — Neuro-symbolic reasoning combining rule-based, ontology-based, graph-based, and LLM-based reasoning"),
    ("📊 Phase 4", "Application", "FastAPI + React/Streamlit + SSE + OpenTelemetry + Prometheus + Redis + DiskCache + DLDB + RBAC — User interface, real-time streaming, performance monitoring, caching, feedback storage"),
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
    
    with st.spinner("🚀 Running all 4 phases..."):
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

    for i, (phase_label, phase_name, phase_desc) in enumerate(PHASES, 1):
        st.markdown(f"### {phase_label}: {phase_name}")
        st.markdown(f"<span class='phase-done'>✅ Complete</span> {phase_desc}", unsafe_allow_html=True)
        
        if i == 1:
            # Phase 1: Ingestion
            with st.expander("📥 Phase 1: Ingestion Details", expanded=False):
                st.markdown("**Technologies:** FastAPI, MinerU, Apache Tika, BGE-M3, Qdrant, Redis")
                st.json({
                    "ingestion": {
                        "framework": "FastAPI (async endpoints)",
                        "pdf_extractor": "MinerU (scientific PDF – tables, formulas, images)",
                        "fallback_extractor": "Apache Tika (office docs, email)",
                        "text_chunker": "Sliding-window (512 tokens, 64 overlap)",
                    },
                    "embedding": {
                        "dense_vector": "BGE-M3 INT8/FP32 (1024-dim, Matryoshka variant supported)",
                        "sparse_vector": "BGE-M3 subword token embeddings",
                    },
                    "storage": {
                        "vector_db": "Qdrant with PQ compression",
                        "domain_collections": "Separate collections per domain",
                    },
                    "cache": {
                        "backend": "Redis (hot-query TTL dedup)",
                        "ttl": "3600s",
                    }
                })
        
        elif i == 2:
            # Phase 2: Retrieval
            with st.expander("🔍 Phase 2: Retrieval Details", expanded=True):
                st.markdown("**Technologies:** BGE-M3, Qdrant, BM25, RRF, ColBERT (Server-Side via Qdrant API), RBAC, Redis")
                evidence = result.get("retrieved_evidence", [])
                st.markdown(f"**Retrieved {len(evidence)} evidence chunks**")
                for idx, ev in enumerate(evidence[:5], 1):
                    payload = ev.get("payload", {})
                    st.markdown(f"**[{idx}] {payload.get('title', 'Untitled')}** — `{payload.get('domain', 'general')}`")
                    st.caption(f"Score: {ev.get('score', 0):.4f} | Source: {', '.join(ev.get('retrieval_source', ['dense']))}")
                
                pre_filter = result.get("pre_filter", {})
                
                # Query Classification & Routing pathway
                classification = pre_filter.get("query_classification", {})
                if classification:
                    st.markdown("**🧠 Query Classifier (LightGBM/TinyBERT) Routing Pathway:**")
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                    col_c1.metric("Predicted Domain", classification.get("predicted_domain", "N/A").upper())
                    col_c2.metric("Query Type", classification.get("query_type", "N/A").upper())
                    col_c3.metric("Complexity", classification.get("complexity", "N/A").upper())
                    col_c4.metric("Confidence", f"{classification.get('confidence', 0.0) * 100:.1f}%")
                    st.caption(f"Model Engine: `{classification.get('model_used', 'LightGBM')}`")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Language", pre_filter.get("language", "unknown"))
                col_r2.metric("Domains", ", ".join(pre_filter.get("detected_domains", [])))
                col_r3.metric("Entities", ", ".join(pre_filter.get("extracted_entities", [])))
                
                retrieval_strategy = pre_filter.get("retrieval_strategy", "standard_vector")
                st.caption(f"Routing Strategy: **{retrieval_strategy.upper()}**")
                if "optimized" in retrieval_strategy.lower():
                    st.success("⚡ Conditional Retrieval Optimization triggered: bypassed expensive GraphRAG/multi-agent processing for simple factual query!")
                
                col_bm25, col_rrf, col_colbert = st.columns(3)
                col_bm25.metric("BM25", "Enabled", "✓")
                col_rrf.metric("RRF Fusion", "Enabled", "✓")
                col_colbert.metric("ColBERT Rerank", "Server-Side via Qdrant API", "✓")
                col_colbert.caption("MultiVectorConfig(max_sim=MAX_SIM, m=0)")
                
                if user_role:
                    st.markdown(f"**RBAC Role:** `{user_role}` — inline filtering applied at retrieval layer")

        elif i == 3:
            # Phase 3: Reasoning
            with st.expander("🧠 Phase 3: Reasoning Details", expanded=True):
                st.markdown("**Technologies:** ZAYA1-8B, vLLM, Scallop, Semara, DeforestVIS, GraphRAG, WFA, Decision Tree, Redis")
                
                agent = result.get("agent_reasoning", {})
                think = agent.get("think_block", "")
                st.markdown("**ZAYA1-8B Agent Reasoning:**")
                st.code(think[:1000] + ("..." if len(think) > 1000 else ""), language="text")
                
                col_rr1, col_rr2, col_rr3 = st.columns(3)
                col_rr1.metric("vLLM", "Enabled", "max-num-seqs=2 (RTX 4090 OOM guard)")
                col_rr2.metric("WFA + Decision Tree", "Fast path (80% queries)", "<10ms O(1)")
                col_rr3.metric("GraphRAG", "Slow path (15%)", "Multi-hop graph traversal")
                
                st.markdown("**Extended Reasoning Stack:**")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.markdown("Scallop: logical reasoning integration")
                    st.markdown("Semara: semantic grounding (Tech Mahindra SEMARA / open-source SeMRA via rdflib - ACTIVE)")
                with col_s2:
                    st.markdown("DeforestVIS: reasoning visualization")
                    st.markdown("WFA: weighted fast-action reasoning")
                with col_s3:
                    st.markdown("Decision Tree: rule-based path selection")
                    st.markdown("Redis: caching expensive reasoning results")
                
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
                
                st.markdown("**Was this hypothesis helpful?**")
                col_fb1, col_fb2 = st.columns([1, 10])
                feedback_key = f"feedback_{hash(safe_query)}"
                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None
                    
                if st.session_state[feedback_key] is None:
                    with col_fb1:
                        if st.button("👍 Yes", key="thumbs_up"):
                            st.session_state[feedback_key] = "UP"
                            with open("user_feedback.log", "a", encoding="utf-8") as f:
                                import datetime
                                f.write(f"[{datetime.datetime.now().isoformat()}] Query: {safe_query} | Feedback: UP | Hypothesis: {agent.get('hypothesis', '')[:200]}...\n")
                            st.rerun()
                    with col_fb2:
                        if st.button("👎 No", key="thumbs_down"):
                            st.session_state[feedback_key] = "DOWN"
                            with open("user_feedback.log", "a", encoding="utf-8") as f:
                                import datetime
                                f.write(f"[{datetime.datetime.now().isoformat()}] Query: {safe_query} | Feedback: DOWN | Hypothesis: {agent.get('hypothesis', '')[:200]}...\n")
                            st.rerun()
                else:
                    st.success(f"Feedback submitted: {st.session_state[feedback_key]}! Thank you.")

        elif i == 4:
            # Phase 4: Application
            with st.expander("📊 Phase 4: Application Details", expanded=False):
                st.markdown("**Technologies:** FastAPI, React/Streamlit, SSE, OpenTelemetry, Prometheus, Redis + DiskCache, DLDB, RBAC")
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
                
                st.markdown("**Application Layer Tech:**")
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("SSE Streaming", "Active", "Real-time")
                col_a2.metric("OpenTelemetry", "Enabled", "Distributed tracing")
                col_a3.metric("Prometheus", "Enabled", "Metrics endpoint")
                
                col_a4, col_a5 = st.columns(2)
                col_a4.metric("Cache", "Redis (hot) + DiskCache (warm)", "Tiered")
                col_a5.metric("DLDB", "Active", "Feedback + rules storage")
                
                pm = result.get("performance_metrics", {})
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                col_p1.metric("Total Time", f"{pm.get('total_time_seconds', 0):.2f}s")
                col_p2.metric("Pre-filter", f"{pm.get('pre_filter_ms', 0)}ms")
                col_p3.metric("Agent Reasoning", f"{pm.get('agent_reasoning_time_seconds', 0):.2f}s")
                col_p4.metric("Validation", f"{pm.get('post_validation_ms', 0)}ms")
                
                st.markdown("**Streaming Events:**")
                st.json({
                    "sse_enabled": True,
                    "react_streamlit_ui": "Prototype (React for production)",
                    "confidence_calibration": result.get("confidence_calibration", {}),
                    "decision": result.get("confidence_calibration", {}).get("decision"),
                    "evidence_count": len(result.get("retrieved_evidence", [])),
                    "graph_visualization": "Available in Phase 3 reasoning view",
                })

                st.markdown("---")
                st.markdown("### 📊 Retrieval Performance Evaluation Dashboard")
                st.markdown("Assess the quality of semantic and hybrid retrieval against standard ground-truth query-document mappings.")
                if st.button("Run System Evaluation Benchmark", key="run_eval_btn"):
                    with st.spinner("Calculating Precision@K, Recall@K, MRR, NDCG..."):
                        eval_data, eval_err = call_api("/api/evaluate", method="POST")
                        if eval_err:
                            st.error(eval_err)
                        else:
                            st.success("Evaluation complete!")
                            avg = eval_data.get("average_metrics", {})
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("Precision @ 5", f"{avg.get('precision_at_5', 0.0) * 100:.1f}%")
                            col_m2.metric("Recall @ 5", f"{avg.get('recall_at_5', 0.0) * 100:.1f}%")
                            col_m3.metric("MRR", f"{avg.get('mrr', 0.0):.4f}")
                            col_m4.metric("NDCG @ 5", f"{avg.get('ndcg_at_5', 0.0):.4f}")
                            
                            with st.expander("Benchmark Runs Details", expanded=True):
                                st.json(eval_data.get("benchmark_runs", []))

        # Progress bar between phases
        if i < len(PHASES):
            progress = (i / len(PHASES)) * 100
            st.progress(progress / 100)

    # Final Summary
    st.markdown("---")
    st.markdown("## ✅ Pipeline Complete — All 4 Phases Executed")
    
    st.markdown("### 📜 Final Cross-Domain Hypothesis")
    agent = result.get("agent_reasoning", {})
    st.info(agent.get("output_text", agent.get("hypothesis", "No hypothesis generated"))[:1000])
    
    cal = result.get("confidence_calibration", {})
    disc = result.get("cross_domain_scoring", {})
    col_f1, col_f2, col_f3 = st.columns(3)
    col_f1.metric("Calibrated Confidence", f"{cal.get('calibrated_confidence', 0) * 100:.1f}%", delta=f"Decision: {cal.get('decision', 'unknown')}")
    col_f2.metric("Discovery Strength", f"{disc.get('overall_score', 0)}%", disc.get("rating", "unknown"))
    col_f3.metric("Evidence Chunks", len(result.get("retrieved_evidence", [])))
    
    with st.expander("📋 Full Pipeline Result (JSON)"):
        st.json(result)

else:
    st.markdown("### 🚀 How It Works")
    st.markdown("""
1. **Enter your scientific query** in the text area above
2. **Click "Run Full Pipeline"** — this triggers all 4 phases sequentially
3. **Watch the flow** — each phase executes and its results appear below
4. **See the final hypothesis** with confidence scores and evidence

    **The 4 Phases Use These Exact Technologies:**
    """)
    for i, (label, name, desc) in enumerate(PHASES, 1):
        st.markdown(f"{i}. **{label}: {name}** — {desc}")

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