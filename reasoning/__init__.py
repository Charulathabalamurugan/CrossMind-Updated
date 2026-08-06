# Phase 1: Retrieval Package

__all__ = [
    "get_wfa_engine",
    "get_query_classifier",
    "get_neuro_symbolic_pipeline",
    "get_multi_agent_orchestrator",
    "get_hybrid_rag_kg",
    "get_dual_memory",
    "get_z3_validator",
    "get_experimental_blueprint_generator",
    "get_evidence_attributor",
    "get_risk_feedback_engine",
    "get_collaboration_recommender",
    "get_sparse_retriever",
    "get_query_cache",
    "get_hypothesis_generator",
    "get_rule_engine",
    "get_bridge_scorer",
    "format_pipeline_result",
    "get_benchmark_collector",
    "get_feedback_collector",
    "get_model_retrainer",
    "get_rule_updater",
    "get_gliner_extractor",
    "get_datalog_engine",
    "get_opa_enforcer",
    "get_neo4j_store",
    "get_zaya1_8b_agent",
    "get_enhanced_abductive_engine",
    "get_dldb",
    "get_drift_detector",
    "get_tree_interpreter",
    "get_prometheus_monitor",
    "get_model_registry",
]


def get_wfa_engine():
    from reasoning.wfa_fast_path import get_wfa_engine as _get_wfa_engine
    return _get_wfa_engine()


def get_query_classifier():
    from reasoning.query_classifier import get_query_classifier as _get_query_classifier
    return _get_query_classifier()


def get_neuro_symbolic_pipeline():
    from reasoning.neuro_symbolic_pipeline import get_neuro_symbolic_pipeline as _get_neuro_symbolic_pipeline
    return _get_neuro_symbolic_pipeline()


def get_multi_agent_orchestrator():
    from reasoning.multi_agent import get_multi_agent_orchestrator as _get_multi_agent_orchestrator
    return _get_multi_agent_orchestrator()


def get_hybrid_rag_kg():
    from reasoning.hybrid_rag_kg import get_hybrid_rag_kg as _get_hybrid_rag_kg
    return _get_hybrid_rag_kg()


def get_dual_memory():
    from reasoning.dual_memory import get_dual_memory as _get_dual_memory
    return _get_dual_memory()


def get_z3_validator():
    from reasoning.z3_validator import get_z3_validator as _get_z3_validator
    return _get_z3_validator()


def get_experimental_blueprint_generator():
    from reasoning.experimental_blueprint import get_experimental_blueprint_generator as _get_experimental_blueprint_generator
    return _get_experimental_blueprint_generator()


def get_evidence_attributor():
    from reasoning.evidence_attribution import get_evidence_attributor as _get_evidence_attributor
    return _get_evidence_attributor()


def get_risk_feedback_engine():
    from reasoning.risk_feedback import get_risk_feedback_engine as _get_risk_feedback_engine
    return _get_risk_feedback_engine()


def get_collaboration_recommender():
    from reasoning.collaboration_recommender import get_collaboration_recommender as _get_collaboration_recommender
    return _get_collaboration_recommender()


def get_sparse_retriever():
    from reasoning.sparse_retriever import get_sparse_retriever as _get_sparse_retriever
    return _get_sparse_retriever()


def get_query_cache():
    from reasoning.query_cache import get_query_cache as _get_query_cache
    return _get_query_cache()


def get_hypothesis_generator():
    from reasoning.hypothesis_generator import get_hypothesis_generator as _get_hypothesis_generator
    return _get_hypothesis_generator()


def get_rule_engine():
    from reasoning.rule_engine import get_rule_engine as _get_rule_engine
    return _get_rule_engine()


def get_bridge_scorer():
    from reasoning.bridge_scorer import get_bridge_scorer as _get_bridge_scorer
    return _get_bridge_scorer()


def format_pipeline_result(*args, **kwargs):
    from reasoning.result_formatter import format_pipeline_result as _format_pipeline_result
    return _format_pipeline_result(*args, **kwargs)


def get_benchmark_collector():
    from reasoning.benchmark_collector import get_benchmark_collector as _get_benchmark_collector
    return _get_benchmark_collector()


def get_feedback_collector():
    from reasoning.feedback_collector import get_feedback_collector as _get_feedback_collector
    return _get_feedback_collector()


def get_model_retrainer():
    from reasoning.retrainer import get_model_retrainer as _get_model_retrainer
    return _get_model_retrainer()


def get_rule_updater():
    from reasoning.rule_updater import get_rule_updater as _get_rule_updater
    return _get_rule_updater()


def get_gliner_extractor():
    from reasoning.gliner_extractor import get_gliner_extractor as _get_gliner_extractor
    return _get_gliner_extractor()


def get_datalog_engine():
    from reasoning.datalog_engine import get_datalog_engine as _get_datalog_engine
    return _get_datalog_engine()


def get_opa_enforcer():
    from reasoning.opa_enforcer import get_opa_enforcer as _get_opa_enforcer
    return _get_opa_enforcer()


def get_neo4j_store():
    from reasoning.neo4j_graph import get_neo4j_store as _get_neo4j_store
    return _get_neo4j_store()


def get_zaya1_8b_agent():
    from reasoning.deepseek_agent import get_zaya1_8b_agent as _get_zaya1_8b_agent
    return _get_zaya1_8b_agent()


def get_enhanced_abductive_engine():
    from reasoning.abductive_engine_enhanced import get_enhanced_abductive_engine as _get_enhanced_abductive_engine
    return _get_enhanced_abductive_engine()


def get_dldb():
    from reasoning.dldb import get_dldb as _get_dldb
    return _get_dldb()


def get_drift_detector():
    from reasoning.drift_detector import get_drift_detector as _get_drift_detector
    return _get_drift_detector()


def get_tree_interpreter():
    from reasoning.treeinterpreter import get_tree_interpreter as _get_tree_interpreter
    return _get_tree_interpreter()


def get_prometheus_monitor():
    from reasoning.prometheus_monitor import get_prometheus_monitor as _get_prometheus_monitor
    return _get_prometheus_monitor()


def get_model_registry():
    from reasoning.mlflow_registry import get_model_registry as _get_model_registry
    return _get_model_registry()