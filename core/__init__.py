"""Core runtime package for CrossMind.

This package groups the pieces that define the system's runtime behavior:
- application configuration
- orchestration and reasoning core
- engine factories used by the API and dashboards
"""

from config import Settings, settings
from reasoning.multi_agent import MultiAgentOrchestrator, get_multi_agent_orchestrator
from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline, get_neuro_symbolic_pipeline
from reasoning.strategy_layer import QualityGate, UnifiedRouter, CostController

__all__ = [
    "Settings",
    "settings",
    "MultiAgentOrchestrator",
    "get_multi_agent_orchestrator",
    "NeuroSymbolicPipeline",
    "get_neuro_symbolic_pipeline",
    "UnifiedRouter",
    "QualityGate",
    "CostController",
]
