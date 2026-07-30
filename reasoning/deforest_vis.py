import json
import logging
from typing import Dict, Any, List
from config import settings

logger = logging.getLogger("crossmind.deforest_vis")

class DeforestVIS:
    def __init__(self, port: int = 8003):
        self.port = port
        self.enabled = settings.DEFORESTVIS_ENABLED
        self.visualization_data: Dict[str, Any] = {}
        self.reasoning_paths: List[Dict[str, Any]] = []

    def render_reasoning_graph(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        agent_result: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []
        for i, ev in enumerate(evidence[:10]):
            node_id = f"evidence_{i}"
            nodes.append({
                "id": node_id,
                "label": ev.get("payload", {}).get("title", "Untitled")[:50],
                "type": "evidence",
                "score": ev.get("score", 0),
            })
        agent_node = {
            "id": "agent",
            "label": agent_result.get("hypothesis", agent_result.get("output_text", "Unknown"))[:80],
            "type": "agent",
        }
        nodes.append(agent_node)

        self.visualization_data = {"nodes": nodes, "edges": edges}
        self.reasoning_paths.append({
            "query": query[:100],
            "nodes_count": len(nodes),
            "edges_count": len(edges),
        })
        return {"nodes": nodes, "edges": edges, "port": self.port}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "port": self.port,
            "total_visualizations": len(self.reasoning_paths),
            "last_path": self.reasoning_paths[-1] if self.reasoning_paths else None,
        }