import logging
from typing import List, Dict, Any, Optional
from config import settings

logger = logging.getLogger("crossmind.neo4j")

class Neo4jGraphStore:
    def __init__(self):
        self.enabled = settings.NEO4J_ENABLED
        self.driver = None
        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except ImportError:
            logger.info("neo4j driver not installed. Graph store disabled.")
            self.enabled = False
        except Exception as exc:
            logger.warning(f"Neo4j connection failed: {exc}. Graph store disabled.")
            self.enabled = False

    def add_node(self, label: str, properties: Dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"MERGE (n:{label} $props) RETURN id(n) as nodeId",
                    props=properties,
                )
                record = result.single()
                if record:
                    return str(record["nodeId"])
        except Exception as exc:
            logger.error(f"Failed to add node: {exc}")
        return ""

    def add_edge(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any] = None):
        if not self.enabled:
            return
        try:
            with self.driver.session() as session:
                session.run(
                    f"MATCH (a) WHERE id(a) = $from_id "
                    f"MATCH (b) WHERE id(b) = $to_id "
                    f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props",
                    from_id=int(from_id), to_id=int(to_id), props=properties or {},
                )
        except Exception as exc:
            logger.error(f"Failed to add edge: {exc}")

    def query_walk(self, start_label: str, depth: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"MATCH path = (start:{start_label})-[*1..{depth}]-(end) "
                    f"RETURN path LIMIT 100"
                )
                paths = []
                for record in result:
                    path = record["path"]
                    nodes = [dict(n) for n in path.nodes]
                    edges = [dict(r) for r in path.relationships]
                    paths.append({"nodes": nodes, "edges": edges})
                return paths
        except Exception as exc:
            logger.error(f"Neo4j query failed: {exc}")
            return []

    def get_graph_context(self, entity_ids: List[str], max_depth: int = 3) -> Dict[str, Any]:
        paths = self.query_walk(entity_ids[0], max_depth) if entity_ids else []
        return {
            "neo4j_enabled": self.enabled,
            "paths_found": len(paths),
            "paths": paths[:settings.GRAPH_RAG_DEPTH],
            "nodes_count": len(set(
                n for p in paths for n in p.get("nodes", [])
            )),
        }

neo4j_store = None

def get_neo4j_store() -> Optional[Neo4jGraphStore]:
    global neo4j_store
    if neo4j_store is None:
        neo4j_store = Neo4jGraphStore()
    return neo4j_store