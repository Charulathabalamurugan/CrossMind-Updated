import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from config import settings

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


class Neo4jGraph:
    """Neo4j-backed graph wrapper with an in-memory fallback for local/dev usage."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI", getattr(settings, "NEO4J_URI", "bolt://localhost:7687"))
        self.user = user or os.getenv("NEO4J_USER", getattr(settings, "NEO4J_USER", "neo4j"))
        self.password = password or os.getenv("NEO4J_PASSWORD", getattr(settings, "NEO4J_PASSWORD", "neo4jpass"))
        self.driver = None
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._connect()

    def _connect(self):
        if not getattr(settings, "NEO4J_ENABLED", False) or GraphDatabase is None:
            return
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self._session() as session:
                session.run("RETURN 1")
        except Exception:
            self.driver = None

    @contextmanager
    def _session(self):
        session = self.driver.session(database="neo4j")
        try:
            enter = session.__enter__
            exit_ = session.__exit__
        except AttributeError:
            enter = exit_ = None

        if enter is not None and exit_ is not None:
            with session as managed_session:
                yield managed_session
            return

        try:
            yield session
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                close()

    def index_documents(self, documents: Iterable[Dict[str, Any]]) -> None:
        documents = list(documents)
        if self.driver is not None:
            try:
                with self._session() as session:
                    for document in documents:
                        doc_id = str(document.get("id") or document.get("title") or "doc")
                        title = document.get("title", "Untitled")
                        domain = document.get("domain", "general")
                        session.run(
                            "MERGE (d:Document {id: $id}) SET d.title = $title, d.domain = $domain",
                            id=doc_id,
                            title=title,
                            domain=domain,
                        )
                return
            except Exception:
                self.driver = None

        for document in documents:
            doc_id = str(document.get("id") or document.get("title") or "doc")
            self.nodes[doc_id] = document

    def graph_rag_context(self, evidence: List[Dict[str, Any]], query_entities: List[str]) -> Dict[str, Any]:
        paths: List[Dict[str, Any]] = []
        if self.driver is not None:
            try:
                with self._session() as session:
                    rows = session.run(
                        "MATCH (d:Document)-[:MENTIONS]->(e:Entity) WHERE d.id IN $ids RETURN d.title AS title, collect(e.name) AS entities LIMIT 20",
                        ids=[str(item.get("id")) for item in evidence],
                    )
                    for row in rows:
                        paths.append({
                            "path": [row["title"]],
                            "bridge_entity": (row["entities"] or ["general"])[0],
                            "cross_domain": False,
                            "path_score": 0.0,
                        })
                return {
                    "strategy": "neo4j_graph_rag",
                    "seed_document_ids": [str(item.get("id")) for item in evidence],
                    "multi_hop_paths": paths,
                }
            except Exception:
                self.driver = None

        for item in evidence:
            payload = item.get("payload") or {}
            title = payload.get("title") or "Untitled"
            paths.append({
                "path": [title],
                "bridge_entity": query_entities[0] if query_entities else "general",
                "cross_domain": False,
                "path_score": 0.0,
            })
        return {
            "strategy": "in_memory_fallback",
            "seed_document_ids": [str(item.get("id")) for item in evidence],
            "multi_hop_paths": paths,
        }


def get_neo4j_store() -> Neo4jGraph:
    return Neo4jGraph()


__all__ = ["Neo4jGraph", "get_neo4j_store"]
