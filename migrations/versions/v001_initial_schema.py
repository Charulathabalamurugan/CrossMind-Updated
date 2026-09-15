"""Initial schema migration for Qdrant collections, Neo4j constraints, and Redis keyspaces."""

from migrations.runner import Migration


def up() -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest_models
        from config import settings

        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        collections = [c.name for c in client.get_collections().collections]
        if settings.QDRANT_COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=rest_models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=rest_models.Distance.COSINE,
                ),
            )
    except Exception as exc:
        raise RuntimeError(f"Qdrant initial schema migration failed: {exc}")

    try:
        from neo4j import GraphDatabase
        from config import settings

        if settings.NEO4J_ENABLED:
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD.get_secret_value()),
            )
            with driver.session() as session:
                session.run(
                    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX document_domain IF NOT EXISTS FOR (d:Document) ON (d.domain)"
                )
                session.run(
                    "CREATE INDEX document_year IF NOT EXISTS FOR (d:Document) ON (d.year)"
                )
            driver.close()
    except Exception as exc:
        raise RuntimeError(f"Neo4j initial schema migration failed: {exc}")


def down() -> None:
    try:
        from qdrant_client import QdrantClient
        from config import settings

        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        client.delete_collection(settings.QDRANT_COLLECTION_NAME)
    except Exception:
        pass


migration = Migration(
    version=1,
    description="Initial schema: Qdrant collection, Neo4j constraints",
    up=up,
    down=down,
)
