"""Add payload indexes and HNSW configuration for Qdrant."""

from migrations.runner import Migration


def up() -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest_models
        from config import settings

        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        client.update_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            optimizers_config=rest_models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="domain",
            field_schema=rest_models.PayloadSchemaSchema.KEYWORD,
        )
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="year",
            field_schema=rest_models.PayloadSchemaSchema.INTEGER,
        )
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            field_name="tags",
            field_schema=rest_models.PayloadSchemaSchema.KEYWORD,
        )
    except Exception as exc:
        raise RuntimeError(f"Qdrant vector indexes migration failed: {exc}")


def down() -> None:
    pass


migration = Migration(
    version=2,
    description="Add Qdrant payload indexes",
    up=up,
    down=down,
)
