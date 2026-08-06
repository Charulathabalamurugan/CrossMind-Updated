# Phase 1: Ingestion Package

__all__ = [
    "get_ingestion_pipeline",
    "get_dynamic_connectors",
    "get_ingestion_cache",
    "get_active_learning_engine",
]


def get_ingestion_pipeline():
    from ingestion.pipeline import get_ingestion_pipeline as _get_ingestion_pipeline
    return _get_ingestion_pipeline()


def get_dynamic_connectors():
    from ingestion.dynamic_connectors import get_dynamic_connectors as _get_dynamic_connectors
    return _get_dynamic_connectors()


def get_ingestion_cache():
    from ingestion.ingestion_cache import get_ingestion_cache as _get_ingestion_cache
    return _get_ingestion_cache()


def get_active_learning_engine():
    from ingestion.active_learning import get_active_learning_engine as _get_active_learning_engine
    return _get_active_learning_engine()
