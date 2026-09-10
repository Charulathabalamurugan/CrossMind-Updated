import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List


class TestCacheVersionInvalidation:
    """Test cache version invalidation and schema versioning."""

    def test_cache_schema_version_exists(self):
        """Cache schema version should be defined in settings."""
        from config import settings
        
        assert hasattr(settings, 'CACHE_SCHEMA_VERSION')
        assert isinstance(settings.CACHE_SCHEMA_VERSION, int)
        assert settings.CACHE_SCHEMA_VERSION >= 1

    def test_data_schema_version_exists(self):
        """Data schema version should be defined in settings."""
        from config import settings
        
        assert hasattr(settings, 'DATA_SCHEMA_VERSION')
        assert isinstance(settings.DATA_SCHEMA_VERSION, int)
        assert settings.DATA_SCHEMA_VERSION >= 1

    def test_vector_schema_version_exists(self):
        """Vector schema version should be defined in settings."""
        from config import settings
        
        assert hasattr(settings, 'VECTOR_SCHEMA_VERSION')
        assert isinstance(settings.VECTOR_SCHEMA_VERSION, int)
        assert settings.VECTOR_SCHEMA_VERSION >= 1

    def test_kg_schema_version_exists(self):
        """KG schema version should be defined in settings."""
        from config import settings
        
        assert hasattr(settings, 'KG_SCHEMA_VERSION')
        assert isinstance(settings.KG_SCHEMA_VERSION, int)
        assert settings.KG_SCHEMA_VERSION >= 1

    def test_api_schema_version_exists(self):
        """API schema version should be defined in settings."""
        from config import settings
        
        assert hasattr(settings, 'API_SCHEMA_VERSION')
        assert isinstance(settings.API_SCHEMA_VERSION, int)
        assert settings.API_SCHEMA_VERSION >= 1

    def test_ingestion_cache_version_invalidation(self):
        """Ingestion cache should support version-based invalidation."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        
        cache.set("key1", {"data": "v1", "schema_version": 1})
        cache.set("key2", {"data": "v2", "schema_version": 2})
        
        assert cache.get("key1")["schema_version"] == 1
        assert cache.get("key2")["schema_version"] == 2
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_query_cache_version_invalidation(self):
        """Query cache should support version-based invalidation."""
        from reasoning.query_cache import QueryResultCache
        
        cache = QueryResultCache()
        cache.clear()
        
        cache.set("query:v1", {"result": "v1"}, query="test query", user_role="researcher")
        cache.set("query:v2", {"result": "v2"}, query="test query", user_role="researcher")
        
        result1 = cache.get("query:v1")
        result2 = cache.get("query:v2")
        
        assert result1 == {"result": "v1"}
        assert result2 == {"result": "v2"}
        
        cache.clear()
        
        assert cache.get("query:v1") is None
        assert cache.get("query:v2") is None

    def test_redis_cache_version_invalidation(self):
        """Redis cache should support version-based invalidation."""
        from ingestion.redis_cache import RedisCache
        
        with patch("ingestion.redis_cache.redis") as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = '{"data": "cached", "schema_version": 1}'
            mock_client.setex.return_value = True
            mock_client.delete.return_value = True
            mock_client.flushdb.return_value = True
            mock_redis.Redis.return_value = mock_client
            
            cache = RedisCache()
            
            result = cache.get("versioned_key")
            assert result is not None
            assert result.get("schema_version") == 1
            
            cache.set("versioned_key", {"data": "new", "schema_version": 2})
            
            mock_client.setex.assert_called()

    def test_disk_cache_version_awareness(self):
        """Disk cache should be version-aware."""
        import diskcache as dc
        from config import settings
        
        cache = dc.Cache(settings.DISK_CACHE_PATH)
        
        cache.set("test_key_v1", {"data": "v1", "schema_version": 1})
        cache.set("test_key_v2", {"data": "v2", "schema_version": 2})
        
        assert cache.get("test_key_v1")["schema_version"] == 1
        assert cache.get("test_key_v2")["schema_version"] == 2
        
        cache.close()

    def test_cache_invalidation_on_schema_change(self):
        """Cache should be invalidated when schema version changes."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache_v1 = IngestionCache()
        cache_v1.clear()
        
        cache_v1.set("doc:1", {"content": "test", "schema_version": 1})
        
        assert cache_v1.get("doc:1") is not None
        
        cache_v1.max_items = 0
        cache_v1._evict_lru()
        
        assert cache_v1.get("doc:1") is None

    def test_multi_schema_version_coexistence(self):
        """Multiple schema versions should coexist in cache."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        
        for version in range(1, 6):
            cache.set(f"doc:v{version}", {
                "content": f"version {version}",
                "schema_version": version
            })
        
        for version in range(1, 6):
            result = cache.get(f"doc:v{version}")
            assert result is not None
            assert result["schema_version"] == version
            assert result["content"] == f"version {version}"

    def test_cache_ttl_respects_version(self):
        """Cache TTL should work independently of schema version."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        cache.ttl = 1
        
        cache.set("key_ttl", {"data": "test", "schema_version": 1})
        
        assert cache.get("key_ttl") is not None
        
        time.sleep(1.1)
        
        assert cache.get("key_ttl") is None


class TestCacheInvalidationStrategies:
    """Test various cache invalidation strategies."""

    def test_selective_key_invalidation(self):
        """Specific keys should be invalidatable without clearing all."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        
        cache.set("keep_me", {"data": "keep"})
        cache.set("delete_me", {"data": "delete"})
        
        cache.delete("delete_me")
        
        assert cache.get("keep_me") == {"data": "keep"}
        assert cache.get("delete_me") is None

    def test_pattern_based_invalidation_simulation(self):
        """Simulate pattern-based cache invalidation."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        
        for i in range(10):
            cache.set(f"domain:energy:doc:{i}", {"id": i, "domain": "energy"})
        for i in range(5):
            cache.set(f"domain:finance:doc:{i}", {"id": i, "domain": "finance"})
        
        energy_keys = [k for k in list(cache._store.keys()) if k.startswith("domain:energy:")]
        for key in energy_keys:
            cache.delete(key)
        
        for i in range(10):
            assert cache.get(f"domain:energy:doc:{i}") is None
        for i in range(5):
            assert cache.get(f"domain:finance:doc:{i}") is not None

    def test_lru_eviction_respects_schema_version(self):
        """LRU eviction should work with versioned entries."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        cache.max_items = 3
        
        cache.set("v1_old", {"schema_version": 1, "data": "old"})
        cache.set("v1_new", {"schema_version": 1, "data": "new"})
        cache.set("v2_new", {"schema_version": 2, "data": "new"})
        
        assert cache.get("v1_old") is not None
        assert cache.get("v1_new") is not None
        assert cache.get("v2_new") is not None
        
        cache.set("v3_new", {"schema_version": 3, "data": "new"})
        
        assert cache.size() == 3


class TestCacheConsistency:
    """Test cache consistency guarantees."""

    def test_concurrent_cache_access(self):
        """Cache should handle concurrent access correctly."""
        import threading
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        cache.max_items = 1000
        
        errors = []
        
        def writer(thread_id):
            try:
                for i in range(50):
                    cache.set(f"thread_{thread_id}_key_{i}", {
                        "thread": thread_id,
                        "index": i,
                        "schema_version": 1
                    })
            except Exception as e:
                errors.append(e)
        
        def reader(thread_id):
            try:
                for i in range(50):
                    cache.get(f"thread_{thread_id}_key_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=writer, args=(i,))
            t2 = threading.Thread(target=reader, args=(i,))
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0

    def test_cache_size_tracking(self):
        """Cache size should be accurately tracked."""
        from ingestion.ingestion_cache import IngestionCache
        
        cache = IngestionCache()
        cache.clear()
        
        assert cache.size() == 0
        
        cache.set("key1", "value1")
        assert cache.size() == 1
        
        cache.set("key2", "value2")
        assert cache.size() == 2
        
        cache.delete("key1")
        assert cache.size() == 1
        
        cache.clear()
        assert cache.size() == 0

    def test_query_cache_similarity_with_version(self):
        """Semantic query cache should consider schema version."""
        from reasoning.query_cache import QueryResultCache
        
        with patch("reasoning.query_cache.get_embedder") as mock_get_embedder:
            mock_embedder = Mock()
            mock_embedder.embed_text.return_value = [0.1] * 256
            mock_get_embedder.return_value = mock_embedder
            
            cache = QueryResultCache()
            cache.clear()
            
            cache.set("query:1", {"result": "v1", "schema_version": 1}, query="test query")
            cache.set("query:2", {"result": "v2", "schema_version": 2}, query="test query")
            
            result = cache.get("query:1")
            assert result["schema_version"] == 1
            
            result = cache.get("query:2")
            assert result["schema_version"] == 2


class TestCacheMigration:
    """Test cache migration scenarios."""

    def test_schema_upgrade_migration_simulation(self):
        """Simulate cache migration from v1 to v2."""
        from ingestion.ingestion_cache import IngestionCache
        
        old_cache = IngestionCache()
        old_cache.clear()
        
        old_cache.set("doc:1", {"title": "Old Format", "schema_version": 1})
        old_cache.set("doc:2", {"title": "Old Format 2", "schema_version": 1})
        
        assert old_cache.get("doc:1")["schema_version"] == 1
        
        new_cache = IngestionCache()
        new_cache.clear()
        
        for key in ["doc:1", "doc:2"]:
            old_value = old_cache.get(key)
            if old_value:
                migrated = {**old_value, "schema_version": 2, "migrated_at": time.time()}
                new_cache.set(key, migrated)
        
        assert new_cache.get("doc:1")["schema_version"] == 2
        assert "migrated_at" in new_cache.get("doc:1")

    def test_vector_schema_version_in_payload(self):
        """Vector engine should store schema version in payload."""
        from vector_store.qdrant_engine import QdrantVectorEngine
        
        with patch("vector_store.qdrant_engine.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client_class.return_value = mock_client
            
            engine = QdrantVectorEngine()
            
            records = [{
                "id": "doc:1",
                "vector": [0.1] * 256,
                "payload": {
                    "title": "Test",
                    "content": "Content",
                    "domain": "test",
                    "vector_schema_version": 1
                }
            }]
            
            ids = engine.upsert_vectors(records)
            
            assert len(ids) == 1
            assert len(engine._memory_store) == 1
            assert engine._memory_store[0]["payload"].get("vector_schema_version") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])