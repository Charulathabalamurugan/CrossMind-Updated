import unittest


class TestRuntimeCompatibility(unittest.TestCase):
    def test_core_missing_runtime_modules_import(self):
        modules = [
            "ingestion.mineru_extractor",
            "ingestion.redis_cache",
            "reasoning.neo4j_graph",
            "reasoning.dldb",
            "reasoning.conflict_detector",
            "reasoning.deepseek_agent",
            "reasoning.abductive_engine_enhanced",
            "reasoning.gliner_extractor",
            "reasoning.datalog_engine",
            "reasoning.opa_enforcer",
            "reasoning.drift_detector",
            "reasoning.treeinterpreter",
            "reasoning.prometheus_monitor",
            "reasoning.mlflow_registry",
        ]

        for module_name in modules:
            __import__(module_name)

    def test_core_runtime_apis_are_callable(self):
        from ingestion.mineru_extractor import MinerUExtractor
        from ingestion.redis_cache import get_redis_cache
        from reasoning.conflict_detector import ConflictDetector
        from reasoning.prometheus_monitor import get_prometheus_monitor
        from reasoning.dldb import get_dldb

        extractor = MinerUExtractor()
        self.assertTrue(hasattr(extractor, "extract"))

        cache = get_redis_cache()
        self.assertTrue(hasattr(cache, "get") and hasattr(cache, "set"))

        detector = ConflictDetector()
        self.assertTrue(hasattr(detector, "detect_conflicts"))

        monitor = get_prometheus_monitor()
        self.assertTrue(hasattr(monitor, "record_metric"))

        dldb = get_dldb()
        self.assertTrue(hasattr(dldb, "record_event"))


if __name__ == "__main__":
    unittest.main()
