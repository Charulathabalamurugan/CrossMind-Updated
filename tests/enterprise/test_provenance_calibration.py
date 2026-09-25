"""Tests for enterprise provenance, versioning, and calibration modules."""
import unittest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("QDRANT_IN_MEMORY", "true")
os.environ.setdefault("NEO4J_ENABLED", "false")

from reasoning.provenance import (
    ProvenanceTracker,
    ProvenanceEventType,
    ProvenanceEvent,
    ProvenanceChain,
    get_provenance_tracker,
    record_provenance_event,
    start_provenance_chain,
    end_provenance_chain,
    _current_chain,
)
from reasoning.versioning import (
    VersionRegistry,
    VersionComponent,
    VersionInfo,
    CompatibilityMatrix,
    get_version_registry,
    get_component_version,
    is_version_deprecated,
    check_version_compatibility,
)
from reasoning.calibration import (
    ConfidenceCalibrator,
    CalibrationConfig,
    CalibrationMethod,
    CalibrationResult,
    Decision,
    get_confidence_calibrator,
    calibrate_confidence,
)


class TestProvenanceTracker(unittest.TestCase):
    def setUp(self):
        from reasoning.provenance import _chains, _chains_lock
        with _chains_lock:
            _chains.clear()
        _current_chain.set(None)

    def test_singleton_tracker(self):
        tracker1 = get_provenance_tracker()
        tracker2 = get_provenance_tracker()
        self.assertIs(tracker1, tracker2)

    def test_start_end_chain(self):
        tracker = get_provenance_tracker()
        chain = tracker.start_chain(correlation_id="corr-123", tenant_id="tenant-1", user_id="user-1")
        self.assertIsNotNone(chain.chain_id)
        self.assertEqual(chain.metadata["correlation_id"], "corr-123")
        self.assertEqual(chain.metadata["tenant_id"], "tenant-1")
        self.assertEqual(chain.metadata["user_id"], "user-1")

        ended = tracker.end_chain()
        self.assertEqual(ended.chain_id, chain.chain_id)
        self.assertIsNone(tracker.get_current_chain())

    def test_record_event(self):
        tracker = get_provenance_tracker()
        chain = tracker.start_chain()
        event = tracker.record_event(
            ProvenanceEventType.QUERY_RECEIVED,
            "query_processing",
            {"query": "test query", "length": 10},
        )
        self.assertEqual(event.event_type, ProvenanceEventType.QUERY_RECEIVED)
        self.assertEqual(event.pipeline_stage, "query_processing")
        self.assertEqual(event.details["query"], "test query")
        self.assertIsNotNone(event.chain_id)
        self.assertEqual(event.chain_id, chain.chain_id)

        chain_obj = tracker.get_current_chain()
        self.assertEqual(len(chain_obj.events), 1)
        tracker.end_chain()

    def test_module_functions(self):
        chain = start_provenance_chain(correlation_id="corr-456")
        self.assertEqual(chain.metadata["correlation_id"], "corr-456")

        event = record_provenance_event(
            ProvenanceEventType.EVIDENCE_RETRIEVED,
            "retrieval",
            {"evidence_count": 5},
        )
        self.assertEqual(event.event_type, ProvenanceEventType.EVIDENCE_RETRIEVED)

        ended = end_provenance_chain()
        self.assertEqual(ended.chain_id, chain.chain_id)

    def test_chain_retrieval(self):
        tracker = get_provenance_tracker()
        chain = tracker.start_chain(correlation_id="corr-789", tenant_id="tenant-2")
        tracker.record_event(ProvenanceEventType.QUERY_RECEIVED, "stage1", {})
        tracker.end_chain()

        retrieved = tracker.get_chain(chain.chain_id)
        self.assertIsNotNone(retrieved)

        chains = tracker.get_chains_by_correlation("corr-789")
        self.assertEqual(len(chains), 1)

        chains = tracker.get_chains_by_tenant("tenant-2")
        self.assertEqual(len(chains), 1)

    def test_export_chain(self):
        tracker = get_provenance_tracker()
        chain = tracker.start_chain()
        tracker.record_event(ProvenanceEventType.REASONING_STEP, "reasoning", {"step": 1})
        tracker.end_chain()

        exported = tracker.export_chain(chain.chain_id)
        self.assertIsNotNone(exported)
        self.assertEqual(exported["chain_id"], chain.chain_id)
        self.assertEqual(len(exported["events"]), 1)

    def test_concurrent_chains(self):
        tracker = get_provenance_tracker()
        results = []

        def worker(corr_id):
            tracker.start_chain(correlation_id=corr_id)
            tracker.record_event(ProvenanceEventType.QUERY_RECEIVED, "stage", {"worker": corr_id})
            time.sleep(0.01)
            chain = tracker.end_chain()
            results.append(chain.chain_id)

        threads = [threading.Thread(target=worker, args=(f"corr-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 5)
        self.assertEqual(len(set(results)), 5)

    def test_chain_cleanup(self):
        tracker = ProvenanceTracker(max_chains=2)
        tracker.start_chain(correlation_id="old")
        time.sleep(0.01)
        tracker.end_chain()

        tracker.start_chain(correlation_id="new")
        tracker.end_chain()

        removed = tracker.clear_old_chains(max_age_seconds=0)
        self.assertGreaterEqual(removed, 1)


class TestVersionRegistry(unittest.TestCase):
    def setUp(self):
        from reasoning.versioning import _version_registry, _compatibility_matrix
        import reasoning.versioning as vr_module
        with threading.Lock():
            _version_registry.clear()
            _compatibility_matrix.clear()
        vr_module._registry_instance = None

    def test_singleton_registry(self):
        reg1 = get_version_registry()
        reg2 = get_version_registry()
        self.assertIs(reg1, reg2)

    def test_default_versions_registered(self):
        registry = get_version_registry()
        for component in VersionComponent:
            latest = registry.get_latest_version(component)
            self.assertIsNotNone(latest, f"No version registered for {component.value}")
            self.assertEqual(latest.component, component)

    def test_register_custom_version(self):
        registry = get_version_registry()
        info = VersionInfo(
            component=VersionComponent.MODEL,
            version="2.0.0",
            description="Custom model version",
            metadata={"author": "test"},
        )
        registry.register_version(info)

        retrieved = registry.get_version(VersionComponent.MODEL, "2.0.0")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.description, "Custom model version")
        self.assertEqual(retrieved.metadata["author"], "test")

    def test_list_versions(self):
        registry = get_version_registry()
        registry.register_version(VersionInfo(
            component=VersionComponent.PIPELINE,
            version="3.1.0",
            description="Pipeline v3.1",
        ))
        versions = registry.list_versions(VersionComponent.PIPELINE)
        self.assertGreaterEqual(len(versions), 2)

    def test_deprecate_version(self):
        registry = get_version_registry()
        registry.register_version(VersionInfo(
            component=VersionComponent.DATA_SCHEMA,
            version="5.0.0",
            description="Old schema",
        ))
        self.assertTrue(registry.deprecate_version(VersionComponent.DATA_SCHEMA, "5.0.0", "6.0.0"))

        info = registry.get_version(VersionComponent.DATA_SCHEMA, "5.0.0")
        self.assertTrue(info.deprecated)
        self.assertEqual(info.replacement_version, "6.0.0")

    def test_get_component_version(self):
        version = get_component_version(VersionComponent.MODEL)
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "unknown")

    def test_is_version_deprecated(self):
        registry = get_version_registry()
        registry.register_version(VersionInfo(
            component=VersionComponent.API_SCHEMA,
            version="1.0.0",
            description="Deprecated API",
        ))
        registry.deprecate_version(VersionComponent.API_SCHEMA, "1.0.0")

        self.assertTrue(is_version_deprecated(VersionComponent.API_SCHEMA, "1.0.0"))
        self.assertFalse(is_version_deprecated(VersionComponent.API_SCHEMA, "999.0.0"))

    def test_compatibility_matrix(self):
        registry = get_version_registry()
        matrix = CompatibilityMatrix(
            component=VersionComponent.PIPELINE,
            from_version="1.0.0",
            to_version="2.0.0",
            compatible=True,
            migration_required=True,
            migration_notes="Migrate config format",
            breaking_changes=["config_format"],
        )
        registry.register_compatibility(matrix)

        checked = registry.check_compatibility(VersionComponent.PIPELINE, "1.0.0", "2.0.0")
        self.assertIsNotNone(checked)
        self.assertTrue(checked.compatible)
        self.assertTrue(checked.migration_required)

    def test_check_version_compatibility(self):
        registry = get_version_registry()
        registry.register_compatibility(CompatibilityMatrix(
            component=VersionComponent.MODEL,
            from_version="1.0.0",
            to_version="2.0.0",
            compatible=True,
        ))
        self.assertTrue(check_version_compatibility(VersionComponent.MODEL, "1.0.0", "2.0.0"))

        registry.register_compatibility(CompatibilityMatrix(
            component=VersionComponent.MODEL,
            from_version="1.0.0",
            to_version="3.0.0",
            compatible=False,
        ))
        self.assertFalse(check_version_compatibility(VersionComponent.MODEL, "1.0.0", "3.0.0"))


class TestConfidenceCalibration(unittest.TestCase):
    def setUp(self):
        from reasoning.calibration import _calibrator_instance
        ConfidenceCalibrator._instance = None

    def test_singleton_calibrator(self):
        cal1 = get_confidence_calibrator()
        cal2 = get_confidence_calibrator()
        self.assertIs(cal1, cal2)

    def test_knowledge_graph_calibration(self):
        config = CalibrationConfig(method=CalibrationMethod.KNOWLEDGE_GRAPH)
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(
            raw_confidence=0.9,
            evidence_quality=0.8,
            validation_score=0.7,
            discovery_score=0.75,
        )
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.method, CalibrationMethod.KNOWLEDGE_GRAPH)
        self.assertGreaterEqual(result.calibrated_confidence, 0.0)
        self.assertLessEqual(result.calibrated_confidence, 1.0)
        self.assertIn(result.decision, [Decision.PROCEED, Decision.INVESTIGATE, Decision.REJECT, Decision.HUMAN_REVIEW])

    def test_temperature_scaling(self):
        config = CalibrationConfig(method=CalibrationMethod.TEMPERATURE_SCALING, temperature=2.0)
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(raw_confidence=0.8)
        self.assertEqual(result.method, CalibrationMethod.TEMPERATURE_SCALING)
        self.assertAlmostEqual(result.calibrated_confidence, 0.4, places=1)
        self.assertEqual(result.metadata["temperature"], 2.0)

    def test_platt_scaling(self):
        config = CalibrationConfig(method=CalibrationMethod.PLATT_SCALING)
        calibrator = ConfidenceCalibrator(config)
        calibrator.update_platt_params(a=1.0, b=0.0)

        result = calibrator.calibrate(raw_confidence=0.5)
        self.assertEqual(result.method, CalibrationMethod.PLATT_SCALING)
        self.assertIn("platt_params", result.metadata)

    def test_isotonic_regression(self):
        config = CalibrationConfig(method=CalibrationMethod.ISOTONIC_REGRESSION)
        calibrator = ConfidenceCalibrator(config)
        calibrator.update_isotonic_bins([(0.3, 0.2), (0.6, 0.5), (1.0, 0.9)])

        result = calibrator.calibrate(raw_confidence=0.5)
        self.assertEqual(result.method, CalibrationMethod.ISOTONIC_REGRESSION)
        self.assertEqual(result.calibrated_confidence, 0.5)

    def test_ensemble_calibration(self):
        config = CalibrationConfig(method=CalibrationMethod.ENSEMBLE)
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(
            raw_confidence=0.8,
            evidence_quality=0.7,
            validation_score=0.6,
            discovery_score=0.75,
        )
        self.assertEqual(result.method, CalibrationMethod.ENSEMBLE)
        self.assertIn("individual_results", result.metadata)

    def test_decision_thresholds(self):
        config = CalibrationConfig(
            proceed_threshold=0.8,
            investigate_threshold=0.5,
            human_review_threshold=0.65,
            enable_human_review=True,
        )
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(0.9, evidence_quality=0.9, validation_score=0.9, discovery_score=0.9, thresholds={"proceed": 0.8, "investigate": 0.5})
        self.assertEqual(result.decision, Decision.PROCEED)

        result = calibrator.calibrate(0.7, evidence_quality=0.8, validation_score=0.8, discovery_score=0.8, thresholds={"proceed": 0.8, "investigate": 0.5})
        self.assertEqual(result.decision, Decision.HUMAN_REVIEW)

        result = calibrator.calibrate(0.6, evidence_quality=0.5, validation_score=0.5, discovery_score=0.5, thresholds={"proceed": 0.8, "investigate": 0.5})
        self.assertEqual(result.decision, Decision.INVESTIGATE)

        result = calibrator.calibrate(0.3, evidence_quality=0.5, validation_score=0.5, discovery_score=0.5, thresholds={"proceed": 0.8, "investigate": 0.5})
        self.assertEqual(result.decision, Decision.REJECT)

    def test_human_review_disabled(self):
        config = CalibrationConfig(enable_human_review=False, proceed_threshold=0.75, investigate_threshold=0.5)
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(0.65)
        self.assertEqual(result.decision, Decision.INVESTIGATE)

    def test_calibrate_confidence_function(self):
        result = calibrate_confidence(
            raw_confidence=0.85,
            evidence_quality=0.8,
            validation_score=0.7,
            discovery_score=0.75,
        )
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.method, CalibrationMethod.KNOWLEDGE_GRAPH)

    def test_record_outcome_and_stats(self):
        calibrator = get_confidence_calibrator()
        calibrator.record_outcome(0.8, 0.75, True)
        calibrator.record_outcome(0.6, 0.55, False)
        calibrator.record_outcome(0.9, 0.85, True)

        stats = calibrator.get_calibration_stats()
        self.assertEqual(stats["samples"], 3)
        self.assertAlmostEqual(stats["accuracy"], 2/3)
        self.assertIn("calibration_error", stats)

    def test_confidence_interval_bounds(self):
        config = CalibrationConfig(min_confidence_interval_width=0.1)
        calibrator = ConfidenceCalibrator(config)

        result = calibrator.calibrate(0.5)
        interval = result.confidence_interval
        self.assertEqual(len(interval), 2)
        self.assertGreaterEqual(interval[0], 0.0)
        self.assertLessEqual(interval[1], 1.0)
        self.assertGreaterEqual(interval[1] - interval[0], 0.1)

    def test_invalid_thresholds(self):
        config = CalibrationConfig()
        calibrator = ConfidenceCalibrator(config)

        with self.assertRaises(ValueError):
            calibrator.calibrate(0.5, thresholds={"proceed": 0.5, "investigate": 0.8})

    def test_to_dict(self):
        result = CalibrationResult(
            raw_confidence=0.8,
            calibrated_confidence=0.75,
            confidence_interval=[0.7, 0.8],
            decision=Decision.PROCEED,
            method=CalibrationMethod.KNOWLEDGE_GRAPH,
            thresholds={"proceed": 0.75},
        )
        d = result.to_dict()
        self.assertEqual(d["raw_confidence"], 0.8)
        self.assertEqual(d["decision"], "proceed_to_experimental_design")
        self.assertEqual(d["method"], "knowledge_graph")


class TestIntegration(unittest.TestCase):
    def test_provenance_with_calibration(self):
        from reasoning.provenance import get_provenance_tracker, ProvenanceEventType
        from reasoning.calibration import get_confidence_calibrator

        tracker = get_provenance_tracker()
        calibrator = get_confidence_calibrator()

        tracker.start_chain(correlation_id="integration-1")
        result = calibrator.calibrate(0.85, evidence_quality=0.8, validation_score=0.75)

        tracker.record_event(
            ProvenanceEventType.CALIBRATION_APPLIED,
            "calibration",
            {"raw": result.raw_confidence, "calibrated": result.calibrated_confidence, "decision": result.decision.value},
        )
        tracker.end_chain()

        chain = tracker.get_current_chain()
        self.assertIsNone(chain)

    def test_versioning_with_provenance(self):
        from reasoning.versioning import get_version_registry, VersionComponent
        from reasoning.provenance import get_provenance_tracker, ProvenanceEventType

        registry = get_version_registry()
        tracker = get_provenance_tracker()

        tracker.start_chain()
        version = get_component_version(VersionComponent.MODEL)

        tracker.record_event(
            ProvenanceEventType.REASONING_STEP,
            "version_check",
            {"model_version": version},
        )
        tracker.end_chain()

        self.assertIsInstance(version, str)


if __name__ == "__main__":
    unittest.main()