"""Enterprise confidence calibration for reasoning outputs."""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from reasoning.knowledge_graph import ConfidenceCalibrator as KGConfidenceCalibrator

logger = logging.getLogger("crossmind.calibration")


class CalibrationMethod(str, Enum):
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    TEMPERATURE_SCALING = "temperature_scaling"
    BAYESIAN_BINNING = "bayesian_binning"
    ENSEMBLE = "ensemble"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class Decision(str, Enum):
    PROCEED = "proceed_to_experimental_design"
    INVESTIGATE = "seek_more_evidence"
    REJECT = "do_not_act_without_validation"
    HUMAN_REVIEW = "require_human_review"


@dataclass
class CalibrationResult:
    raw_confidence: float
    calibrated_confidence: float
    confidence_interval: List[float]
    decision: Decision
    method: CalibrationMethod
    thresholds: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_interval": self.confidence_interval,
            "decision": self.decision.value,
            "method": self.method.value,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
        }


@dataclass
class CalibrationConfig:
    method: CalibrationMethod = CalibrationMethod.KNOWLEDGE_GRAPH
    proceed_threshold: float = 0.75
    investigate_threshold: float = 0.50
    human_review_threshold: float = 0.60
    min_confidence_interval_width: float = 0.05
    max_confidence_interval_width: float = 0.30
    enable_human_review: bool = True
    temperature: float = 1.0


class ConfidenceCalibrator:
    """Enterprise-grade confidence calibration with multiple methods."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self._history: deque = deque(maxlen=10000)
        self._lock = threading.RLock()
        self._platt_params: Optional[Dict[str, float]] = None
        self._isotonic_bins: List[tuple] = []
        self._temperature = self.config.temperature

    def calibrate(
        self,
        raw_confidence: float,
        evidence_quality: Optional[float] = None,
        validation_score: Optional[float] = None,
        discovery_score: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> CalibrationResult:
        raw = max(0.0, min(1.0, float(raw_confidence)))
        thresholds = thresholds or {}
        proceed_th = float(thresholds.get("proceed", self.config.proceed_threshold))
        investigate_th = float(thresholds.get("investigate", self.config.investigate_threshold))
        human_review_th = float(thresholds.get("human_review", self.config.human_review_threshold))

        if not 0 <= investigate_th <= proceed_th <= 1:
            raise ValueError("Thresholds must satisfy 0 <= investigate <= proceed <= 1")

        if self.config.method == CalibrationMethod.KNOWLEDGE_GRAPH:
            return self._calibrate_kg(raw, evidence_quality, validation_score, discovery_score, thresholds)
        elif self.config.method == CalibrationMethod.TEMPERATURE_SCALING:
            return self._calibrate_temperature(raw, thresholds)
        elif self.config.method == CalibrationMethod.PLATT_SCALING:
            return self._calibrate_platt(raw, thresholds)
        elif self.config.method == CalibrationMethod.ISOTONIC_REGRESSION:
            return self._calibrate_isotonic(raw, thresholds)
        else:
            return self._calibrate_ensemble(raw, evidence_quality, validation_score, discovery_score, thresholds)

    def _calibrate_kg(
        self,
        raw: float,
        evidence_quality: Optional[float],
        validation_score: Optional[float],
        discovery_score: Optional[float],
        thresholds: Dict[str, float],
    ) -> CalibrationResult:
        discovery = {"overall_score": (discovery_score or 0.5) * 100}
        validation = {"validation_score": (validation_score or 0.5) * 100}
        kg_result = KGConfidenceCalibrator.calibrate(raw, discovery, validation, thresholds)
        calibrated = kg_result["calibrated_confidence"]
        interval = kg_result["confidence_interval"]
        decision_str = kg_result["decision"]

        proceed_th = float(thresholds.get("proceed", self.config.proceed_threshold))
        investigate_th = float(thresholds.get("investigate", self.config.investigate_threshold))
        human_review_th = float(thresholds.get("human_review", self.config.human_review_threshold))

        if self.config.enable_human_review and calibrated >= human_review_th and calibrated < proceed_th:
            decision = Decision.HUMAN_REVIEW
        else:
            decision = Decision(decision_str)

        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_interval=interval,
            decision=decision,
            method=CalibrationMethod.KNOWLEDGE_GRAPH,
            thresholds=thresholds,
            metadata={
                "basis": kg_result["basis"],
                "evidence_quality": evidence_quality,
                "validation_score": validation_score,
                "discovery_score": discovery_score,
            },
        )

    def _calibrate_temperature(
        self,
        raw: float,
        thresholds: Dict[str, float],
    ) -> CalibrationResult:
        calibrated = raw / self._temperature
        calibrated = max(0.0, min(1.0, calibrated))
        spread = self.config.min_confidence_interval_width
        interval = [max(0.0, calibrated - spread), min(1.0, calibrated + spread)]
        decision = self._make_decision(calibrated, thresholds)
        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_interval=interval,
            decision=decision,
            method=CalibrationMethod.TEMPERATURE_SCALING,
            thresholds=thresholds,
            metadata={"temperature": self._temperature},
        )

    def _calibrate_platt(
        self,
        raw: float,
        thresholds: Dict[str, float],
    ) -> CalibrationResult:
        if self._platt_params is None:
            calibrated = raw
        else:
            a = self._platt_params.get("a", 1.0)
            b = self._platt_params.get("b", 0.0)
            calibrated = 1.0 / (1.0 + pow(2.71828, -(a * raw + b)))
        calibrated = max(0.0, min(1.0, calibrated))
        spread = self.config.min_confidence_interval_width
        interval = [max(0.0, calibrated - spread), min(1.0, calibrated + spread)]
        decision = self._make_decision(calibrated, thresholds)
        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_interval=interval,
            decision=decision,
            method=CalibrationMethod.PLATT_SCALING,
            thresholds=thresholds,
            metadata={"platt_params": self._platt_params},
        )

    def _calibrate_isotonic(
        self,
        raw: float,
        thresholds: Dict[str, float],
    ) -> CalibrationResult:
        if not self._isotonic_bins:
            calibrated = raw
        else:
            calibrated = raw
            for bin_edge, bin_value in self._isotonic_bins:
                if raw <= bin_edge:
                    calibrated = bin_value
                    break
        calibrated = max(0.0, min(1.0, calibrated))
        spread = self.config.min_confidence_interval_width
        interval = [max(0.0, calibrated - spread), min(1.0, calibrated + spread)]
        decision = self._make_decision(calibrated, thresholds)
        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_interval=interval,
            decision=decision,
            method=CalibrationMethod.ISOTONIC_REGRESSION,
            thresholds=thresholds,
            metadata={"num_bins": len(self._isotonic_bins)},
        )

    def _calibrate_ensemble(
        self,
        raw: float,
        evidence_quality: Optional[float],
        validation_score: Optional[float],
        discovery_score: Optional[float],
        thresholds: Dict[str, float],
    ) -> CalibrationResult:
        methods = [
            CalibrationMethod.KNOWLEDGE_GRAPH,
            CalibrationMethod.TEMPERATURE_SCALING,
        ]
        results = []
        for method in methods:
            old_method = self.config.method
            self.config.method = method
            try:
                r = self.calibrate(raw, evidence_quality, validation_score, discovery_score, thresholds)
                results.append(r.calibrated_confidence)
            finally:
                self.config.method = old_method

        calibrated = sum(results) / len(results) if results else raw
        spread = self.config.min_confidence_interval_width
        interval = [max(0.0, calibrated - spread), min(1.0, calibrated + spread)]
        decision = self._make_decision(calibrated, thresholds)

        return CalibrationResult(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            confidence_interval=interval,
            decision=decision,
            method=CalibrationMethod.ENSEMBLE,
            thresholds=thresholds,
            metadata={"individual_results": results},
        )

    def _make_decision(self, calibrated: float, thresholds: Dict[str, float]) -> Decision:
        proceed_th = thresholds.get("proceed", self.config.proceed_threshold)
        investigate_th = thresholds.get("investigate", self.config.investigate_threshold)
        human_review_th = thresholds.get("human_review", self.config.human_review_threshold)

        if calibrated >= proceed_th:
            return Decision.PROCEED
        elif self.config.enable_human_review and calibrated >= human_review_th:
            return Decision.HUMAN_REVIEW
        elif calibrated >= investigate_th:
            return Decision.INVESTIGATE
        else:
            return Decision.REJECT

    def update_platt_params(self, a: float, b: float) -> None:
        with self._lock:
            self._platt_params = {"a": a, "b": b}

    def update_isotonic_bins(self, bins: List[tuple]) -> None:
        with self._lock:
            self._isotonic_bins = sorted(bins, key=lambda x: x[0])

    def update_temperature(self, temperature: float) -> None:
        with self._lock:
            self._temperature = max(0.1, temperature)

    def record_outcome(self, raw_confidence: float, calibrated_confidence: float, correct: bool) -> None:
        with self._lock:
            self._history.append({
                "raw": raw_confidence,
                "calibrated": calibrated_confidence,
                "correct": correct,
            })

    def get_calibration_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {"samples": 0}
            total = len(self._history)
            correct = sum(1 for h in self._history if h["correct"])
            avg_raw = sum(h["raw"] for h in self._history) / total
            avg_cal = sum(h["calibrated"] for h in self._history) / total
            return {
                "samples": total,
                "accuracy": correct / total,
                "avg_raw_confidence": avg_raw,
                "avg_calibrated_confidence": avg_cal,
                "calibration_error": abs(avg_cal - (correct / total)),
            }


_calibrator_instance: Optional[ConfidenceCalibrator] = None
_calibrator_lock = threading.Lock()


def get_confidence_calibrator(config: Optional[CalibrationConfig] = None) -> ConfidenceCalibrator:
    global _calibrator_instance
    if _calibrator_instance is None:
        with _calibrator_lock:
            if _calibrator_instance is None:
                _calibrator_instance = ConfidenceCalibrator(config)
    return _calibrator_instance


def calibrate_confidence(
    raw_confidence: float,
    evidence_quality: Optional[float] = None,
    validation_score: Optional[float] = None,
    discovery_score: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    calibrator = get_confidence_calibrator(config)
    return calibrator.calibrate(raw_confidence, evidence_quality, validation_score, discovery_score, thresholds)