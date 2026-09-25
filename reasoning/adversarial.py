import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("crossmind.adversarial")


class ThreatType(Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DOCUMENT_POISONING = "document_poisoning"
    EMBEDDING_MANIPULATION = "embedding_manipulation"
    MALICIOUS_URL = "malicious_url"
    MALICIOUS_SCRIPT = "malicious_script"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    POLICY_VIOLATION = "policy_violation"


class RiskLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: "RiskLevel") -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.value >= other.value


class Action(Enum):
    ALLOW = "allow"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    ALERT = "alert"


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    threat_type: ThreatType
    risk_score: float
    risk_level: RiskLevel
    action: Action
    content_hash: str
    details: Dict[str, Any]
    request_id: str


@dataclass
class DetectionResult:
    threat_type: ThreatType
    risk_score: float
    risk_level: RiskLevel
    matched_patterns: List[str]
    indicators: List[str]
    confidence: float


@dataclass
class DefenseResult:
    request_id: str
    original_content: str
    sanitized_content: str
    overall_risk_score: float
    overall_risk_level: RiskLevel
    detections: List[DetectionResult]
    actions: List[Action]
    quarantine_decision: bool
    audit_events: List[AuditEvent]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternMatcher:
    PROMPT_INJECTION_PATTERNS = [
        r"(?i)ignore\s+(?:previous|above|all)(?:\s+(?:previous|above))?\s+(?:instructions?|prompts?|rules?)",
        r"(?i)disregard\s+(?:previous|above|all)(?:\s+(?:previous|above))?\s+(?:instructions?|prompts?|rules?)",
        r"(?i)forget\s+(?:previous|above|all)(?:\s+(?:previous|above))?\s+(?:instructions?|prompts?|rules?)",
        r"(?i)new\s+(?:instructions?|prompt|rules?)\s*:",
        r"(?i)system\s*:",
        r"(?i)assistant\s*:",
        r"(?i)<\|im_start\|>",
        r"(?i)<\|im_end\|>",
        r"(?i)\[INST\]",
        r"(?i)\[/INST\]",
        r"(?i)###\s*(?:Instruction|System|User|Assistant)\s*:",
        r"(?i)role\s*:\s*(?:system|assistant)",
        r"(?i)you\s+are\s+(?:now|an?)\s+",
        r"(?i)pretend\s+(?:to\s+be|you\s+are)",
        r"(?i)simulate\s+(?:a|an)\s+",
        r"(?i)act\s+as\s+(?:a|an)\s+",
        r"(?i)as\s+a\s+(?:language\s+model|AI)\s*,?\s*i\s+(?:cannot|won't|will\s+not)",
        r"(?i)override\s+(?:safety|security|guardrails?)",
        r"(?i)bypass\s+(?:safety|security|filter|guardrails?)",
    ]

    JAILBREAK_PATTERNS = [
        r"(?i)DAN\s*(?:mode|prompt)?",
        r"(?i)do\s+anything\s+now",
        r"(?i)developer\s+mode",
        r"(?i)unrestricted\s+(?:mode|AI)",
        r"(?i)no\s+(?:restrictions?|limits?|rules?)",
        r"(?i)free\s+(?:mode|AI)",
        r"(?i)evil\s+(?:mode|AI|assistant)",
        r"(?i)harmful\s+(?:content|response)",
        r"(?i)generate\s+(?:harmful|illegal|dangerous)",
        r"(?i)how\s+to\s+(?:make|create|build)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison|virus|malware)",
        r"(?i)instructions?\s+(?:for|on)\s+(?:making|creating|building)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison|virus|malware)",
        r"(?i)suicide|self-harm|self\s+harm",
        r"(?i)child\s+(?:sexual|abuse|exploitation)",
        r"(?i)terrorist|terrorism",
    ]

    DOCUMENT_POISONING_INDICATORS = [
        r"(?i)ignore\s+previous\s+instructions",
        r"(?i)system\s+prompt\s+leak",
        r"(?i)reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
        r"(?i)what\s+(?:is|are)\s+your\s+(?:instructions?|rules?|guidelines?)",
        r"(?i)output\s+(?:your|the)\s+(?:system\s+)?prompt",
        r"(?i)print\s+(?:your|the)\s+(?:system\s+)?prompt",
        r"(?i)show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?prompt",
        r"(?i)repeat\s+(?:your|the)\s+(?:system\s+)?prompt",
        r"(?i)base64|encoded|obfuscated",
        r"(?i)hidden\s+(?:instruction|prompt|message)",
        r"(?i)injected\s+(?:content|instruction|prompt)",
    ]

    MALICIOUS_URL_PATTERNS = [
        r"(?i)https?://(?:[a-z0-9-]+\.)*(?:bit\.ly|tinyurl|t\.co|goo\.gl|short\.link|cutt\.ly|is\.gd|v\.gd|tiny\.cc|short\.io|rebrand\.ly|clck\.ru)",
        r"(?i)https?://(?:[a-z0-9-]+\.)*(?:pastebin|ghostbin|hastebin|rentry|txt\.git\.io|raw\.githubusercontent)\.",
        r"(?i)https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        r"(?i)https?://(?:[a-z0-9-]+\.)*\.onion/",
        r"(?i)javascript:",
        r"(?i)data:(?:text|image|application)/",
        r"(?i)file://",
        r"(?i)ftp://",
    ]

    MALICIOUS_SCRIPT_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"(?i)on\w+\s*=",
        r"(?i)javascript:",
        r"(?i)eval\s*\(",
        r"(?i)Function\s*\(",
        r"(?i)setTimeout\s*\(",
        r"(?i)setInterval\s*\(",
        r"(?i)document\.(?:write|cookie|location)",
        r"(?i)window\.(?:location|open)",
        r"(?i)<\s*iframe",
        r"(?i)<\s*object",
        r"(?i)<\s*embed",
        r"(?i)expression\s*\(",
        r"(?i)@import",
        r"(?i)behavior\s*:",
        r"(?i)vbscript:",
    ]

    RESOURCE_EXHAUSTION_PATTERNS = [
        r"(?i)repeat\s+(?:this|the\s+following)\s+(?:100|1000|10000|\d{4,})\s+times?",
        r"(?i)generate\s+(?:a\s+)?(?:100|1000|10000|\d{4,})\s+(?:words?|tokens?|characters?|paragraphs?)",
        r"(?i)write\s+(?:a\s+)?(?:100|1000|10000|\d{4,})\s+page",
        r"(?i)infinite\s+(?:loop|recursion)",
        r"(?i)while\s+true\s*:",
        r"(?i)for\s*\(\s*;\s*;\s*\)",
        r"(?i)allocate\s+(?:large|huge|massive)\s+(?:memory|array|buffer)",
        r"(?i)crash\s+(?:the\s+)?(?:system|server|process)",
        r"(?i)denial\s+of\s+service|DoS|DDoS",
    ]

    POLICY_VIOLATION_PATTERNS = [
        r"(?i)hate\s+speech",
        r"(?i)discriminat(?:e|ion|ory)",
        r"(?i)harass(?:ment|ing)",
        r"(?i)violence|violent",
        r"(?i)illegal\s+(?:activity|activities|act)",
        r"(?i)copyright\s+(?:infringement|violation)",
        r"(?i)piracy|pirated",
        r"(?i)PII|personally\s+identifiable\s+information",
        r"(?i)SSN|social\s+security\s+number",
        r"(?i)credit\s+card|CCV|CVV",
        r"(?i)password|secret|api[_-]?key|token",
        r"(?i)private\s+key",
    ]

    def __init__(self):
        self._compiled_patterns: Dict[ThreatType, List[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        pattern_map = {
            ThreatType.PROMPT_INJECTION: self.PROMPT_INJECTION_PATTERNS,
            ThreatType.JAILBREAK: self.JAILBREAK_PATTERNS,
            ThreatType.DOCUMENT_POISONING: self.DOCUMENT_POISONING_INDICATORS,
            ThreatType.MALICIOUS_URL: self.MALICIOUS_URL_PATTERNS,
            ThreatType.MALICIOUS_SCRIPT: self.MALICIOUS_SCRIPT_PATTERNS,
            ThreatType.RESOURCE_EXHAUSTION: self.RESOURCE_EXHAUSTION_PATTERNS,
            ThreatType.POLICY_VIOLATION: self.POLICY_VIOLATION_PATTERNS,
        }
        for threat_type, patterns in pattern_map.items():
            self._compiled_patterns[threat_type] = [re.compile(p) for p in patterns]

    def match(self, content: str, threat_type: ThreatType) -> Tuple[List[str], List[str]]:
        matched = []
        indicators = []
        for pattern in self._compiled_patterns.get(threat_type, []):
            matches = pattern.findall(content)
            if matches:
                matched.append(pattern.pattern)
                indicators.extend(matches if isinstance(matches[0], str) else [m[0] if isinstance(m, tuple) else m for m in matches])
        return matched, indicators


class EmbeddingAnalyzer:
    def __init__(self, outlier_threshold: float = 3.0, window_size: int = 100):
        self._outlier_threshold = outlier_threshold
        self._window_size = window_size
        self._embedding_history: List[np.ndarray] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._lock = None

    def analyze(self, embedding: np.ndarray) -> Tuple[float, List[str]]:
        if embedding is None or embedding.size == 0:
            return 0.0, []

        indicators = []
        anomaly_score = 0.0

        # Check for NaN/Inf first
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            anomaly_score = 5.0
            indicators.append("embedding_nan_inf_detected")
            # Still add to history but return early with max score
            self._embedding_history.append(embedding.copy())
            if len(self._embedding_history) > self._window_size:
                self._embedding_history.pop(0)
            return min(anomaly_score, 5.0), indicators

        norm = float(np.linalg.norm(embedding))
        if norm > 1000:
            anomaly_score = max(anomaly_score, 2.0)
            indicators.append(f"embedding_large_norm:{norm:.2f}")
        elif norm < 1e-6:
            anomaly_score = max(anomaly_score, 1.5)
            indicators.append(f"embedding_near_zero_norm:{norm:.2e}")

        if len(self._embedding_history) >= 10:
            self._update_stats()
            if self._mean is not None and self._std is not None:
                z_scores = np.abs((embedding - self._mean) / (self._std + 1e-8))
                max_z = float(np.max(z_scores))
                mean_z = float(np.mean(z_scores))
                
                if max_z > self._outlier_threshold:
                    anomaly_score = max(anomaly_score, min(max_z / self._outlier_threshold, 5.0))
                    indicators.append(f"embedding_outlier_max_z:{max_z:.2f}")
                elif mean_z > self._outlier_threshold / 2:
                    anomaly_score = max(anomaly_score, min(mean_z / (self._outlier_threshold / 2), 3.0))
                    indicators.append(f"embedding_outlier_mean_z:{mean_z:.2f}")

        self._embedding_history.append(embedding.copy())
        if len(self._embedding_history) > self._window_size:
            self._embedding_history.pop(0)

        return min(anomaly_score, 5.0), indicators

    def _update_stats(self) -> None:
        if len(self._embedding_history) < 2:
            return
        stacked = np.stack(self._embedding_history)
        self._mean = np.mean(stacked, axis=0)
        self._std = np.std(stacked, axis=0)


class AdversarialDefenseController:
    DEFAULT_WEIGHTS = {
        ThreatType.PROMPT_INJECTION: 2.0,
        ThreatType.JAILBREAK: 3.0,
        ThreatType.DOCUMENT_POISONING: 2.0,
        ThreatType.EMBEDDING_MANIPULATION: 2.0,
        ThreatType.MALICIOUS_URL: 1.5,
        ThreatType.MALICIOUS_SCRIPT: 2.5,
        ThreatType.RESOURCE_EXHAUSTION: 2.0,
        ThreatType.POLICY_VIOLATION: 2.0,
    }

    RISK_THRESHOLDS = {
        RiskLevel.NONE: 0.0,
        RiskLevel.LOW: 0.5,
        RiskLevel.MEDIUM: 1.5,
        RiskLevel.HIGH: 3.0,
        RiskLevel.CRITICAL: 4.5,
    }

    def __init__(
        self,
        weights: Optional[Dict[ThreatType, float]] = None,
        risk_thresholds: Optional[Dict[RiskLevel, float]] = None,
        embedding_analyzer: Optional[EmbeddingAnalyzer] = None,
        sanitize_mode: bool = True,
        quarantine_threshold: RiskLevel = RiskLevel.HIGH,
        block_threshold: RiskLevel = RiskLevel.CRITICAL,
    ):
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._risk_thresholds = risk_thresholds or self.RISK_THRESHOLDS.copy()
        self._pattern_matcher = PatternMatcher()
        self._embedding_analyzer = embedding_analyzer or EmbeddingAnalyzer()
        self._sanitize_mode = sanitize_mode
        self._quarantine_threshold = quarantine_threshold
        self._block_threshold = block_threshold
        self._audit_log: List[AuditEvent] = []
        _audit_lock = threading.Lock() if (threading := __import__('threading')) else None
        self._lock = _audit_lock
        self._request_count = 0
        self._blocked_count = 0
        self._quarantined_count = 0

    def _compute_risk_level(self, score: float) -> RiskLevel:
        for level in reversed(RiskLevel):
            if score >= self._risk_thresholds.get(level, 0):
                return level
        return RiskLevel.NONE

    def _determine_actions(self, risk_level: RiskLevel, detections: List[DetectionResult]) -> List[Action]:
        actions = []
        if risk_level == RiskLevel.NONE:
            actions.append(Action.ALLOW)
        elif risk_level == RiskLevel.LOW:
            actions.append(Action.ALERT)
            actions.append(Action.ALLOW)
        elif risk_level == RiskLevel.MEDIUM:
            actions.append(Action.ALERT)
            if self._sanitize_mode:
                actions.append(Action.SANITIZE)
            actions.append(Action.ALLOW)
        elif risk_level == RiskLevel.HIGH:
            actions.append(Action.ALERT)
            actions.append(Action.SANITIZE)
            actions.append(Action.QUARANTINE)
        elif risk_level == RiskLevel.CRITICAL:
            actions.append(Action.ALERT)
            actions.append(Action.BLOCK)
        return actions

    def _sanitize_content(self, content: str, detections: List[DetectionResult]) -> str:
        sanitized = content
        for detection in detections:
            for pattern in detection.matched_patterns:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    sanitized = regex.sub("[REDACTED]", sanitized)
                except re.error:
                    pass
        return sanitized

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _create_audit_event(
        self,
        request_id: str,
        detection: DetectionResult,
        action: Action,
        content_hash: str
    ) -> AuditEvent:
        return AuditEvent(
            timestamp=datetime.utcnow(),
            threat_type=detection.threat_type,
            risk_score=detection.risk_score,
            risk_level=detection.risk_level,
            action=action,
            content_hash=content_hash,
            details={
                "matched_patterns": detection.matched_patterns,
                "indicators": detection.indicators,
                "confidence": detection.confidence
            },
            request_id=request_id
        )

    def analyze_text(self, text: str, request_id: Optional[str] = None) -> DefenseResult:
        start_time = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        content_hash = self._hash_content(text)

        detections = []
        overall_score = 0.0

        for threat_type in ThreatType:
            if threat_type == ThreatType.EMBEDDING_MANIPULATION:
                continue
            
            matched_patterns, indicators = self._pattern_matcher.match(text, threat_type)
            if matched_patterns:
                weight = self._weights.get(threat_type, 1.0)
                confidence = min(len(matched_patterns) / 5.0, 1.0)
                risk_score = weight * (0.5 + 0.5 * confidence) * len(matched_patterns)
                risk_level = self._compute_risk_level(risk_score)
                
                detections.append(DetectionResult(
                    threat_type=threat_type,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    matched_patterns=matched_patterns,
                    indicators=indicators,
                    confidence=confidence
                ))
                overall_score += risk_score

        overall_risk_level = self._compute_risk_level(overall_score)
        actions = self._determine_actions(overall_risk_level, detections)
        
        quarantine_decision = Action.QUARANTINE in actions
        block_decision = Action.BLOCK in actions

        sanitized_content = text
        if self._sanitize_mode and (Action.SANITIZE in actions or quarantine_decision):
            sanitized_content = self._sanitize_content(text, detections)

        audit_events = []
        if self._lock:
            with self._lock:
                for detection in detections:
                    for action in actions:
                        if action in (Action.SANITIZE, Action.QUARANTINE, Action.BLOCK, Action.ALERT):
                            event = self._create_audit_event(request_id, detection, action, content_hash)
                            audit_events.append(event)
                            self._audit_log.append(event)
        else:
            for detection in detections:
                for action in actions:
                    if action in (Action.SANITIZE, Action.QUARANTINE, Action.BLOCK, Action.ALERT):
                        event = self._create_audit_event(request_id, detection, action, content_hash)
                        audit_events.append(event)
                        self._audit_log.append(event)

        self._request_count += 1
        if block_decision:
            self._blocked_count += 1
        if quarantine_decision:
            self._quarantined_count += 1

        processing_time = (time.perf_counter() - start_time) * 1000

        return DefenseResult(
            request_id=request_id,
            original_content=text,
            sanitized_content=sanitized_content,
            overall_risk_score=round(overall_score, 3),
            overall_risk_level=overall_risk_level,
            detections=detections,
            actions=actions,
            quarantine_decision=quarantine_decision,
            audit_events=audit_events,
            processing_time_ms=round(processing_time, 2),
            metadata={
                "request_count": self._request_count,
                "blocked_count": self._blocked_count,
                "quarantined_count": self._quarantined_count
            }
        )

    def analyze_embedding(self, embedding: np.ndarray, request_id: Optional[str] = None) -> DefenseResult:
        start_time = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        content_hash = self._hash_content(str(embedding.shape))

        anomaly_score, indicators = self._embedding_analyzer.analyze(embedding)
        
        detections = []
        if anomaly_score > 0:
            risk_level = self._compute_risk_level(anomaly_score)
            detections.append(DetectionResult(
                threat_type=ThreatType.EMBEDDING_MANIPULATION,
                risk_score=anomaly_score,
                risk_level=risk_level,
                matched_patterns=["statistical_outlier_detection"],
                indicators=indicators,
                confidence=min(anomaly_score / 5.0, 1.0)
            ))

        overall_risk_level = self._compute_risk_level(anomaly_score)
        actions = self._determine_actions(overall_risk_level, detections)
        quarantine_decision = Action.QUARANTINE in actions

        audit_events = []
        if self._lock:
            with self._lock:
                for detection in detections:
                    for action in actions:
                        if action in (Action.SANITIZE, Action.QUARANTINE, Action.BLOCK, Action.ALERT):
                            event = self._create_audit_event(request_id, detection, action, content_hash)
                            audit_events.append(event)
                            self._audit_log.append(event)
        else:
            for detection in detections:
                for action in actions:
                    if action in (Action.SANITIZE, Action.QUARANTINE, Action.BLOCK, Action.ALERT):
                        event = self._create_audit_event(request_id, detection, action, content_hash)
                        audit_events.append(event)
                        self._audit_log.append(event)

        processing_time = (time.perf_counter() - start_time) * 1000

        return DefenseResult(
            request_id=request_id,
            original_content=str(embedding.shape),
            sanitized_content=str(embedding.shape),
            overall_risk_score=round(anomaly_score, 3),
            overall_risk_level=overall_risk_level,
            detections=detections,
            actions=actions,
            quarantine_decision=quarantine_decision,
            audit_events=audit_events,
            processing_time_ms=round(processing_time, 2),
            metadata={"embedding_shape": embedding.shape}
        )

    def analyze(self, content: str, embedding: Optional[np.ndarray] = None, request_id: Optional[str] = None) -> DefenseResult:
        text_result = self.analyze_text(content, request_id)
        
        if embedding is not None:
            embedding_result = self.analyze_embedding(embedding, request_id)
            
            combined_score = text_result.overall_risk_score + embedding_result.overall_risk_score
            combined_level = self._compute_risk_level(combined_score)
            combined_actions = self._determine_actions(combined_level, text_result.detections + embedding_result.detections)
            combined_quarantine = Action.QUARANTINE in combined_actions
            
            combined_sanitized = text_result.sanitized_content
            if self._sanitize_mode and embedding_result.detections:
                combined_sanitized = text_result.sanitized_content
            
            return DefenseResult(
                request_id=text_result.request_id,
                original_content=text_result.original_content,
                sanitized_content=combined_sanitized,
                overall_risk_score=round(combined_score, 3),
                overall_risk_level=combined_level,
                detections=text_result.detections + embedding_result.detections,
                actions=combined_actions,
                quarantine_decision=combined_quarantine,
                audit_events=text_result.audit_events + embedding_result.audit_events,
                processing_time_ms=text_result.processing_time_ms + embedding_result.processing_time_ms,
                metadata={**text_result.metadata, **embedding_result.metadata}
            )
        
        return text_result

    def get_audit_log(self, limit: Optional[int] = None) -> List[AuditEvent]:
        if self._lock:
            with self._lock:
                return self._audit_log[-limit:] if limit else self._audit_log.copy()
        return self._audit_log[-limit:] if limit else self._audit_log.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._request_count,
            "blocked_requests": self._blocked_count,
            "quarantined_requests": self._quarantined_count,
            "block_rate": self._blocked_count / max(self._request_count, 1),
            "quarantine_rate": self._quarantined_count / max(self._request_count, 1),
            "audit_events": len(self._audit_log)
        }

    def reset_stats(self) -> None:
        self._request_count = 0
        self._blocked_count = 0
        self._quarantined_count = 0
        if self._lock:
            with self._lock:
                self._audit_log.clear()
        else:
            self._audit_log.clear()


_defense_controller_instance: Optional[AdversarialDefenseController] = None
_instance_lock = threading.Lock() if (threading := __import__('threading')) else None


def get_adversarial_defense_controller(
    weights: Optional[Dict[ThreatType, float]] = None,
    risk_thresholds: Optional[Dict[RiskLevel, float]] = None,
    embedding_analyzer: Optional[EmbeddingAnalyzer] = None,
    sanitize_mode: bool = True,
    quarantine_threshold: RiskLevel = RiskLevel.HIGH,
    block_threshold: RiskLevel = RiskLevel.CRITICAL,
) -> AdversarialDefenseController:
    global _defense_controller_instance
    lock = _instance_lock if _instance_lock else threading.Lock()
    with lock:
        if _defense_controller_instance is None:
            _defense_controller_instance = AdversarialDefenseController(
                weights=weights,
                risk_thresholds=risk_thresholds,
                embedding_analyzer=embedding_analyzer,
                sanitize_mode=sanitize_mode,
                quarantine_threshold=quarantine_threshold,
                block_threshold=block_threshold,
            )
        return _defense_controller_instance


def reset_adversarial_defense_controller() -> None:
    global _defense_controller_instance
    lock = _instance_lock if _instance_lock else threading.Lock()
    with lock:
        _defense_controller_instance = None