import re
import threading
import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Set, Callable, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from config import settings
from reasoning.tenant_manager import get_tenant_manager, ConsentType, AuditEvent

logger = logging.getLogger("crossmind.compliance")


class PolicyType(str, Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    INSTITUTIONAL = "institutional"


class DataCategory(str, Enum):
    PII = "pii"
    PHI = "phi"
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    BEHAVIORAL = "behavioral"
    CREDENTIALS = "credentials"


class ProcessingPurpose(str, Enum):
    SERVICE_PROVISION = "service_provision"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    RESEARCH = "research"
    LEGAL_COMPLIANCE = "legal_compliance"
    SECURITY = "security"
    PERSONALIZATION = "personalization"


@dataclass
class RetentionPolicy:
    data_category: DataCategory
    retention_days: int
    auto_delete: bool = True
    archive_before_delete: bool = False
    archive_location: str = ""
    legal_hold_exempt: bool = False


@dataclass
class PolicyRule:
    policy_type: PolicyType
    name: str
    description: str
    data_categories: List[DataCategory] = field(default_factory=list)
    allowed_purposes: List[ProcessingPurpose] = field(default_factory=list)
    required_consent_types: List[ConsentType] = field(default_factory=list)
    retention_policies: List[RetentionPolicy] = field(default_factory=list)
    pii_patterns: List[Pattern] = field(default_factory=list)
    phi_patterns: List[Pattern] = field(default_factory=list)
    custom_validators: List[Callable[[str], bool]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_category(self, category: DataCategory) -> bool:
        return category in self.data_categories or not self.data_categories

    def allows_purpose(self, purpose: ProcessingPurpose) -> bool:
        return purpose in self.allowed_purposes or not self.allowed_purposes


@dataclass
class ComplianceViolation:
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    policy_type: PolicyType = PolicyType.INSTITUTIONAL
    rule_name: str = ""
    violation_type: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolution_notes: str = ""


@dataclass
class DeletionRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    reason: str = ""
    status: str = "pending"
    requested_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    processed_by: str = ""
    notes: str = ""


class PIIDetector:
    DEFAULT_PII_PATTERNS: Dict[str, Pattern] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "ipv6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
        "uuid": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        "date_of_birth": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        "passport_us": re.compile(r"\b[0-9]{9}\b"),
        "drivers_license_ca": re.compile(r"\b[D]\d{7}\b"),
    }

    DEFAULT_PHI_PATTERNS: Dict[str, Pattern] = {
        "mrn": re.compile(r"MRN[:\s]\s*\d{6,10}", re.IGNORECASE),
        "patient_id": re.compile(r"\b(?:patient|pt)[\s_]?id[:\s]?\d+\b", re.IGNORECASE),
        "icd10": re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?\b"),
        "cpt": re.compile(r"\b\d{5}\b"),
        "npi": re.compile(r"\b\d{10}\b"),
        "dea": re.compile(r"\b[A-Z]{2}\d{7}\b"),
    }

    def __init__(
        self,
        custom_pii_patterns: Optional[Dict[str, Pattern]] = None,
        custom_phi_patterns: Optional[Dict[str, Pattern]] = None,
    ):
        self.pii_patterns = {**self.DEFAULT_PII_PATTERNS, **(custom_pii_patterns or {})}
        self.phi_patterns = {**self.DEFAULT_PHI_PATTERNS, **(custom_phi_patterns or {})}

    def detect_pii(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        for name, pattern in self.pii_patterns.items():
            matches = []
            for match in pattern.finditer(text):
                matches.append({
                    "type": name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
            if matches:
                results[name] = matches
        return results

    def detect_phi(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        for name, pattern in self.phi_patterns.items():
            matches = []
            for match in pattern.finditer(text):
                matches.append({
                    "type": name,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
            if matches:
                results[name] = matches
        return results

    def detect_all(self, text: str) -> Dict[str, Any]:
        return {
            "pii": self.detect_pii(text),
            "phi": self.detect_phi(text),
        }

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        redacted = text
        for pattern in self.pii_patterns.values():
            redacted = pattern.sub(replacement, redacted)
        for pattern in self.phi_patterns.values():
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def redact_selective(
        self,
        text: str,
        pii_types: Optional[List[str]] = None,
        phi_types: Optional[List[str]] = None,
        replacement: str = "[REDACTED]",
    ) -> str:
        redacted = text
        pii_types = pii_types or list(self.pii_patterns.keys())
        phi_types = phi_types or list(self.phi_patterns.keys())
        for name in pii_types:
            if name in self.pii_patterns:
                redacted = self.pii_patterns[name].sub(replacement, redacted)
        for name in phi_types:
            if name in self.phi_patterns:
                redacted = self.phi_patterns[name].sub(replacement, redacted)
        return redacted


class ComplianceController:
    _instance: Optional["ComplianceController"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._policies: Dict[PolicyType, List[PolicyRule]] = {
            PolicyType.GDPR: [],
            PolicyType.HIPAA: [],
            PolicyType.INSTITUTIONAL: [],
        }
        self._policy_lock = threading.RLock()
        self._violations: List[ComplianceViolation] = []
        self._violation_lock = threading.RLock()
        self._deletion_requests: Dict[str, DeletionRequest] = {}
        self._deletion_lock = threading.RLock()
        self._pii_detector = PIIDetector()
        self._retention_timers: Dict[str, threading.Timer] = {}
        self._timer_lock = threading.RLock()
        self._violation_callbacks: List[Callable[[ComplianceViolation], None]] = []
        self._callback_lock = threading.RLock()
        self._initialized = True
        self._load_default_policies()
        logger.info("ComplianceController initialized")

    def _load_default_policies(self):
        gdpr_rule = PolicyRule(
            policy_type=PolicyType.GDPR,
            name="gdpr_core",
            description="Core GDPR compliance rules",
            data_categories=[
                DataCategory.PII,
                DataCategory.BIOMETRIC,
                DataCategory.LOCATION,
                DataCategory.BEHAVIORAL,
            ],
            allowed_purposes=[
                ProcessingPurpose.SERVICE_PROVISION,
                ProcessingPurpose.ANALYTICS,
                ProcessingPurpose.LEGAL_COMPLIANCE,
                ProcessingPurpose.SECURITY,
            ],
            required_consent_types=[
                ConsentType.ANALYTICS,
            ],
            retention_policies=[
                RetentionPolicy(DataCategory.PII, 730, auto_delete=True),
                RetentionPolicy(DataCategory.BEHAVIORAL, 365, auto_delete=True),
                RetentionPolicy(DataCategory.LOCATION, 90, auto_delete=True),
            ],
        )
        hipaa_rule = PolicyRule(
            policy_type=PolicyType.HIPAA,
            name="hipaa_core",
            description="Core HIPAA compliance rules",
            data_categories=[DataCategory.PHI, DataCategory.HEALTH],
            allowed_purposes=[
                ProcessingPurpose.SERVICE_PROVISION,
                ProcessingPurpose.LEGAL_COMPLIANCE,
                ProcessingPurpose.SECURITY,
                ProcessingPurpose.RESEARCH,
            ],
            required_consent_types=[ConsentType.RESEARCH, ConsentType.LEGAL_REQUIRED],
            retention_policies=[
                RetentionPolicy(DataCategory.PHI, 2555, auto_delete=False, archive_before_delete=True),
                RetentionPolicy(DataCategory.HEALTH, 2555, auto_delete=False),
            ],
        )
        institutional_rule = PolicyRule(
            policy_type=PolicyType.INSTITUTIONAL,
            name="institutional_core",
            description="Institutional data governance rules",
            data_categories=[
                DataCategory.FINANCIAL,
                DataCategory.CREDENTIALS,
                DataCategory.PII,
            ],
            allowed_purposes=[
                ProcessingPurpose.SERVICE_PROVISION,
                ProcessingPurpose.ANALYTICS,
                ProcessingPurpose.SECURITY,
                ProcessingPurpose.LEGAL_COMPLIANCE,
            ],
            required_consent_types=[ConsentType.ANALYTICS, ConsentType.LEGAL_REQUIRED],
            retention_policies=[
                RetentionPolicy(DataCategory.FINANCIAL, 2555, auto_delete=False, archive_before_delete=True),
                RetentionPolicy(DataCategory.CREDENTIALS, 365, auto_delete=True),
            ],
        )
        self.add_policy(gdpr_rule)
        self.add_policy(hipaa_rule)
        self.add_policy(institutional_rule)

    @classmethod
    def get_instance(cls) -> "ComplianceController":
        return cls()

    def add_policy(self, rule: PolicyRule) -> None:
        with self._policy_lock:
            self._policies[rule.policy_type].append(rule)
        logger.info(f"Added policy rule: {rule.name} ({rule.policy_type.value})")

    def remove_policy(self, policy_type: PolicyType, rule_name: str) -> bool:
        with self._policy_lock:
            rules = self._policies.get(policy_type, [])
            for i, rule in enumerate(rules):
                if rule.name == rule_name:
                    rules.pop(i)
                    logger.info(f"Removed policy rule: {rule_name} ({policy_type.value})")
                    return True
        return False

    def get_policies(self, policy_type: Optional[PolicyType] = None) -> List[PolicyRule]:
        with self._policy_lock:
            if policy_type:
                return self._policies.get(policy_type, []).copy()
            all_rules = []
            for rules in self._policies.values():
                all_rules.extend(rules)
            return all_rules

    def check_purpose_limitation(
        self,
        tenant_id: str,
        user_id: str,
        data_category: DataCategory,
        purpose: ProcessingPurpose,
        policy_types: Optional[List[PolicyType]] = None,
    ) -> bool:
        policy_types = policy_types or list(PolicyType)
        tenant_manager = get_tenant_manager()
        for ptype in policy_types:
            rules = self.get_policies(ptype)
            for rule in rules:
                if rule.matches_category(data_category):
                    if not rule.allows_purpose(purpose):
                        self._record_violation(
                            tenant_id=tenant_id,
                            policy_type=ptype,
                            rule_name=rule.name,
                            violation_type="purpose_limitation",
                            resource_type="data_processing",
                            resource_id=f"{data_category.value}:{purpose.value}",
                            details={"data_category": data_category.value, "purpose": purpose.value},
                            severity="high",
                        )
                        return False
                    for consent_type in rule.required_consent_types:
                        if not tenant_manager.has_valid_consent(tenant_id, user_id, consent_type):
                            self._record_violation(
                                tenant_id=tenant_id,
                                policy_type=ptype,
                                rule_name=rule.name,
                                violation_type="missing_consent",
                                resource_type="consent",
                                resource_id=consent_type.value,
                                details={"required_consent": consent_type.value},
                                severity="high",
                            )
                            return False
        return True

    def scan_content(
        self,
        content: str,
        tenant_id: str = "",
        user_id: str = "",
        resource_type: str = "content",
        resource_id: str = "",
    ) -> Dict[str, Any]:
        detections = self._pii_detector.detect_all(content)
        violations = []
        for pii_type, matches in detections.get("pii", {}).items():
            for match in matches:
                violations.append({
                    "type": "pii",
                    "subtype": pii_type,
                    "value": match["value"],
                    "position": {"start": match["start"], "end": match["end"]},
                })
        for phi_type, matches in detections.get("phi", {}).items():
            for match in matches:
                violations.append({
                    "type": "phi",
                    "subtype": phi_type,
                    "value": match["value"],
                    "position": {"start": match["start"], "end": match["end"]},
                })
        if violations and tenant_id:
            for v in violations:
                self._record_violation(
                    tenant_id=tenant_id,
                    policy_type=PolicyType.GDPR if v["type"] == "pii" else PolicyType.HIPAA,
                    rule_name="content_scan",
                    violation_type=f"{v['type']}_detected",
                    resource_type=resource_type,
                    resource_id=resource_id or "unknown",
                    details={"detection": v},
                    severity="medium",
                )
        return {
            "clean": len(violations) == 0,
            "violations": violations,
            "pii_detected": len(detections.get("pii", {})) > 0,
            "phi_detected": len(detections.get("phi", {})) > 0,
        }

    def redact_content(
        self,
        content: str,
        pii_types: Optional[List[str]] = None,
        phi_types: Optional[List[str]] = None,
        replacement: str = "[REDACTED]",
    ) -> str:
        return self._pii_detector.redact_selective(content, pii_types, phi_types, replacement)

    def enforce_retention(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        data_category: DataCategory,
        policy_types: Optional[List[PolicyType]] = None,
    ) -> Optional[RetentionPolicy]:
        policy_types = policy_types or list(PolicyType)
        for ptype in policy_types:
            rules = self.get_policies(ptype)
            for rule in rules:
                for retention in rule.retention_policies:
                    if retention.data_category == data_category:
                        return retention
        return None

    def schedule_deletion(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        data_category: DataCategory,
        policy_types: Optional[List[PolicyType]] = None,
        callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> Optional[threading.Timer]:
        retention = self.enforce_retention(tenant_id, resource_type, resource_id, data_category, policy_types)
        if not retention or not retention.auto_delete:
            return None
        delay_seconds = retention.retention_days * 86400
        # Cap delay to avoid threading.Timer overflow (max ~24 days on some platforms)
        max_delay = 86400 * 20  # 20 days
        if delay_seconds > max_delay:
            delay_seconds = max_delay
        timer = threading.Timer(delay_seconds, self._execute_deletion, args=[tenant_id, resource_type, resource_id, callback])
        timer.daemon = True
        timer.start()
        with self._timer_lock:
            key = f"{tenant_id}:{resource_type}:{resource_id}"
            self._retention_timers[key] = timer
        return timer

    def _execute_deletion(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        tenant_manager = get_tenant_manager()
        tenant_manager.create_deletion_tombstone(
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            deleted_by="retention_policy",
            reason=f"Automatic deletion after retention period",
        )
        if callback:
            try:
                callback(tenant_id, resource_type, resource_id)
            except Exception as e:
                logger.error(f"Deletion callback error: {e}")
        with self._timer_lock:
            key = f"{tenant_id}:{resource_type}:{resource_id}"
            self._retention_timers.pop(key, None)
        logger.info(f"Executed retention deletion: {tenant_id}:{resource_type}:{resource_id}")

    def cancel_scheduled_deletion(self, tenant_id: str, resource_type: str, resource_id: str) -> bool:
        with self._timer_lock:
            key = f"{tenant_id}:{resource_type}:{resource_id}"
            timer = self._retention_timers.pop(key, None)
            if timer:
                timer.cancel()
                return True
        return False

    def request_deletion(
        self,
        tenant_id: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        reason: str = "User requested deletion",
    ) -> DeletionRequest:
        request = DeletionRequest(
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            reason=reason,
        )
        with self._deletion_lock:
            self._deletion_requests[request.request_id] = request
        self._audit_event(
            tenant_id=tenant_id,
            user_id=user_id,
            event_type="deletion_requested",
            action="request",
            resource_type=resource_type,
            resource_id=resource_id,
            success=True,
            details={"reason": reason},
        )
        return request

    def process_deletion_request(
        self,
        request_id: str,
        processed_by: str,
        approve: bool = True,
        notes: str = "",
    ) -> Optional[DeletionRequest]:
        with self._deletion_lock:
            request = self._deletion_requests.get(request_id)
            if not request:
                return None
            request.status = "approved" if approve else "rejected"
            request.processed_at = time.time()
            request.processed_by = processed_by
            request.notes = notes
            if approve:
                tenant_manager = get_tenant_manager()
                tenant_manager.create_deletion_tombstone(
                    tenant_id=request.tenant_id,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    deleted_by=processed_by,
                    reason=f"Deletion request {request_id}: {notes or 'Approved'}",
                )
        self._audit_event(
            tenant_id=request.tenant_id,
            user_id=processed_by,
            event_type="deletion_processed",
            action="approve" if approve else "reject",
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            success=True,
            details={"request_id": request_id, "approved": approve},
        )
        return request

    def get_deletion_request(self, request_id: str) -> Optional[DeletionRequest]:
        with self._deletion_lock:
            return self._deletion_requests.get(request_id)

    def list_deletion_requests(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DeletionRequest]:
        with self._deletion_lock:
            requests = list(self._deletion_requests.values())
            if tenant_id:
                requests = [r for r in requests if r.tenant_id == tenant_id]
            if status:
                requests = [r for r in requests if r.status == status]
            return requests

    def _record_violation(
        self,
        tenant_id: str,
        policy_type: PolicyType,
        rule_name: str,
        violation_type: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        severity: str = "medium",
    ) -> ComplianceViolation:
        violation = ComplianceViolation(
            tenant_id=tenant_id,
            policy_type=policy_type,
            rule_name=rule_name,
            violation_type=violation_type,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            severity=severity,
        )
        with self._violation_lock:
            self._violations.append(violation)
        for callback in self._violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")
        self._audit_event(
            tenant_id=tenant_id,
            user_id="system",
            event_type="compliance_violation",
            action="detect",
            resource_type=resource_type,
            resource_id=resource_id,
            success=False,
            details={"violation": violation.violation_type, "rule": rule_name, "severity": severity},
        )
        return violation

    def get_violations(
        self,
        tenant_id: Optional[str] = None,
        policy_type: Optional[PolicyType] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[ComplianceViolation]:
        with self._violation_lock:
            violations = self._violations
            if tenant_id:
                violations = [v for v in violations if v.tenant_id == tenant_id]
            if policy_type:
                violations = [v for v in violations if v.policy_type == policy_type]
            if resolved is not None:
                violations = [v for v in violations if v.resolved == resolved]
            return violations[-limit:]

    def resolve_violation(
        self,
        violation_id: str,
        resolved_by: str,
        notes: str = "",
    ) -> Optional[ComplianceViolation]:
        with self._violation_lock:
            for v in self._violations:
                if v.violation_id == violation_id:
                    v.resolved = True
                    v.resolved_at = time.time()
                    v.resolution_notes = notes
                    self._audit_event(
                        tenant_id=v.tenant_id,
                        user_id=resolved_by,
                        event_type="violation_resolved",
                        action="resolve",
                        resource_type="violation",
                        resource_id=violation_id,
                        success=True,
                        details={"notes": notes},
                    )
                    return v
        return None

    def _audit_event(self, **kwargs) -> AuditEvent:
        tenant_manager = get_tenant_manager()
        return tenant_manager.record_audit_event(**kwargs)

    def get_audit_trail(
        self,
        tenant_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        tenant_manager = get_tenant_manager()
        events = tenant_manager.get_audit_events(tenant_id=tenant_id, limit=limit * 2)
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        return events[-limit:]

    def register_violation_callback(self, callback: Callable[[ComplianceViolation], None]) -> None:
        with self._callback_lock:
            self._violation_callbacks.append(callback)

    def unregister_violation_callback(self, callback: Callable[[ComplianceViolation], None]) -> bool:
        with self._callback_lock:
            if callback in self._violation_callbacks:
                self._violation_callbacks.remove(callback)
                return True
        return False

    def export_compliance_report(self, tenant_id: str) -> Dict[str, Any]:
        violations = self.get_violations(tenant_id=tenant_id)
        deletion_requests = self.list_deletion_requests(tenant_id=tenant_id)
        audit_trail = self.get_audit_trail(tenant_id=tenant_id)
        return {
            "tenant_id": tenant_id,
            "generated_at": time.time(),
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "policy_type": v.policy_type.value,
                    "rule_name": v.rule_name,
                    "violation_type": v.violation_type,
                    "resource_type": v.resource_type,
                    "resource_id": v.resource_id,
                    "details": v.details,
                    "severity": v.severity,
                    "detected_at": v.detected_at,
                    "resolved": v.resolved,
                    "resolved_at": v.resolved_at,
                    "resolution_notes": v.resolution_notes,
                }
                for v in violations
            ],
            "deletion_requests": [
                {
                    "request_id": d.request_id,
                    "resource_type": d.resource_type,
                    "resource_id": d.resource_id,
                    "reason": d.reason,
                    "status": d.status,
                    "requested_at": d.requested_at,
                    "processed_at": d.processed_at,
                    "processed_by": d.processed_by,
                    "notes": d.notes,
                }
                for d in deletion_requests
            ],
            "audit_events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "action": e.action,
                    "resource_type": e.resource_type,
                    "resource_id": e.resource_id,
                    "timestamp": e.timestamp,
                    "success": e.success,
                    "details": e.details,
                }
                for e in audit_trail
            ],
        }

    def reset_for_testing(self) -> None:
        """Reset all internal state for testing purposes."""
        with self._policy_lock:
            self._policies = {
                PolicyType.GDPR: [],
                PolicyType.HIPAA: [],
                PolicyType.INSTITUTIONAL: [],
            }
        with self._violation_lock:
            self._violations.clear()
        with self._deletion_lock:
            self._deletion_requests.clear()
        with self._timer_lock:
            for timer in self._retention_timers.values():
                timer.cancel()
            self._retention_timers.clear()
        with self._callback_lock:
            self._violation_callbacks.clear()
        self._load_default_policies()


_compliance_controller_instance: Optional[ComplianceController] = None


def get_compliance_controller() -> ComplianceController:
    global _compliance_controller_instance
    if _compliance_controller_instance is None:
        _compliance_controller_instance = ComplianceController()
    return _compliance_controller_instance


__all__ = [
    "ComplianceController",
    "get_compliance_controller",
    "PolicyType",
    "DataCategory",
    "ProcessingPurpose",
    "PolicyRule",
    "RetentionPolicy",
    "ComplianceViolation",
    "DeletionRequest",
    "PIIDetector",
]