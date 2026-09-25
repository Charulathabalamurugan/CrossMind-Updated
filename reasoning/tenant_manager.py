import threading
import contextvars
import uuid
import time
import json
import logging
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from config import settings

logger = logging.getLogger("crossmind.tenant")

_current_tenant: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_tenant", default=None)


class ResidencyRegion(str, Enum):
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_CENTRAL = "eu-central-1"
    EU_WEST = "eu-west-1"
    AP_SOUTHEAST = "ap-southeast-1"
    AP_NORTHEAST = "ap-northeast-1"


class ConsentType(str, Enum):
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PERSONALIZATION = "personalization"
    RESEARCH = "research"
    LEGAL_REQUIRED = "legal_required"


@dataclass
class TenantBudget:
    max_queries_per_minute: int = 100
    max_storage_mb: int = 1024
    max_api_calls_per_day: int = 10000
    max_concurrent_sessions: int = 10
    custom_limits: Dict[str, int] = field(default_factory=dict)

    def check_limit(self, limit_name: str, current_value: int) -> bool:
        limit = getattr(self, limit_name, None)
        if limit is not None:
            return current_value <= limit
        return self.custom_limits.get(limit_name, float('inf')) >= current_value

    def get_limit(self, limit_name: str) -> int:
        return getattr(self, limit_name, self.custom_limits.get(limit_name, float('inf')))


@dataclass
class TenantQuota:
    queries_used: int = 0
    storage_used_mb: int = 0
    api_calls_used: int = 0
    active_sessions: int = 0
    last_reset: float = field(default_factory=time.time)

    def reset_if_needed(self, interval_seconds: int = 86400):
        if time.time() - self.last_reset > interval_seconds:
            self.queries_used = 0
            self.api_calls_used = 0
            self.last_reset = time.time()


@dataclass
class ConsentRecord:
    tenant_id: str
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: float = field(default_factory=time.time)
    expiry: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        if self.expiry and time.time() > self.expiry:
            return False
        return self.granted


@dataclass
class DeletionTombstone:
    tenant_id: str
    resource_type: str
    resource_id: str
    deleted_at: float = field(default_factory=time.time)
    deleted_by: str = "system"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    event_type: str = ""
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""


@dataclass
class Tenant:
    tenant_id: str
    name: str
    display_name: str
    residency_region: ResidencyRegion = ResidencyRegion.US_EAST
    budget: TenantBudget = field(default_factory=TenantBudget)
    quota: TenantQuota = field(default_factory=TenantQuota)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    allowed_domains: List[str] = field(default_factory=list)
    data_residency_strict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "display_name": self.display_name,
            "residency_region": self.residency_region.value,
            "budget": {
                "max_queries_per_minute": self.budget.max_queries_per_minute,
                "max_storage_mb": self.budget.max_storage_mb,
                "max_api_calls_per_day": self.budget.max_api_calls_per_day,
                "max_concurrent_sessions": self.budget.max_concurrent_sessions,
                "custom_limits": self.budget.custom_limits,
            },
            "quota": {
                "queries_used": self.quota.queries_used,
                "storage_used_mb": self.quota.storage_used_mb,
                "api_calls_used": self.quota.api_calls_used,
                "active_sessions": self.quota.active_sessions,
                "last_reset": self.quota.last_reset,
            },
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "allowed_domains": self.allowed_domains,
            "data_residency_strict": self.data_residency_strict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tenant":
        budget_data = data.get("budget", {})
        quota_data = data.get("quota", {})
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            display_name=data["display_name"],
            residency_region=ResidencyRegion(data.get("residency_region", "us-east-1")),
            budget=TenantBudget(
                max_queries_per_minute=budget_data.get("max_queries_per_minute", 100),
                max_storage_mb=budget_data.get("max_storage_mb", 1024),
                max_api_calls_per_day=budget_data.get("max_api_calls_per_day", 10000),
                max_concurrent_sessions=budget_data.get("max_concurrent_sessions", 10),
                custom_limits=budget_data.get("custom_limits", {}),
            ),
            quota=TenantQuota(
                queries_used=quota_data.get("queries_used", 0),
                storage_used_mb=quota_data.get("storage_used_mb", 0),
                api_calls_used=quota_data.get("api_calls_used", 0),
                active_sessions=quota_data.get("active_sessions", 0),
                last_reset=quota_data.get("last_reset", time.time()),
            ),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
            allowed_domains=data.get("allowed_domains", []),
            data_residency_strict=data.get("data_residency_strict", True),
        )


class TenantManager:
    _instance: Optional["TenantManager"] = None
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
        self._tenants: Dict[str, Tenant] = {}
        self._tenant_lock = threading.RLock()
        self._consent_records: Dict[str, List[ConsentRecord]] = {}
        self._consent_lock = threading.RLock()
        self._deletion_tombstones: Dict[str, List[DeletionTombstone]] = {}
        self._tombstone_lock = threading.RLock()
        self._audit_events: List[AuditEvent] = []
        self._audit_lock = threading.RLock()
        self._cache_scopes: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self._retrieval_scopes: Dict[str, Set[str]] = {}
        self._retrieval_lock = threading.RLock()
        self._dldb_partition_map: Dict[str, str] = {}
        self._partition_lock = threading.RLock()
        self._audit_callbacks: List[Callable[[AuditEvent], None]] = []
        self._callback_lock = threading.RLock()
        self._initialized = True
        logger.info("TenantManager initialized")

    @classmethod
    def get_instance(cls) -> "TenantManager":
        return cls()

    def create_tenant(
        self,
        name: str,
        display_name: str,
        residency_region: ResidencyRegion = ResidencyRegion.US_EAST,
        budget: Optional[TenantBudget] = None,
        allowed_domains: Optional[List[str]] = None,
        data_residency_strict: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            display_name=display_name,
            residency_region=residency_region,
            budget=budget or TenantBudget(),
            allowed_domains=allowed_domains or [],
            data_residency_strict=data_residency_strict,
            metadata=metadata or {},
        )
        with self._tenant_lock:
            self._tenants[tenant_id] = tenant
            self._consent_records[tenant_id] = []
            self._deletion_tombstones[tenant_id] = []
            self._cache_scopes[tenant_id] = {}
            self._retrieval_scopes[tenant_id] = set()
            self._dldb_partition_map[tenant_id] = f"{tenant_id}_partition"
        self._audit_event(
            tenant_id=tenant_id,
            user_id="system",
            event_type="tenant_created",
            action="create",
            resource_type="tenant",
            resource_id=tenant_id,
            success=True,
            details={"display_name": display_name, "region": residency_region.value},
        )
        logger.info(f"Created tenant: {tenant_id} ({name})")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        with self._tenant_lock:
            return self._tenants.get(tenant_id)

    def get_tenant_by_name(self, name: str) -> Optional[Tenant]:
        with self._tenant_lock:
            for tenant in self._tenants.values():
                if tenant.name == name:
                    return tenant
        return None

    def list_tenants(self, active_only: bool = True) -> List[Tenant]:
        with self._tenant_lock:
            tenants = list(self._tenants.values())
            if active_only:
                tenants = [t for t in tenants if t.is_active]
            return tenants

    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        with self._tenant_lock:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return None
            for key, value in kwargs.items():
                if hasattr(tenant, key) and key not in ("tenant_id", "created_at"):
                    setattr(tenant, key, value)
            tenant.updated_at = time.time()
            self._audit_event(
                tenant_id=tenant_id,
                user_id="system",
                event_type="tenant_updated",
                action="update",
                resource_type="tenant",
                resource_id=tenant_id,
                success=True,
                details={"updated_fields": list(kwargs.keys())},
            )
            return tenant

    def delete_tenant(self, tenant_id: str, deleted_by: str = "system", reason: str = "") -> bool:
        with self._tenant_lock:
            tenant = self._tenants.pop(tenant_id, None)
            if not tenant:
                return False
            tenant.is_active = False
            self._tenants[tenant_id] = tenant
        self._audit_event(
            tenant_id=tenant_id,
            user_id=deleted_by,
            event_type="tenant_deleted",
            action="delete",
            resource_type="tenant",
            resource_id=tenant_id,
            success=True,
            details={"reason": reason},
        )
        logger.info(f"Deleted tenant: {tenant_id}")
        return True

    def set_current_tenant(self, tenant_id: Optional[str]) -> contextvars.Token:
        if tenant_id and not self.get_tenant(tenant_id):
            raise ValueError(f"Tenant {tenant_id} does not exist")
        return _current_tenant.set(tenant_id)

    def get_current_tenant(self) -> Optional[str]:
        return _current_tenant.get()

    def clear_current_tenant(self, token: contextvars.Token) -> None:
        _current_tenant.reset(token)

    def require_tenant(self) -> str:
        tenant_id = self.get_current_tenant()
        if not tenant_id:
            raise RuntimeError("No tenant context set. Use set_current_tenant() first.")
        return tenant_id

    def check_budget(self, tenant_id: str, limit_name: str, current_value: int) -> bool:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        return tenant.budget.check_limit(limit_name, current_value)

    def increment_quota(self, tenant_id: str, quota_type: str, amount: int = 1) -> bool:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        # Map quota_type to budget limit attribute
        limit_map = {
            "queries_used": "max_queries_per_minute",
            "storage_used_mb": "max_storage_mb",
            "api_calls_used": "max_api_calls_per_day",
            "active_sessions": "max_concurrent_sessions",
        }
        budget_attr = limit_map.get(quota_type, quota_type)
        with self._tenant_lock:
            tenant.quota.reset_if_needed()
            current = getattr(tenant.quota, quota_type, 0)
            limit = tenant.budget.get_limit(budget_attr)
            if current + amount >= limit:
                return False
            setattr(tenant.quota, quota_type, current + amount)
            return True

    def get_quota_usage(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None
        tenant.quota.reset_if_needed()
        budget = tenant.budget
        quota = tenant.quota
        return {
            "queries": {"used": quota.queries_used, "limit": budget.max_queries_per_minute},
            "storage_mb": {"used": quota.storage_used_mb, "limit": budget.max_storage_mb},
            "api_calls": {"used": quota.api_calls_used, "limit": budget.max_api_calls_per_day},
            "sessions": {"used": quota.active_sessions, "limit": budget.max_concurrent_sessions},
            "custom": {k: {"used": getattr(quota, k, 0), "limit": v} for k, v in budget.custom_limits.items()},
        }

    def set_cache_scope(self, tenant_id: str, key: str, value: Any) -> None:
        with self._cache_lock:
            if tenant_id not in self._cache_scopes:
                self._cache_scopes[tenant_id] = {}
            self._cache_scopes[tenant_id][key] = value

    def get_cache_scope(self, tenant_id: str, key: str, default: Any = None) -> Any:
        with self._cache_lock:
            return self._cache_scopes.get(tenant_id, {}).get(key, default)

    def delete_cache_scope(self, tenant_id: str, key: str) -> bool:
        with self._cache_lock:
            if tenant_id in self._cache_scopes and key in self._cache_scopes[tenant_id]:
                del self._cache_scopes[tenant_id][key]
                return True
        return False

    def clear_cache_scope(self, tenant_id: str) -> None:
        with self._cache_lock:
            self._cache_scopes[tenant_id] = {}

    def set_retrieval_scope(self, tenant_id: str, collection: str) -> None:
        with self._retrieval_lock:
            if tenant_id not in self._retrieval_scopes:
                self._retrieval_scopes[tenant_id] = set()
            self._retrieval_scopes[tenant_id].add(collection)

    def get_retrieval_scopes(self, tenant_id: str) -> Set[str]:
        with self._retrieval_lock:
            return self._retrieval_scopes.get(tenant_id, set()).copy()

    def remove_retrieval_scope(self, tenant_id: str, collection: str) -> bool:
        with self._retrieval_lock:
            if tenant_id in self._retrieval_scopes:
                return self._retrieval_scopes[tenant_id].discard(collection)
        return False

    def clear_retrieval_scopes(self, tenant_id: str) -> None:
        with self._retrieval_lock:
            self._retrieval_scopes[tenant_id] = set()

    def get_dldb_partition(self, tenant_id: str) -> str:
        with self._partition_lock:
            if tenant_id not in self._dldb_partition_map:
                self._dldb_partition_map[tenant_id] = f"{tenant_id}_partition"
            return self._dldb_partition_map[tenant_id]

    def set_dldb_partition(self, tenant_id: str, partition: str) -> None:
        with self._partition_lock:
            self._dldb_partition_map[tenant_id] = partition

    def check_data_residency(self, tenant_id: str, target_region: ResidencyRegion) -> bool:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        if not tenant.data_residency_strict:
            return True
        return tenant.residency_region == target_region

    def record_consent(
        self,
        tenant_id: str,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        expiry: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentRecord:
        record = ConsentRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            expiry=expiry,
            metadata=metadata or {},
        )
        with self._consent_lock:
            if tenant_id not in self._consent_records:
                self._consent_records[tenant_id] = []
            self._consent_records[tenant_id].append(record)
        self._audit_event(
            tenant_id=tenant_id,
            user_id=user_id,
            event_type="consent_recorded",
            action="create" if granted else "revoke",
            resource_type="consent",
            resource_id=consent_type.value,
            success=True,
            details={"granted": granted, "expiry": expiry},
        )
        return record

    def get_consent(self, tenant_id: str, user_id: str, consent_type: ConsentType) -> Optional[ConsentRecord]:
        with self._consent_lock:
            records = self._consent_records.get(tenant_id, [])
            for record in reversed(records):
                if record.user_id == user_id and record.consent_type == consent_type:
                    return record
        return None

    def has_valid_consent(self, tenant_id: str, user_id: str, consent_type: ConsentType) -> bool:
        record = self.get_consent(tenant_id, user_id, consent_type)
        return record is not None and record.is_valid()

    def list_consents(self, tenant_id: str, user_id: Optional[str] = None) -> List[ConsentRecord]:
        with self._consent_lock:
            records = self._consent_records.get(tenant_id, [])
            if user_id:
                records = [r for r in records if r.user_id == user_id]
            return records.copy()

    def create_deletion_tombstone(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        deleted_by: str = "system",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeletionTombstone:
        tombstone = DeletionTombstone(
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            deleted_by=deleted_by,
            reason=reason,
            metadata=metadata or {},
        )
        with self._tombstone_lock:
            if tenant_id not in self._deletion_tombstones:
                self._deletion_tombstones[tenant_id] = []
            self._deletion_tombstones[tenant_id].append(tombstone)
        self._audit_event(
            tenant_id=tenant_id,
            user_id=deleted_by,
            event_type="deletion_tombstone",
            action="delete",
            resource_type=resource_type,
            resource_id=resource_id,
            success=True,
            details={"reason": reason},
        )
        return tombstone

    def is_deleted(self, tenant_id: str, resource_type: str, resource_id: str) -> bool:
        with self._tombstone_lock:
            tombstones = self._deletion_tombstones.get(tenant_id, [])
            for ts in tombstones:
                if ts.resource_type == resource_type and ts.resource_id == resource_id:
                    return True
        return False

    def get_deletion_tombstones(self, tenant_id: str, resource_type: Optional[str] = None) -> List[DeletionTombstone]:
        with self._tombstone_lock:
            tombstones = self._deletion_tombstones.get(tenant_id, [])
            if resource_type:
                tombstones = [ts for ts in tombstones if ts.resource_type == resource_type]
            return tombstones.copy()

    def _audit_event(self, **kwargs) -> AuditEvent:
        event = AuditEvent(**kwargs)
        with self._audit_lock:
            self._audit_events.append(event)
            for callback in self._audit_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Audit callback error: {e}")
        return event

    def record_audit_event(self, **kwargs) -> AuditEvent:
        return self._audit_event(**kwargs)

    def get_audit_events(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[AuditEvent]:
        with self._audit_lock:
            events = self._audit_events
            if tenant_id:
                events = [e for e in events if e.tenant_id == tenant_id]
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            if since:
                events = [e for e in events if e.timestamp >= since]
            return events[-limit:]

    def register_audit_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        with self._callback_lock:
            self._audit_callbacks.append(callback)

    def unregister_audit_callback(self, callback: Callable[[AuditEvent], None]) -> bool:
        with self._callback_lock:
            if callback in self._audit_callbacks:
                self._audit_callbacks.remove(callback)
                return True
        return False

    def export_tenant_data(self, tenant_id: str) -> Dict[str, Any]:
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {}
        with self._consent_lock:
            consents = self._consent_records.get(tenant_id, [])
        with self._tombstone_lock:
            tombstones = self._deletion_tombstones.get(tenant_id, [])
        with self._audit_lock:
            audits = [e for e in self._audit_events if e.tenant_id == tenant_id]
        return {
            "tenant": tenant.to_dict(),
            "consents": [
                {
                    "tenant_id": c.tenant_id,
                    "user_id": c.user_id,
                    "consent_type": c.consent_type.value,
                    "granted": c.granted,
                    "timestamp": c.timestamp,
                    "expiry": c.expiry,
                    "metadata": c.metadata,
                }
                for c in consents
            ],
            "tombstones": [
                {
                    "tenant_id": t.tenant_id,
                    "resource_type": t.resource_type,
                    "resource_id": t.resource_id,
                    "deleted_at": t.deleted_at,
                    "deleted_by": t.deleted_by,
                    "reason": t.reason,
                    "metadata": t.metadata,
                }
                for t in tombstones
            ],
            "audit_events": [
                {
                    "event_id": e.event_id,
                    "tenant_id": e.tenant_id,
                    "user_id": e.user_id,
                    "event_type": e.event_type,
                    "action": e.action,
                    "resource_type": e.resource_type,
                    "resource_id": e.resource_id,
                    "timestamp": e.timestamp,
                    "success": e.success,
                    "details": e.details,
                    "ip_address": e.ip_address,
                    "user_agent": e.user_agent,
                }
                for e in audits
            ],
        }

    def reset_for_testing(self) -> None:
        """Reset all internal state for testing purposes."""
        with self._tenant_lock:
            self._tenants.clear()
        with self._consent_lock:
            self._consent_records.clear()
        with self._tombstone_lock:
            self._deletion_tombstones.clear()
        with self._audit_lock:
            self._audit_events.clear()
        with self._cache_lock:
            self._cache_scopes.clear()
        with self._retrieval_lock:
            self._retrieval_scopes.clear()
        with self._partition_lock:
            self._dldb_partition_map.clear()
        with self._callback_lock:
            self._audit_callbacks.clear()


_tenant_manager_instance: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    global _tenant_manager_instance
    if _tenant_manager_instance is None:
        _tenant_manager_instance = TenantManager()
    return _tenant_manager_instance


def set_current_tenant_context(tenant_id: Optional[str]) -> contextvars.Token:
    return get_tenant_manager().set_current_tenant(tenant_id)


def get_current_tenant_context() -> Optional[str]:
    return get_tenant_manager().get_current_tenant()


def clear_current_tenant_context(token: contextvars.Token) -> None:
    get_tenant_manager().clear_current_tenant(token)


__all__ = [
    "TenantManager",
    "get_tenant_manager",
    "Tenant",
    "TenantBudget",
    "TenantQuota",
    "ConsentRecord",
    "ConsentType",
    "DeletionTombstone",
    "AuditEvent",
    "ResidencyRegion",
    "set_current_tenant_context",
    "get_current_tenant_context",
    "clear_current_tenant_context",
    "_current_tenant",
]