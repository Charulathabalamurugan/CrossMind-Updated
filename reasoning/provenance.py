"""Enterprise provenance tracking for reasoning pipelines."""
from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
from threading import Lock

logger = logging.getLogger("crossmind.provenance")


class ProvenanceEventType(str, Enum):
    QUERY_RECEIVED = "query_received"
    EVIDENCE_RETRIEVED = "evidence_retrieved"
    REASONING_STEP = "reasoning_step"
    VALIDATION_PERFORMED = "validation_performed"
    CALIBRATION_APPLIED = "calibration_applied"
    RESULT_EMITTED = "result_emitted"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class ProvenanceEvent:
    event_id: str
    event_type: ProvenanceEventType
    timestamp: float
    pipeline_stage: str
    details: Dict[str, Any]
    chain_id: str
    parent_event_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceChain:
    chain_id: str
    events: List[ProvenanceEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: ProvenanceEvent) -> None:
        self.events.append(event)

    def get_events_by_type(self, event_type: ProvenanceEventType) -> List[ProvenanceEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }


_current_chain: ContextVar[Optional[ProvenanceChain]] = ContextVar("current_chain", default=None)
_chains: Dict[str, ProvenanceChain] = {}
_chains_lock = Lock()


class ProvenanceTracker:
    """Tracks and stores provenance chains for reasoning pipelines."""

    def __init__(self, max_chains: int = 10000):
        self.max_chains = max_chains

    def start_chain(
        self,
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceChain:
        chain_id = str(uuid.uuid4())
        chain = ProvenanceChain(
            chain_id=chain_id,
            metadata={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                **(metadata or {}),
            },
        )
        with _chains_lock:
            if len(_chains) >= self.max_chains:
                oldest = min(_chains.values(), key=lambda c: c.created_at)
                _chains.pop(oldest.chain_id, None)
            _chains[chain_id] = chain
        _current_chain.set(chain)
        logger.debug(f"Started provenance chain {chain_id}")
        return chain

    def end_chain(self) -> Optional[ProvenanceChain]:
        chain = _current_chain.get()
        if chain:
            _current_chain.set(None)
            logger.debug(f"Ended provenance chain {chain.chain_id}")
        return chain

    def get_current_chain(self) -> Optional[ProvenanceChain]:
        return _current_chain.get()

    def record_event(
        self,
        event_type: ProvenanceEventType,
        pipeline_stage: str,
        details: Dict[str, Any],
        parent_event_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ProvenanceEvent:
        chain = _current_chain.get()
        if not chain:
            chain = self.start_chain(correlation_id=correlation_id, tenant_id=tenant_id, user_id=user_id)

        event = ProvenanceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            pipeline_stage=pipeline_stage,
            details=details,
            chain_id=chain.chain_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id or chain.metadata.get("correlation_id"),
            tenant_id=tenant_id or chain.metadata.get("tenant_id"),
            user_id=user_id or chain.metadata.get("user_id"),
        )
        chain.add_event(event)
        logger.debug(f"Recorded provenance event {event.event_id} ({event_type.value})")
        return event

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        return _chains.get(chain_id)

    def get_chains_by_correlation(self, correlation_id: str) -> List[ProvenanceChain]:
        return [c for c in _chains.values() if c.metadata.get("correlation_id") == correlation_id]

    def get_chains_by_tenant(self, tenant_id: str) -> List[ProvenanceChain]:
        return [c for c in _chains.values() if c.metadata.get("tenant_id") == tenant_id]

    def export_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        chain = self.get_chain(chain_id)
        return chain.to_dict() if chain else None

    def clear_old_chains(self, max_age_seconds: float = 86400) -> int:
        now = time.time()
        removed = 0
        with _chains_lock:
            to_remove = [cid for cid, chain in _chains.items() if now - chain.created_at > max_age_seconds]
            for cid in to_remove:
                _chains.pop(cid, None)
                removed += 1
        return removed


_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = Lock()


def get_provenance_tracker() -> ProvenanceTracker:
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = ProvenanceTracker()
    return _tracker_instance


def record_provenance_event(
    event_type: ProvenanceEventType,
    pipeline_stage: str,
    details: Dict[str, Any],
    **kwargs,
) -> ProvenanceEvent:
    return get_provenance_tracker().record_event(event_type, pipeline_stage, details, **kwargs)


def start_provenance_chain(**kwargs) -> ProvenanceChain:
    return get_provenance_tracker().start_chain(**kwargs)


def end_provenance_chain() -> Optional[ProvenanceChain]:
    return get_provenance_tracker().end_chain()