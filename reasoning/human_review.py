import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("crossmind.human_review")


class ReviewStatus(Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Decision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    OVERRIDE = "override"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    actor: str
    action: str
    details: Dict[str, Any]
    case_id: str


@dataclass
class HumanReviewCase:
    case_id: str
    claim: str
    context: Dict[str, Any]
    priority: Priority
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    assigned_expert: Optional[str] = None
    reviewer: Optional[str] = None
    decision: Optional[Decision] = None
    decision_rationale: Optional[str] = None
    sla_deadline: Optional[datetime] = None
    sign_off_status: bool = False
    sign_off_by: Optional[str] = None
    sign_off_at: Optional[datetime] = None
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    history: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_annotation(self, author: str, content: str, annotation_type: str = "note") -> None:
        self.annotations.append({
            "id": str(uuid.uuid4()),
            "author": author,
            "content": content,
            "type": annotation_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()
        self._add_history(author, "annotate", {"content": content, "type": annotation_type})

    def _add_history(self, actor: str, action: str, details: Dict[str, Any]) -> None:
        self.history.append(AuditEvent(
            timestamp=datetime.utcnow(),
            actor=actor,
            action=action,
            details=details,
            case_id=self.case_id
        ))

    def assign_expert(self, expert_id: str, actor: str) -> None:
        self.assigned_expert = expert_id
        self.updated_at = datetime.utcnow()
        self._add_history(actor, "assign_expert", {"expert_id": expert_id})

    def start_review(self, reviewer: str) -> None:
        self.status = ReviewStatus.UNDER_REVIEW
        self.reviewer = reviewer
        self.updated_at = datetime.utcnow()
        self._add_history(reviewer, "start_review", {})

    def make_decision(self, decision: Decision, rationale: str, actor: str) -> None:
        self.decision = decision
        self.decision_rationale = rationale
        self.updated_at = datetime.utcnow()
        
        if decision == Decision.APPROVE:
            self.status = ReviewStatus.APPROVED
        elif decision == Decision.REJECT:
            self.status = ReviewStatus.REJECTED
        elif decision == Decision.OVERRIDE:
            self.status = ReviewStatus.OVERRIDDEN
        elif decision == Decision.ESCALATE:
            self.status = ReviewStatus.ESCALATED
        
        self._add_history(actor, "decision", {
            "decision": decision.value,
            "rationale": rationale
        })

    def sign_off(self, signer: str) -> None:
        self.sign_off_status = True
        self.sign_off_by = signer
        self.sign_off_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._add_history(signer, "sign_off", {})

    def is_sla_breached(self) -> bool:
        if self.sla_deadline is None:
            return False
        return datetime.utcnow() > self.sla_deadline

    def time_remaining(self) -> Optional[timedelta]:
        if self.sla_deadline is None:
            return None
        remaining = self.sla_deadline - datetime.utcnow()
        return remaining if remaining.total_seconds() > 0 else timedelta(0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "claim": self.claim,
            "context": self.context,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_expert": self.assigned_expert,
            "reviewer": self.reviewer,
            "decision": self.decision.value if self.decision else None,
            "decision_rationale": self.decision_rationale,
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None,
            "sign_off_status": self.sign_off_status,
            "sign_off_by": self.sign_off_by,
            "sign_off_at": self.sign_off_at.isoformat() if self.sign_off_at else None,
            "annotations": self.annotations,
            "history": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "action": e.action,
                    "details": e.details
                }
                for e in self.history
            ],
            "metadata": self.metadata,
            "sla_breached": self.is_sla_breached(),
            "time_remaining_seconds": self.time_remaining().total_seconds() if self.time_remaining() else None
        }


class HumanReviewQueue:
    DEFAULT_SLA_HOURS = {
        Priority.LOW: 168,
        Priority.MEDIUM: 72,
        Priority.HIGH: 24,
        Priority.CRITICAL: 4
    }

    def __init__(
        self,
        sla_hours: Optional[Dict[Priority, int]] = None,
        callback_hooks: Optional[Dict[str, Callable]] = None
    ):
        self._cases: Dict[str, HumanReviewCase] = {}
        self._lock = threading.RLock()
        self._sla_hours = sla_hours if sla_hours is not None else self.DEFAULT_SLA_HOURS.copy()
        self._callback_hooks = callback_hooks or {}
        self._expert_assignments: Dict[str, Set[str]] = {}
        self._case_index_by_expert: Dict[str, Set[str]] = {}
        self._case_index_by_status: Dict[ReviewStatus, Set[str]] = {s: set() for s in ReviewStatus}
        self._case_index_by_priority: Dict[Priority, Set[str]] = {p: set() for p in Priority}
        self._shutdown = False

    def _fire_callback(self, hook_name: str, case: HumanReviewCase, **kwargs) -> None:
        hook = self._callback_hooks.get(hook_name)
        if hook:
            try:
                hook(case, **kwargs)
            except Exception as e:
                logger.error(f"Callback hook '{hook_name}' failed: {e}")

    def _update_indexes(self, case: HumanReviewCase, old_status: Optional[ReviewStatus] = None, old_priority: Optional[Priority] = None) -> None:
        if old_status and old_status != case.status:
            self._case_index_by_status[old_status].discard(case.case_id)
        self._case_index_by_status[case.status].add(case.case_id)

        if old_priority and old_priority != case.priority:
            self._case_index_by_priority[old_priority].discard(case.case_id)
        self._case_index_by_priority[case.priority].add(case.case_id)

        if case.assigned_expert:
            if case.assigned_expert not in self._case_index_by_expert:
                self._case_index_by_expert[case.assigned_expert] = set()
            self._case_index_by_expert[case.assigned_expert].add(case.case_id)

    def create_case(
        self,
        claim: str,
        context: Dict[str, Any],
        priority: Priority = Priority.MEDIUM,
        sla_hours_override: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor: str = "system"
    ) -> HumanReviewCase:
        with self._lock:
            case_id = str(uuid.uuid4())
            
            if sla_hours_override is not None:
                sla_hours = sla_hours_override
                sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
            elif priority in self._sla_hours:
                sla_hours = self._sla_hours[priority]
                sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
            else:
                sla_hours = None
                sla_deadline = None
            
            case = HumanReviewCase(
                case_id=case_id,
                claim=claim,
                context=context,
                priority=priority,
                sla_deadline=sla_deadline,
                metadata=metadata or {}
            )
            case._add_history(actor, "create", {"priority": priority.name, "sla_hours": sla_hours})
            
            self._cases[case_id] = case
            self._update_indexes(case)
            
            self._fire_callback("on_case_created", case)
            
            logger.info(f"Created review case {case_id} with priority {priority.name}")
            return case

    def get_case(self, case_id: str) -> Optional[HumanReviewCase]:
        with self._lock:
            return self._cases.get(case_id)

    def get_cases_by_status(self, status: ReviewStatus) -> List[HumanReviewCase]:
        with self._lock:
            case_ids = self._case_index_by_status.get(status, set())
            return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_cases_by_priority(self, priority: Priority) -> List[HumanReviewCase]:
        with self._lock:
            case_ids = self._case_index_by_priority.get(priority, set())
            return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_cases_by_expert(self, expert_id: str) -> List[HumanReviewCase]:
        with self._lock:
            case_ids = self._case_index_by_expert.get(expert_id, set())
            return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_pending_cases(self) -> List[HumanReviewCase]:
        with self._lock:
            pending_ids = self._case_index_by_status.get(ReviewStatus.PENDING, set())
            return [self._cases[cid] for cid in pending_ids if cid in self._cases]

    def get_overdue_cases(self) -> List[HumanReviewCase]:
        with self._lock:
            overdue = []
            for case in self._cases.values():
                if case.is_sla_breached() and case.status in (ReviewStatus.PENDING, ReviewStatus.UNDER_REVIEW):
                    overdue.append(case)
            return overdue

    def annotate_case(self, case_id: str, author: str, content: str, annotation_type: str = "note") -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            case.add_annotation(author, content, annotation_type)
            self._fire_callback("on_case_annotated", case, author=author, content=content)
            return True

    def assign_expert(self, case_id: str, expert_id: str, actor: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            
            old_expert = case.assigned_expert
            if old_expert and old_expert in self._case_index_by_expert:
                self._case_index_by_expert[old_expert].discard(case_id)
            
            case.assign_expert(expert_id, actor)
            self._update_indexes(case)
            
            if expert_id not in self._expert_assignments:
                self._expert_assignments[expert_id] = set()
            self._expert_assignments[expert_id].add(case_id)
            
            self._fire_callback("on_expert_assigned", case, expert_id=expert_id)
            return True

    def start_review(self, case_id: str, reviewer: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            if case.status != ReviewStatus.PENDING:
                return False
            
            old_status = case.status
            case.start_review(reviewer)
            self._update_indexes(case, old_status=old_status)
            self._fire_callback("on_review_started", case, reviewer=reviewer)
            return True

    def make_decision(
        self,
        case_id: str,
        decision: Decision,
        rationale: str,
        actor: str
    ) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            if case.status not in (ReviewStatus.PENDING, ReviewStatus.UNDER_REVIEW):
                return False
            
            old_status = case.status
            case.make_decision(decision, rationale, actor)
            self._update_indexes(case, old_status=old_status)
            
            hook_name = f"on_{decision.value.lower()}"
            self._fire_callback(hook_name, case, rationale=rationale)
            self._fire_callback("on_decision_made", case, decision=decision, rationale=rationale)
            return True

    def sign_off(self, case_id: str, signer: str) -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            if case.status not in (ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.OVERRIDDEN):
                return False
            if case.sign_off_status:
                return False
            
            case.sign_off(signer)
            self._fire_callback("on_signed_off", case, signer=signer)
            return True

    def escalate(self, case_id: str, actor: str, reason: str = "") -> bool:
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return False
            
            old_status = case.status
            case.make_decision(Decision.ESCALATE, reason or "Escalated by " + actor, actor)
            self._update_indexes(case, old_status=old_status)
            self._fire_callback("on_escalated", case, reason=reason)
            return True

    def list_all_cases(self) -> List[HumanReviewCase]:
        with self._lock:
            return list(self._cases.values())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._cases)
            by_status = {s.value: len(ids) for s, ids in self._case_index_by_status.items()}
            by_priority = {p.name: len(ids) for p, ids in self._case_index_by_priority.items()}
            overdue_count = len(self.get_overdue_cases())
            signed_off = sum(1 for c in self._cases.values() if c.sign_off_status)
            
            return {
                "total_cases": total,
                "by_status": by_status,
                "by_priority": by_priority,
                "overdue_cases": overdue_count,
                "signed_off_cases": signed_off,
                "experts_assigned": len(self._expert_assignments)
            }

    def register_callback(self, hook_name: str, callback: Callable) -> None:
        with self._lock:
            self._callback_hooks[hook_name] = callback

    def unregister_callback(self, hook_name: str) -> None:
        with self._lock:
            self._callback_hooks.pop(hook_name, None)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._fire_callback("on_shutdown", None)


_review_queue_instance: Optional[HumanReviewQueue] = None
_instance_lock = threading.Lock()


def get_human_review_queue(
    sla_hours: Optional[Dict[Priority, int]] = None,
    callback_hooks: Optional[Dict[str, Callable]] = None
) -> HumanReviewQueue:
    global _review_queue_instance
    with _instance_lock:
        if _review_queue_instance is None:
            _review_queue_instance = HumanReviewQueue(sla_hours=sla_hours, callback_hooks=callback_hooks)
        return _review_queue_instance


def reset_human_review_queue() -> None:
    global _review_queue_instance
    with _instance_lock:
        if _review_queue_instance:
            _review_queue_instance.shutdown()
        _review_queue_instance = None