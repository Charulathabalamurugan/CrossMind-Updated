from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnomalyResult:
    context: str
    label: str
    score: float
    confidence: float
    threshold: float
    action: str
    model: str
    input_hash: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.label == "anomaly"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDetector(ABC):
    @abstractmethod
    async def detect(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AnomalyResult:
        """Return a context-specific, structured anomaly assessment."""

