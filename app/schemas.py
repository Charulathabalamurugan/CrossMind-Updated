from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config import settings

Role = Literal["public", "researcher", "admin", "analyst", "viewer"]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )


class DocumentRecord(StrictModel):
    id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=settings.MAX_DOC_CONTENT_LENGTH)
    domain: str = Field(default="general", min_length=1, max_length=100)
    year: int = Field(default=2024, ge=1900, le=2100)
    authors: List[str] = Field(default_factory=list, max_length=100)
    tags: List[str] = Field(default_factory=list, max_length=100)
    allowed_roles: List[Role] = Field(default_factory=lambda: ["public", "researcher"])
    source: Optional[str] = Field(default=None, min_length=1, max_length=256)
    content_hash: Optional[str] = Field(default=None, min_length=1, max_length=256)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("authors", "tags")
    @classmethod
    def validate_string_list(cls, value: List[str]) -> List[str]:
        cleaned = []
        for item in value:
            item = str(item).strip()
            if item:
                cleaned.append(item[:256])
        return cleaned

    @field_validator("content", "title")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("Control characters are not allowed")
        return value


class DocumentIngestRequest(StrictModel):
    documents: List[DocumentRecord] = Field(..., min_length=1, max_length=1000)


class QueryRequest(StrictModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    user_role: Role = "researcher"
    confidence_proceed_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    confidence_investigate_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    session_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("Control characters are not allowed")
        if not value.strip():
            raise ValueError("Query cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_thresholds(self) -> "QueryRequest":
        if self.confidence_investigate_threshold > self.confidence_proceed_threshold:
            raise ValueError("confidence_investigate_threshold cannot exceed confidence_proceed_threshold")
        return self


class StreamQueryRequest(StrictModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    user_role: Role = "researcher"
    session_id: str = Field(default="default", min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class RiskFeedbackRequest(StrictModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    doc_id: str = Field(..., min_length=1, max_length=256)
    score: float = Field(..., ge=0.0, le=1.0)
    user_role: Role = "researcher"
    evidence_domains: List[str] = Field(default_factory=list, max_length=50)


class RuleRequest(StrictModel):
    rule_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    description: str = Field(default="", max_length=2000)
    keywords: List[str] = Field(default_factory=list, max_length=100)
    required_safety_tags: List[str] = Field(default_factory=list, max_length=100)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoginRequest(StrictModel):
    username: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(..., min_length=8, max_length=1024)


class RefreshRequest(StrictModel):
    refresh_token: str = Field(..., min_length=20, max_length=1024)


class CreateUserRequest(StrictModel):
    username: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.@-]+$")
    password: str = Field(..., min_length=8, max_length=1024)
    email: str = Field(..., min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    roles: List[Role] = Field(..., min_length=1, max_length=10)
    full_name: str = Field(default="", max_length=256)


class SaveQueryRequest(StrictModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    tags: List[str] = Field(default_factory=list, max_length=50)
    note: str = Field(default="", max_length=2000)


class UpdatePreferencesRequest(StrictModel):
    preferences: Dict[str, Any] = Field(..., max_length=100)


class AutoDiscoverStartRequest(StrictModel):
    interval_seconds: int = Field(default=300, ge=60, le=86400)


class SimulationRunRequest(StrictModel):
    hypothesis: Dict[str, Any] = Field(..., max_length=1000)
    domain: str = Field(default="materials", min_length=1, max_length=100)


class SimulationLogRequest(StrictModel):
    hypothesis_id: str = Field(..., min_length=1, max_length=256)
    domain: str = Field(..., min_length=1, max_length=100)
    outcome: Dict[str, Any] = Field(..., max_length=1000)


class HealthResponse(StrictModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str = settings.VERSION
    environment: str = settings.ENVIRONMENT
    schema_version: int = settings.API_SCHEMA_VERSION
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(StrictModel):
    error: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    schema_version: int = settings.API_SCHEMA_VERSION


class IngestionResponse(StrictModel):
    status: Literal["success", "accepted", "degraded"]
    ingested_count: int
    inserted_ids: List[str]
    task_id: Optional[str] = None
    schema_version: int = settings.API_SCHEMA_VERSION


class QueryResponse(StrictModel):
    query: str
    status: Literal["success", "degraded"] = "success"
    schema_version: int = settings.API_SCHEMA_VERSION
    data: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AutoDiscoverStartRequest",
    "CreateUserRequest",
    "DocumentIngestRequest",
    "DocumentRecord",
    "ErrorResponse",
    "HealthResponse",
    "IngestionResponse",
    "LoginRequest",
    "QueryRequest",
    "QueryResponse",
    "RefreshRequest",
    "RiskFeedbackRequest",
    "RuleRequest",
    "SaveQueryRequest",
    "SimulationLogRequest",
    "SimulationRunRequest",
    "StreamQueryRequest",
    "UpdatePreferencesRequest",
]
