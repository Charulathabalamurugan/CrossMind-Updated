"""Enterprise versioning for models, pipelines, and data schemas."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from threading import RLock

logger = logging.getLogger("crossmind.versioning")


class VersionComponent(str, Enum):
    MODEL = "model"
    PIPELINE = "pipeline"
    DATA_SCHEMA = "data_schema"
    API_SCHEMA = "api_schema"
    VECTOR_SCHEMA = "vector_schema"
    KG_SCHEMA = "kg_schema"
    CACHE_SCHEMA = "cache_schema"


@dataclass
class VersionInfo:
    component: VersionComponent
    version: str
    description: str = ""
    released_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecated_at: Optional[str] = None
    replacement_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompatibilityMatrix:
    component: VersionComponent
    from_version: str
    to_version: str
    compatible: bool
    migration_required: bool = False
    migration_notes: str = ""
    breaking_changes: List[str] = field(default_factory=list)


_version_registry: Dict[str, Dict[str, VersionInfo]] = {}
_compatibility_matrix: Dict[str, List[CompatibilityMatrix]] = {}
_registry_lock = RLock()

_registry_instance: Optional["VersionRegistry"] = None


class VersionRegistry:
    """Central registry for all versioned components."""

    def __new__(cls):
        global _registry_instance
        if _registry_instance is None:
            with _registry_lock:
                if _registry_instance is None:
                    _registry_instance = super().__new__(cls)
                    _registry_instance._initialized = False
        return _registry_instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            with _registry_lock:
                if not self._initialized:
                    self._register_defaults()
                    self._initialized = True

    def _register_defaults(self) -> None:
        from config import settings
        defaults = {
            VersionComponent.MODEL: VersionInfo(
                component=VersionComponent.MODEL,
                version=settings.VERSION,
                description="CrossMind model version",
            ),
            VersionComponent.PIPELINE: VersionInfo(
                component=VersionComponent.PIPELINE,
                version="3.0.0",
                description="Neuro-symbolic pipeline version",
            ),
            VersionComponent.DATA_SCHEMA: VersionInfo(
                component=VersionComponent.DATA_SCHEMA,
                version=str(settings.DATA_SCHEMA_VERSION),
                description="Data schema version",
            ),
            VersionComponent.API_SCHEMA: VersionInfo(
                component=VersionComponent.API_SCHEMA,
                version=str(settings.API_SCHEMA_VERSION),
                description="API schema version",
            ),
            VersionComponent.VECTOR_SCHEMA: VersionInfo(
                component=VersionComponent.VECTOR_SCHEMA,
                version=str(settings.VECTOR_SCHEMA_VERSION),
                description="Vector index schema version",
            ),
            VersionComponent.KG_SCHEMA: VersionInfo(
                component=VersionComponent.KG_SCHEMA,
                version=str(settings.KG_SCHEMA_VERSION),
                description="Knowledge graph schema version",
            ),
            VersionComponent.CACHE_SCHEMA: VersionInfo(
                component=VersionComponent.CACHE_SCHEMA,
                version=str(settings.CACHE_SCHEMA_VERSION),
                description="Cache schema version",
            ),
        }
        for comp, info in defaults.items():
            self.register_version(info)

    def register_version(self, version_info: VersionInfo) -> None:
        with _registry_lock:
            comp_key = version_info.component.value
            if comp_key not in _version_registry:
                _version_registry[comp_key] = {}
            _version_registry[comp_key][version_info.version] = version_info
            logger.info(f"Registered version {version_info.version} for {comp_key}")

    def get_version(self, component: VersionComponent, version: str) -> Optional[VersionInfo]:
        self._ensure_initialized()
        return _version_registry.get(component.value, {}).get(version)

    def get_latest_version(self, component: VersionComponent) -> Optional[VersionInfo]:
        self._ensure_initialized()
        versions = _version_registry.get(component.value, {})
        if not versions:
            return None
        non_deprecated = [v for v in versions.values() if not v.deprecated]
        if non_deprecated:
            return max(non_deprecated, key=lambda v: v.version)
        return max(versions.values(), key=lambda v: v.version)

    def list_versions(self, component: VersionComponent) -> List[VersionInfo]:
        self._ensure_initialized()
        return list(_version_registry.get(component.value, {}).values())

    def deprecate_version(
        self,
        component: VersionComponent,
        version: str,
        replacement_version: Optional[str] = None,
    ) -> bool:
        with _registry_lock:
            info = _version_registry.get(component.value, {}).get(version)
            if not info:
                return False
            info.deprecated = True
            info.deprecated_at = datetime.utcnow().isoformat() + "Z"
            info.replacement_version = replacement_version
            logger.warning(f"Deprecated {component.value} version {version}")
            return True

    def check_compatibility(
        self,
        component: VersionComponent,
        from_version: str,
        to_version: str,
    ) -> Optional[CompatibilityMatrix]:
        self._ensure_initialized()
        key = f"{component.value}:{from_version}->{to_version}"
        for matrix in _compatibility_matrix.get(key, []):
            if matrix.from_version == from_version and matrix.to_version == to_version:
                return matrix
        return None

    def register_compatibility(self, matrix: CompatibilityMatrix) -> None:
        with _registry_lock:
            key = f"{matrix.component.value}:{matrix.from_version}->{matrix.to_version}"
            if key not in _compatibility_matrix:
                _compatibility_matrix[key] = []
            _compatibility_matrix[key].append(matrix)

    def get_compatibility_path(
        self,
        component: VersionComponent,
        from_version: str,
        to_version: str,
    ) -> List[CompatibilityMatrix]:
        self._ensure_initialized()
        path = []
        current = from_version
        while current != to_version:
            found = False
            for matrices in _compatibility_matrix.values():
                for m in matrices:
                    if m.from_version == current and m.to_version != current:
                        path.append(m)
                        current = m.to_version
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return path


def get_version_registry() -> VersionRegistry:
    return VersionRegistry()


def get_component_version(component: VersionComponent) -> str:
    registry = get_version_registry()
    latest = registry.get_latest_version(component)
    return latest.version if latest else "unknown"


def is_version_deprecated(component: VersionComponent, version: str) -> bool:
    registry = get_version_registry()
    info = registry.get_version(component, version)
    return info.deprecated if info else False


def check_version_compatibility(
    component: VersionComponent,
    from_version: str,
    to_version: str,
) -> bool:
    registry = get_version_registry()
    matrix = registry.check_compatibility(component, from_version, to_version)
    return matrix.compatible if matrix else False