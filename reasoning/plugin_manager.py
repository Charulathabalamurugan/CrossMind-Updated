import importlib
import importlib.metadata
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

from reasoning.agent_registry import (
    Agent,
    AgentCapability,
    AgentDescriptor,
    AgentRegistry,
    LifecycleHooks,
    RegisteredAgent,
    get_agent_registry,
)

logger = logging.getLogger("crossmind.plugin_manager")


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str = ""
    author: str = ""
    entry_point: str = ""
    dependencies: List[str] = field(default_factory=list)
    provides_agents: List[str] = field(default_factory=list)
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        caps = []
        for cap in data.get("required_capabilities", []):
            if isinstance(cap, str):
                try:
                    caps.append(AgentCapability(cap))
                except ValueError:
                    logger.warning("Unknown capability: %s", cap)
            elif isinstance(cap, AgentCapability):
                caps.append(cap)
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=data.get("entry_point", ""),
            dependencies=data.get("dependencies", []),
            provides_agents=data.get("provides_agents", []),
            required_capabilities=caps,
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "provides_agents": self.provides_agents,
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "metadata": self.metadata,
        }


class PluginLoadError(Exception):
    pass


class PluginManager:
    def __init__(self, registry: Optional[AgentRegistry] = None):
        self._registry = registry or get_agent_registry()
        self._loaded_plugins: Dict[str, PluginManifest] = {}
        self._plugin_modules: Dict[str, Any] = {}
        self._lock = __import__("threading").RLock()

    def load_from_entry_points(self, group: str = "crossmind.agents") -> List[PluginManifest]:
        loaded = []
        try:
            for entry_point in importlib.metadata.entry_points().select(group=group):
                try:
                    manifest = self._load_plugin_from_entry_point(entry_point)
                    loaded.append(manifest)
                except Exception as e:
                    logger.exception("Failed to load plugin from entry point %s", entry_point.name)
        except Exception as e:
            logger.warning("Entry point discovery failed: %s", e)
        return loaded

    def _load_plugin_from_entry_point(self, entry_point) -> PluginManifest:
        module = entry_point.load()
        if not hasattr(module, "PLUGIN_MANIFEST"):
            raise PluginLoadError(f"Plugin {entry_point.name} missing PLUGIN_MANIFEST")
        manifest = module.PLUGIN_MANIFEST
        if isinstance(manifest, dict):
            manifest = PluginManifest.from_dict(manifest)
        self._register_plugin(manifest, module)
        return manifest

    def load_from_directory(self, directory: str, pattern: str = "plugin_*.py") -> List[PluginManifest]:
        loaded = []
        path = Path(directory)
        if not path.exists():
            logger.warning("Plugin directory does not exist: %s", directory)
            return loaded

        for plugin_file in path.glob(pattern):
            try:
                manifest = self._load_plugin_from_file(plugin_file)
                loaded.append(manifest)
            except Exception as e:
                logger.exception("Failed to load plugin from file %s", plugin_file)
        return loaded

    def _load_plugin_from_file(self, plugin_file: Path) -> PluginManifest:
        spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
        if not spec or not spec.loader:
            raise PluginLoadError(f"Cannot load spec for {plugin_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_file.stem] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "PLUGIN_MANIFEST"):
            raise PluginLoadError(f"Plugin {plugin_file.name} missing PLUGIN_MANIFEST")

        manifest = module.PLUGIN_MANIFEST
        if isinstance(manifest, dict):
            manifest = PluginManifest.from_dict(manifest)

        self._register_plugin(manifest, module)
        return manifest

    def _register_plugin(self, manifest: PluginManifest, module: Any):
        with self._lock:
            if manifest.name in self._loaded_plugins:
                logger.warning("Plugin %s already loaded, skipping", manifest.name)
                return

            self._check_dependencies(manifest)
            self._loaded_plugins[manifest.name] = manifest
            self._plugin_modules[manifest.name] = module

            for agent_class_name in manifest.provides_agents:
                if hasattr(module, agent_class_name):
                    agent_class = getattr(module, agent_class_name)
                    self._register_agent_class(agent_class, manifest, module)
                else:
                    logger.warning("Agent class %s not found in plugin %s", agent_class_name, manifest.name)

            logger.info("Loaded plugin: %s v%s", manifest.name, manifest.version)

    def _check_dependencies(self, manifest: PluginManifest):
        for dep in manifest.dependencies:
            if dep not in self._loaded_plugins:
                raise PluginLoadError(f"Missing dependency: {dep} (required by {manifest.name})")

    def _register_agent_class(self, agent_class: Type[Agent], manifest: PluginManifest, module: Any):
        try:
            agent_instance = agent_class()
            descriptor = agent_instance.get_descriptor()

            hooks = LifecycleHooks()
            if hasattr(module, "on_agent_register"):
                hooks.on_register = module.on_agent_register
            if hasattr(module, "on_agent_start"):
                hooks.on_start = module.on_agent_start
            if hasattr(module, "on_agent_stop"):
                hooks.on_stop = module.on_agent_stop
            if hasattr(module, "on_agent_error"):
                hooks.on_error = module.on_agent_error
            if hasattr(module, "on_agent_health_check"):
                hooks.on_health_check = module.on_agent_health_check

            self._registry.register(agent_instance, hooks=hooks)
            logger.info("Registered agent %s from plugin %s", descriptor.agent_id, manifest.name)
        except Exception as e:
            logger.exception("Failed to register agent %s from plugin %s", agent_class.__name__, manifest.name)
            raise

    def unload_plugin(self, name: str) -> bool:
        with self._lock:
            manifest = self._loaded_plugins.pop(name, None)
            module = self._plugin_modules.pop(name, None)

            if not manifest:
                return False

            for agent_class_name in manifest.provides_agents:
                if hasattr(module, agent_class_name):
                    agent_class = getattr(module, agent_class_name)
                    try:
                        temp_instance = agent_class()
                        self._registry.unregister(temp_instance.get_descriptor().agent_id)
                    except Exception as e:
                        logger.warning("Failed to unregister agent %s from plugin %s: %s", agent_class_name, name, e)

            logger.info("Unloaded plugin: %s", name)
            return True

    def get_plugin(self, name: str) -> Optional[PluginManifest]:
        with self._lock:
            return self._loaded_plugins.get(name)

    def list_plugins(self) -> List[PluginManifest]:
        with self._lock:
            return list(self._loaded_plugins.values())

    def resolve_dependencies(self, manifests: List[PluginManifest]) -> List[PluginManifest]:
        resolved: List[PluginManifest] = []
        remaining = {m.name: m for m in manifests}

        while remaining:
            progress = False
            for name, manifest in list(remaining.items()):
                deps_met = all(dep in self._loaded_plugins or dep in [r.name for r in resolved] for dep in manifest.dependencies)
                if deps_met:
                    resolved.append(manifest)
                    del remaining[name]
                    progress = True

            if not progress:
                raise PluginLoadError(f"Circular or missing dependencies: {[m.name for m in remaining.values()]}")

        return resolved


_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager(registry: Optional[AgentRegistry] = None) -> PluginManager:
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager(registry)
    return _global_plugin_manager


def reset_plugin_manager() -> None:
    global _global_plugin_manager
    _global_plugin_manager = None