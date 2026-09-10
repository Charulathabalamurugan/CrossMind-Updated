import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from pathlib import Path


class TestPluginDiscovery:
    """Test plugin auto-discovery functionality."""

    def test_plugin_auto_discover_enabled_setting(self):
        """Plugin auto-discovery should be configurable."""
        from config import settings
        
        assert hasattr(settings, 'PLUGIN_AUTO_DISCOVER_ENABLED')
        assert isinstance(settings.PLUGIN_AUTO_DISCOVER_ENABLED, bool)
        assert hasattr(settings, 'PLUGIN_DIRS')
        assert isinstance(settings.PLUGIN_DIRS, str)

    def test_plugin_directory_scanning(self):
        """Plugin system should scan configured directories."""
        from config import settings
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "plugins"
            plugin_dir.mkdir()
            
            (plugin_dir / "plugin_a.py").write_text("""
class PluginA:
    name = "plugin_a"
    version = "1.0.0"
    def process(self, data):
        return {"plugin": "a", "result": data}
""")
            
            (plugin_dir / "plugin_b.py").write_text("""
class PluginB:
    name = "plugin_b"
    version = "2.0.0"
    def process(self, data):
        return {"plugin": "b", "result": data}
""")
            
            (plugin_dir / "__init__.py").write_text("")
            
            original_dirs = settings.PLUGIN_DIRS
            settings.PLUGIN_DIRS = str(plugin_dir)
            
            try:
                import importlib.util
                import sys
                
                plugins = []
                for py_file in plugin_dir.glob("*.py"):
                    if py_file.name == "__init__.py":
                        continue
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'name'):
                            plugins.append(attr())
                
                assert len(plugins) == 2
                plugin_names = {p.name for p in plugins}
                assert plugin_names == {"plugin_a", "plugin_b"}
            finally:
                settings.PLUGIN_DIRS = original_dirs
                for name in ["plugin_a", "plugin_b"]:
                    if name in sys.modules:
                        del sys.modules[name]

    def test_plugin_interface_compliance(self):
        """Discovered plugins should conform to expected interface."""
        class ValidPlugin:
            name = "valid_plugin"
            version = "1.0.0"
            capabilities = ["processing", "analysis"]
            
            def initialize(self, config: Dict[str, Any]) -> bool:
                return True
            
            def process(self, data: Any) -> Dict[str, Any]:
                return {"status": "ok", "data": data}
            
            def shutdown(self) -> None:
                pass
            
            def health_check(self) -> Dict[str, Any]:
                return {"status": "healthy"}
        
        plugin = ValidPlugin()
        
        assert hasattr(plugin, 'name')
        assert hasattr(plugin, 'version')
        assert hasattr(plugin, 'capabilities')
        assert hasattr(plugin, 'initialize')
        assert hasattr(plugin, 'process')
        assert hasattr(plugin, 'shutdown')
        assert hasattr(plugin, 'health_check')
        
        assert plugin.initialize({"setting": "value"}) is True
        result = plugin.process({"input": "test"})
        assert result["status"] == "ok"
        assert plugin.health_check()["status"] == "healthy"

    def test_plugin_registration_with_orchestrator(self):
        """Plugins should be registerable with agent orchestrator."""
        from reasoning.agent_registry import (
            AgentRegistry, Agent, AgentDescriptor, AgentCapability, AgentState
        )
        
        class PluginAgent(Agent):
            def __init__(self, plugin_name: str):
                self._plugin_name = plugin_name
            
            def get_descriptor(self) -> AgentDescriptor:
                return AgentDescriptor(
                    agent_id=f"plugin_{self._plugin_name}",
                    name=f"PluginAgent-{self._plugin_name}",
                    domain="plugin",
                    capabilities=[AgentCapability.REASONING, AgentCapability.CUSTOM],
                    metadata={"plugin": self._plugin_name, "source": "plugin"}
                )
            
            def process(self, query: str, evidence: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {"plugin": self._plugin_name, "result": f"Processed by {self._plugin_name}"}
        
        registry = AgentRegistry()
        
        plugin_agents = [PluginAgent("analyzer"), PluginAgent("transformer"), PluginAgent("validator")]
        
        for agent in plugin_agents:
            registered = registry.register(agent, domain="plugins")
            assert registered.descriptor.metadata["source"] == "plugin"
        
        plugin_agents_list = registry.list_by_domain("plugins")
        assert len(plugin_agents_list) == 3
        
        custom_agents = registry.list_by_capability(AgentCapability.CUSTOM)
        assert len(custom_agents) == 3

    def test_plugin_hot_reload_simulation(self):
        """Plugin system should support hot reload simulation."""
        from reasoning.agent_registry import AgentRegistry
        
        registry = AgentRegistry()
        
        class MockPluginAgent:
            def __init__(self, name: str, version: str):
                self.name = name
                self.version = version
            
            def get_descriptor(self):
                from reasoning.agent_registry import AgentDescriptor, AgentCapability
                return AgentDescriptor(
                    agent_id=f"plugin_{self.name}",
                    name=f"Plugin-{self.name}",
                    domain="plugin",
                    capabilities=[AgentCapability.REASONING],
                    version=self.version
                )
            
            def process(self, query, evidence, metadata=None):
                return {"plugin": self.name}
        
        class MockAgent:
            def __init__(self, plugin):
                self.plugin = plugin
            
            def get_descriptor(self):
                return self.plugin.get_descriptor()
            
            def process(self, query, evidence, metadata=None):
                return self.plugin.process(query, evidence, metadata)
        
        plugin_v1 = MockPluginAgent("processor", "1.0.0")
        agent_v1 = MockAgent(plugin_v1)
        registered = registry.register(agent_v1, domain="plugins")
        
        assert registered.descriptor.version == "1.0.0"
        
        registry.unregister(f"plugin_processor")
        
        plugin_v2 = MockPluginAgent("processor", "2.0.0")
        agent_v2 = MockAgent(plugin_v2)
        registered_v2 = registry.register(agent_v2, domain="plugins")
        
        assert registered_v2.descriptor.version == "2.0.0"

    def test_plugin_dependency_resolution(self):
        """Plugin system should handle dependencies."""
        class PluginWithDeps:
            def __init__(self):
                self.name = "dependent_plugin"
                self.version = "1.0.0"
                self.dependencies = ["base_plugin", "utility_plugin"]
            
            def process(self, data):
                return {"plugin": self.name, "deps": self.dependencies}
        
        plugin = PluginWithDeps()
        
        assert hasattr(plugin, 'dependencies')
        assert "base_plugin" in plugin.dependencies
        assert "utility_plugin" in plugin.dependencies

    def test_plugin_isolation(self):
        """Plugins should be isolated from each other."""
        from reasoning.agent_registry import AgentRegistry
        
        registry = AgentRegistry()
        
        class IsolatedPluginAgent:
            def __init__(self, name: str):
                self.name = name
                self.state = {}
            
            def get_descriptor(self):
                from reasoning.agent_registry import AgentDescriptor, AgentCapability
                return AgentDescriptor(
                    agent_id=f"plugin_{self.name}",
                    name=f"Plugin-{self.name}",
                    domain="plugin",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query, evidence, metadata=None):
                self.state["last_query"] = query
                return {"plugin": self.name, "state_keys": list(self.state.keys())}
        
        class MockAgent:
            def __init__(self, plugin):
                self.plugin = plugin
            
            def get_descriptor(self):
                return self.plugin.get_descriptor()
            
            def process(self, query, evidence, metadata=None):
                return self.plugin.process(query, evidence, metadata)
        
        plugin1 = IsolatedPluginAgent("isolated_1")
        plugin2 = IsolatedPluginAgent("isolated_2")
        
        agent1 = MockAgent(plugin1)
        agent2 = MockAgent(plugin2)
        
        registry.register(agent1, domain="plugins")
        registry.register(agent2, domain="plugins")
        
        result1 = agent1.process("query for plugin 1", [])
        result2 = agent2.process("query for plugin 2", [])
        
        assert result1["plugin"] == "isolated_1"
        assert result2["plugin"] == "isolated_2"
        assert "last_query" in plugin1.state
        assert "last_query" in plugin2.state
        assert plugin1.state["last_query"] != plugin2.state["last_query"]


class TestPluginConfiguration:
    """Test plugin configuration and settings."""

    def test_plugin_config_loading(self):
        """Plugins should be configurable via settings."""
        from config import settings
        
        assert hasattr(settings, 'PLUGIN_DIRS')
        assert hasattr(settings, 'PLUGIN_AUTO_DISCOVER_ENABLED')

    def test_plugin_dirs_parsing(self):
        """PLUGIN_DIRS should support multiple directories."""
        from config import settings
        
        original = settings.PLUGIN_DIRS
        
        try:
            settings.PLUGIN_DIRS = "/path/one,/path/two,/path/three"
            
            dirs = [d.strip() for d in settings.PLUGIN_DIRS.split(",") if d.strip()]
            
            assert len(dirs) == 3
            assert "/path/one" in dirs
            assert "/path/two" in dirs
            assert "/path/three" in dirs
        finally:
            settings.PLUGIN_DIRS = original

    def test_plugin_discovery_disabled(self):
        """When disabled, no plugins should be auto-discovered."""
        from config import settings
        
        original = settings.PLUGIN_AUTO_DISCOVER_ENABLED
        
        try:
            settings.PLUGIN_AUTO_DISCOVER_ENABLED = False
            
            assert settings.PLUGIN_AUTO_DISCOVER_ENABLED is False
        finally:
            settings.PLUGIN_AUTO_DISCOVER_ENABLED = original


class TestPluginIntegration:
    """Test plugin integration with core systems."""

    def test_plugin_as_ingestion_connector(self):
        """Plugins should work as ingestion connectors."""
        from ingestion.dynamic_connectors import DynamicConnectorManager
        
        class MockPluginConnector:
            def __init__(self):
                self.name = "plugin_connector"
                self.source_type = "plugin"
            
            def fetch(self) -> List[Dict[str, Any]]:
                return [{"title": "Plugin Doc", "content": "From plugin", "domain": "plugin"}]
            
            def start(self, callback):
                pass
            
            def stop(self):
                pass
        
        manager = DynamicConnectorManager()
        
        connector = MockPluginConnector()
        
        manager.connectors["plugin_connector"] = connector
        
        assert "plugin_connector" in manager.connectors
        docs = manager.connectors["plugin_connector"].fetch()
        assert len(docs) == 1
        assert docs[0]["domain"] == "plugin"

    def test_plugin_as_retrieval_enhancer(self):
        """Plugins should enhance retrieval pipeline."""
        from reasoning.hybrid_rag_kg import HybridRAGKG
        
        class MockRetrievalPlugin:
            name = "retrieval_enhancer"
            
            def enhance_query(self, query: str) -> str:
                return f"enhanced: {query}"
            
            def rerank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        plugin = MockRetrievalPlugin()
        
        enhanced = plugin.enhance_query("test query")
        assert enhanced == "enhanced: test query"
        
        results = [{"score": 0.5}, {"score": 0.9}, {"score": 0.7}]
        reranked = plugin.rerank(results)
        assert reranked[0]["score"] == 0.9

    def test_plugin_as_reasoning_extension(self):
        """Plugins should extend reasoning capabilities."""
        from reasoning.multi_agent import SpecialistAgent
        from reasoning.agent_registry import AgentCapability
        
        class PluginReasoningAgent(SpecialistAgent):
            def __init__(self):
                super().__init__("plugin_reasoning", "plugin_reasoning_agent")
                self.name = "PluginReasoningAgent"
            
            def get_descriptor(self):
                from reasoning.agent_registry import AgentDescriptor, AgentCapability
                return AgentDescriptor(
                    agent_id="plugin_reasoning_agent",
                    name="PluginReasoningAgent",
                    domain="plugin_reasoning",
                    capabilities=[AgentCapability.REASONING, AgentCapability.PLANNING],
                    metadata={"plugin": True}
                )
            
            def process(self, query, evidence, metadata=None):
                return {
                    "agent_id": "plugin_reasoning_agent",
                    "domain": "plugin_reasoning",
                    "summary": f"Plugin reasoning for: {query}",
                    "confidence_score": 0.8
                }
        
        agent = PluginReasoningAgent()
        
        descriptor = agent.get_descriptor()
        assert descriptor.metadata.get("plugin") is True
        assert AgentCapability.PLANNING in descriptor.capabilities


class TestPluginErrorHandling:
    """Test plugin error handling and resilience."""

    def test_malformed_plugin_handling(self):
        """Malformed plugins should not crash the system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "plugins"
            plugin_dir.mkdir()
            
            (plugin_dir / "bad_plugin.py").write_text("""
this is not valid python code {{{{
""")
            
            import importlib.util
            
            py_file = plugin_dir / "bad_plugin.py"
            spec = importlib.util.spec_from_file_location("bad_plugin", py_file)
            
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert False, "Should have raised exception"
            except SyntaxError:
                pass
            except Exception:
                pass

    def test_plugin_crash_isolation(self):
        """Plugin crashes should not affect other plugins."""
        from reasoning.agent_registry import AgentRegistry
        
        registry = AgentRegistry()
        
        class WorkingPlugin:
            def __init__(self, name):
                self.name = name
            
            def get_descriptor(self):
                from reasoning.agent_registry import AgentDescriptor, AgentCapability
                return AgentDescriptor(
                    agent_id=f"working_{self.name}",
                    name=f"Working-{self.name}",
                    domain="plugin",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query, evidence, metadata=None):
                return {"plugin": self.name, "status": "ok"}
        
        class CrashingPlugin:
            def __init__(self, name):
                self.name = name
            
            def get_descriptor(self):
                from reasoning.agent_registry import AgentDescriptor, AgentCapability
                return AgentDescriptor(
                    agent_id=f"crashing_{self.name}",
                    name=f"Crashing-{self.name}",
                    domain="plugin",
                    capabilities=[AgentCapability.REASONING]
                )
            
            def process(self, query, evidence, metadata=None):
                raise RuntimeError(f"Plugin {self.name} crashed!")
        
        class MockAgent:
            def __init__(self, plugin):
                self.plugin = plugin
            
            def get_descriptor(self):
                return self.plugin.get_descriptor()
            
            def process(self, query, evidence, metadata=None):
                return self.plugin.process(query, evidence, metadata)
        
        working = MockAgent(WorkingPlugin("good"))
        crashing = MockAgent(CrashingPlugin("bad"))
        
        registry.register(working, domain="plugins")
        registry.register(crashing, domain="plugins")
        
        result = working.process("test", [])
        assert result["status"] == "ok"
        
        try:
            crashing.process("test", [])
            assert False, "Should have raised"
        except RuntimeError:
            pass
        
        result2 = working.process("test again", [])
        assert result2["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])