import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("QDRANT_IN_MEMORY", "true")
os.environ.setdefault("NEO4J_ENABLED", "false")

if __name__ == "__main__":
    exit_code = pytest.main([
        "tests/integration",
        "-v",
        "--tb=short",
        "-x",
        "--asyncio-mode=auto"
    ])
    sys.exit(exit_code)