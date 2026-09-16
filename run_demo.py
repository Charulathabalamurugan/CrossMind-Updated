"""CrossMind end-to-end demo: runs ingestion, reasoning, and API."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=== CrossMind Demo ===")
    print()

    print("Step 1: Health check")
    from app.main import app
    print("FastAPI app loaded successfully")
    print()

    print("Step 2: Ingestion pipeline")
    from ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    print("Ingestion pipeline ready")
    print()

    print("Step 3: Reasoning pipeline")
    from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
    reasoning = NeuroSymbolicPipeline()
    result = reasoning.process_query(query="test query", user_role="researcher")
    print("Reasoning result keys: " + str(list(result.keys())))
    print()

    print("Step 4: API endpoints")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    health = client.get("/healthz")
    print("Health endpoint: " + str(health.status_code) + " " + str(health.json()))
    root = client.get("/")
    print("Root endpoint: " + str(root.status_code) + " " + str(root.json()))
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
