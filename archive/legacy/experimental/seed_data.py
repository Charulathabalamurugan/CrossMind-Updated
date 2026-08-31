from typing import Any, Dict, List


def build_seed_data() -> List[Dict[str, Any]]:
    return [
        {
            "id": "seed-1",
            "title": "CrossMind Scientific Seed",
            "content": "This is a minimal seed dataset for the demo knowledge base.",
            "domain": "general",
            "tags": ["demo", "seed", "general"],
        }
    ]


__all__ = ["build_seed_data"]
