import unittest
from unittest.mock import patch

from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
from reasoning.query_cache import QueryResultCache


class TestReasoningRouter(unittest.TestCase):
    def test_medium_complexity_routes_to_zaya1b(self):
        pipeline = NeuroSymbolicPipeline.__new__(NeuroSymbolicPipeline)
        self.assertEqual(pipeline._route_reasoning_model("low"), "lite_llm")
        self.assertEqual(pipeline._route_reasoning_model("medium"), "zaya1b")
        self.assertEqual(pipeline._route_reasoning_model("high"), "zaya1_8b")


class TestEvidenceCompression(unittest.TestCase):
    def test_compression_keeps_query_relevant_sentences(self):
        pipeline = NeuroSymbolicPipeline.__new__(NeuroSymbolicPipeline)
        evidence = [
            {
                "payload": {
                    "title": "Example",
                    "content": "This sentence mentions the target drug mechanism. Another sentence is irrelevant noise. The target drug mechanism is still relevant and should be preserved.",
                }
            }
        ]

        compressed = pipeline._compress_evidence_context("target drug mechanism", evidence)

        self.assertEqual(len(compressed), 1)
        self.assertIn("target drug mechanism", compressed[0]["payload"]["content"])
        self.assertIn("relevant", compressed[0]["payload"]["content"])


class TestSemanticQueryCache(unittest.TestCase):
    def test_delete_clears_embedded_cache_entries(self):
        with patch("reasoning.query_cache.get_embedder") as mock_embedder:
            mock_embedder.return_value.embed_text.return_value = [1.0, 0.0, 0.0]
            cache = QueryResultCache()
            cache.set("alpha", {"value": 1}, query="hello world")
            cache.delete("alpha")
            self.assertIsNone(cache.get("alpha"))
            self.assertNotIn("alpha", cache._query_embeddings)


if __name__ == "__main__":
    unittest.main()
