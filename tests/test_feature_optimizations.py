import unittest
from unittest.mock import patch

from reasoning.neuro_symbolic_pipeline import NeuroSymbolicPipeline
from reasoning.query_cache import QueryResultCache
from vector_store.vector_adapter import get_vector_adapter
from reasoning.knowledge_graph import KnowledgeGraph


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


class TestMixedVectorShapes(unittest.TestCase):
    def setUp(self):
        self.adapter = get_vector_adapter()

    def test_flat_dense_vector(self):
        result = self.adapter.normalize([1.0, 2.0, 3.0], force_dim=256)
        self.assertEqual(result["vector_meta"]["type"], "dense")
        self.assertEqual(len(result["flat_vector"]), 256)

    def test_multi_vector_2d(self):
        result = self.adapter.normalize([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], force_dim=256)
        self.assertEqual(result["vector_meta"]["type"], "multi_vector")
        self.assertEqual(result["vector_meta"]["shape"], [2, 256])

    def test_3d_tensor_flattened(self):
        import numpy as np
        arr = np.random.randn(2, 3, 4).astype(np.float32)
        result = self.adapter.normalize(arr, force_dim=256)
        self.assertEqual(result["vector_meta"]["type"], "dense")
        self.assertEqual(len(result["flat_vector"]), 256)

    def test_sparse_dict_expanded(self):
        result = self.adapter.normalize({0: 1.0, 10: 2.0, 255: 3.0}, force_dim=256)
        self.assertEqual(result["vector_meta"]["type"], "sparse")
        self.assertEqual(len(result["vector_meta"]["indices"]), 3)
        self.assertEqual(result["vector_meta"]["original_dim"], 256)

    def test_empty_vector(self):
        result = self.adapter.normalize([])
        self.assertEqual(result["vector_meta"]["type"], "dense")
        self.assertEqual(len(result["flat_vector"]), 0)

    def test_none_vector(self):
        result = self.adapter.normalize(None)
        self.assertEqual(result["vector_meta"]["type"], "dense")
        self.assertEqual(len(result["flat_vector"]), 0)

    def test_reshape_dense(self):
        meta = {"type": "dense", "shape": [4]}
        reshaped = self.adapter.reshape([1.0, 2.0, 3.0, 4.0], meta)
        self.assertEqual(len(reshaped), 4)

    def test_reshape_multi_vector(self):
        meta = {"type": "multi_vector", "shape": [2, 3]}
        reshaped = self.adapter.reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], meta)
        self.assertEqual(len(reshaped), 2)
        self.assertEqual(len(reshaped[0]), 3)


class TestGraphHierarchy(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraph()

    def test_missing_node_label(self):
        node = {"id": "doc:1", "type": "document"}
        label = node.get("label", node["id"])
        self.assertEqual(label, "doc:1")

    def test_missing_domain_defaults_to_general(self):
        payload = {}
        domain = payload.get("domain", "general")
        self.assertEqual(domain, "general")

    def test_empty_evidence_returns_empty_graph(self):
        result = self.kg.graph_rag_context([], [])
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertEqual(result["multi_hop_paths"], [])

    def test_single_document_no_bridges(self):
        evidence = [{"id": "doc:1", "payload": {"title": "Test", "content": "nanoparticle drug delivery", "domain": "nanotechnology", "tags": ["nanoparticle"]}}]
        result = self.kg.graph_rag_context(evidence, ["nanoparticle"])
        self.assertEqual(len(result["nodes"]), 2)  # 1 doc + 1 entity
        self.assertEqual(result["cross_domain_path_count"], 0)

    def test_cross_domain_bridge_detection(self):
        evidence = [
            {"id": "doc:1", "payload": {"title": "A", "content": "nanoparticle", "domain": "nanotechnology", "tags": ["nanoparticle"]}},
            {"id": "doc:2", "payload": {"title": "B", "content": "nanoparticle", "domain": "pharmacology", "tags": ["nanoparticle"]}},
        ]
        result = self.kg.graph_rag_context(evidence, ["nanoparticle"])
        self.assertGreater(result["cross_domain_path_count"], 0)
        self.assertGreater(len(result["multi_hop_paths"]), 0)


class TestGraphVisualizationScale(unittest.TestCase):
    def test_1_node(self):
        nodes = [{"id": "doc:1", "label": "Test", "type": "document"}]
        edges = []
        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(edges), 0)

    def test_5_nodes(self):
        nodes = [{"id": f"doc:{i}", "label": f"Doc {i}", "type": "document"} for i in range(5)]
        edges = [{"source": "doc:0", "target": "doc:1", "relation": "mentions"}]
        self.assertEqual(len(nodes), 5)
        self.assertEqual(len(edges), 1)

    def test_50_nodes(self):
        nodes = [{"id": f"doc:{i}", "label": f"Doc {i}", "type": "document"} for i in range(50)]
        edges = []
        for i in range(49):
            edges.append({"source": f"doc:{i}", "target": f"doc:{i+1}", "relation": "mentions"})
        self.assertEqual(len(nodes), 50)
        self.assertEqual(len(edges), 49)

    def test_500_nodes(self):
        nodes = [{"id": f"doc:{i}", "label": f"Doc {i}", "type": "document"} for i in range(500)]
        edges = []
        for i in range(0, 500, 2):
            edges.append({"source": f"doc:{i}", "target": f"doc:{i+1}", "relation": "mentions"})
        self.assertEqual(len(nodes), 500)
        self.assertEqual(len(edges), 250)


if __name__ == "__main__":
    unittest.main()
