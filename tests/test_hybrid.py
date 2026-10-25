import unittest
from unittest.mock import Mock
from finance_rag.retrieval import Evidence, KeywordIndex, Retriever, fuse_rankings, named_publishers


def hit(key, text, publisher="IMF"):
    return Evidence(key, text, publisher, 2025, key + ".pdf", 0, "1", None)


class HybridTests(unittest.TestCase):
    def test_keyword_search_respects_publisher_and_ignores_no_match(self):
        index = KeywordIndex([hit("a", "Euro area GDP growth"),
                              hit("b", "Eurozone GDP growth", "Goldman Sachs"), hit("c", "Bonds")])
        self.assertEqual([h.chunk_id for h in index.search("Eurozone growth", "IMF", 6)], ["a"])
        self.assertEqual(index.search("unknowntoken", None, 6), [])

    def test_fusion_deduplicates_and_rewards_agreement(self):
        a, b, c = hit("a", "a"), hit("b", "b"), hit("c", "c")
        fused = fuse_rankings([[a, b], [c, b]])
        self.assertEqual([h.chunk_id for h in fused], ["b", "a", "c"])
        self.assertIsNone(b.retrieval_score)  # Do not mutate cached evidence.

    def test_routing_uses_query_names_not_unknown_sources(self):
        self.assertEqual(named_publishers("Compare Goldman Sachs and the IMF"), ["Goldman Sachs", "IMF"])
        self.assertEqual(named_publishers("Compare growth forecasts"), [])

    def test_multiview_keeps_original_comparison_query(self):
        retriever = Retriever.__new__(Retriever)
        retriever.config = {"top_k": 6, "candidate_k": 40, "rrf_constant": 60}
        retriever.strategy = "hybrid"
        retriever.query_mode = "multi-view"
        retriever.context_mode = "none"
        retriever.dense_search = Mock(return_value=[])
        retriever.keywords = Mock()
        retriever.keywords.search.return_value = []
        retriever.search("Compare Goldman Sachs January outlook and IMF October outlook global annual growth in 2025")
        queries = [call.args[0] for call in retriever.dense_search.call_args_list]
        self.assertEqual(len(queries), 4)  # Two query views for each source.
        self.assertIn("January", queries[0])
        self.assertNotIn("january", queries[1])
        self.assertIn("annual", queries[1])
        self.assertIn("2025", queries[1])

    def test_comparison_preserves_each_source_after_reranking(self):
        retriever = Retriever.__new__(Retriever)
        retriever.config = {"top_k": 6, "candidate_k": 40, "rrf_constant": 60}
        retriever.strategy = "hybrid-rerank"
        retriever.query_mode = "original"
        retriever.context_mode = "none"
        retriever.dense_search = lambda q, source, k: [hit(source + str(i), "growth", source) for i in range(5)]
        retriever.keywords = Mock()
        retriever.keywords.search.return_value = []
        retriever.reranker = Mock()
        retriever.reranker.predict.return_value = [1, 2, 3, 4, 5]
        hits = retriever.search("Compare Goldman Sachs and IMF growth")
        self.assertEqual(len(hits), 6)
        self.assertEqual([h.publisher for h in hits], ["Goldman Sachs", "IMF"] * 3)
        self.assertEqual(hits[0].rerank_score, 5)
        with self.assertRaises(ValueError):
            retriever.search("Compare Goldman Sachs and IMF growth", k=1)
