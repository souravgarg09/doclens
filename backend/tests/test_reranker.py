import unittest
from unittest import mock

from app.services import reranker


def cand(region_id: int, fused: float, score: float, rtype: str = "text", lexical: float = 0.0) -> dict:
    return {"id": region_id, "region_id": region_id, "type": rtype, "page": 1, "content": "", "structured": {},
            "fused": fused, "score": score, "lexical": lexical}


@mock.patch.object(reranker.settings, "gemini_rerank", False)
@mock.patch.object(reranker.embeddings, "text_enabled", lambda: True)
class RerankTests(unittest.TestCase):
    def test_fused_rank_decides_the_order(self) -> None:
        kept, _, _ = reranker.rerank("what did they find?", [cand(1, 0.02, 0.70), cand(2, 0.03, 0.60)])
        self.assertEqual([2, 1], [c["id"] for c in kept])

    def test_named_type_is_preferred(self) -> None:
        kept, _, _ = reranker.rerank("what does the table show?", [cand(1, 0.030, 0.7), cand(2, 0.028, 0.7, "table")])
        self.assertEqual(2, kept[0]["id"])

    def test_grounding_is_calibrated_on_the_text_cosine(self) -> None:
        _, on_topic, _ = reranker.rerank("q", [cand(1, 0.03, 0.70)])
        _, off_topic, _ = reranker.rerank("q", [cand(1, 0.03, 0.53)])
        self.assertGreaterEqual(on_topic, 0.35)  # answered
        self.assertLess(off_topic, 0.35)  # refused without a Gemini call

    def test_relevance_mapping(self) -> None:
        self.assertEqual(0.0, reranker.relevance(0.40))
        self.assertAlmostEqual(0.5, reranker.relevance(0.60))
        self.assertEqual(1.0, reranker.relevance(0.90))


if __name__ == "__main__":
    unittest.main()
