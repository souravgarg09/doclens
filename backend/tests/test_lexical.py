import unittest

from app.services import lexical


class Bm25Tests(unittest.TestCase):
    def test_exact_term_and_number_rank_first(self):
        docs = [
            "Training volume and muscle strength in older adults.",
            "Multiple sets produced an effect size of 0.24 for hypertrophy.",
            "Table 2 lists the included studies.",
        ]
        scores = lexical.bm25_scores("effect size hypertrophy 0.24", docs)
        self.assertEqual(max(range(3), key=scores.__getitem__), 1)
        self.assertEqual(scores[2], 0.0)

    def test_stopwords_only_query_scores_zero(self):
        self.assertEqual(lexical.bm25_scores("what is the", ["the cat is here"]), [0.0])

    def test_empty_inputs(self):
        self.assertEqual(lexical.bm25_scores("anything", []), [])

    def test_tokenizer_keeps_decimals(self):
        self.assertEqual(lexical.tokenize("ES = 0.24 (95% CI)"), ["es", "0.24", "95", "ci"])


class FusionTests(unittest.TestCase):
    def test_item_ranked_well_in_both_lists_wins(self):
        fused = lexical.reciprocal_rank_fusion([[1, 2, 3], [2, 3, 9]])
        self.assertEqual(max(fused, key=fused.get), 2)
        self.assertIn(9, fused)

    def test_weights_change_the_winner(self):
        fused = lexical.reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[1.0, 3.0])
        self.assertEqual(max(fused, key=fused.get), 2)


if __name__ == "__main__":
    unittest.main()
