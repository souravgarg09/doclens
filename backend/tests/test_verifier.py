import unittest

from app.services.verifier import (
    extract_region_citation_ids,
    normalize_region_citations,
    numeric_claims,
    uncited_factual_sentences,
    verify,
)


def region(region_id: int, values: list[float]) -> dict:
    return {
        "id": region_id,
        "type": "table",
        "content": "",
        "structured": {
            "cells": [
                {"row": index, "col": 0, "text": str(value), "value": value}
                for index, value in enumerate(values)
            ]
        },
    }


class NumericVerifierTests(unittest.TestCase):
    def test_direct_structured_value_matches(self) -> None:
        result = verify("The reported accuracy is 91.5%. [R7]", [region(7, [91.5])])
        self.assertEqual("consistent", result.status)
        self.assertTrue(result.checks[0].matched)

    def test_unrequested_arithmetic_does_not_justify_claim(self) -> None:
        result = verify("The reported result is 30.0. [R1]", [region(1, [10, 20])])
        self.assertEqual("contradiction", result.status)

    def test_explicit_difference_is_recomputed_within_one_region(self) -> None:
        result = verify("The difference is 30.0. [R1]", [region(1, [50, 20])])
        self.assertEqual("consistent", result.status)
        self.assertEqual("recomputed from cited cells", result.checks[0].note)

    def test_values_from_different_regions_are_not_combined(self) -> None:
        result = verify("The difference is 30.0. [R1] [R2]", [region(1, [50]), region(2, [20])])
        self.assertEqual("contradiction", result.status)

    def test_percentage_change_is_recomputed(self) -> None:
        result = verify("The percentage increase is 50.0%. [R1]", [region(1, [10, 15])])
        self.assertEqual("consistent", result.status)

    def test_document_labels_and_citations_are_not_claims(self) -> None:
        claims = numeric_claims("Figure 2 on page 4 reports 18.5% in 2024. [R12]")
        self.assertEqual([("Figure 2 on page 4 reports 18.5% in 2024.", 18.5)], claims)

    def test_introductory_heading_is_not_an_uncited_claim(self) -> None:
        answer = "Main findings:\n- Multiple sets produced 40% greater hypertrophy [R98]."
        self.assertEqual([], uncited_factual_sentences(answer))

    def test_grouped_region_citations_are_valid_and_normalized(self) -> None:
        answer = "Multiple sets produced 40% greater hypertrophy [R81, R98]."
        self.assertEqual([], uncited_factual_sentences(answer))
        self.assertEqual([81, 98], extract_region_citation_ids(answer))
        self.assertEqual(
            "Multiple sets produced 40% greater hypertrophy [R81] [R98].",
            normalize_region_citations(answer),
        )

    def test_uncited_prose_claim_is_reported(self) -> None:
        answer = "Multiple sets produced greater muscle hypertrophy."
        self.assertEqual([answer], uncited_factual_sentences(answer))


if __name__ == "__main__":
    unittest.main()
