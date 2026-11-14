import unittest

import pymupdf

from app.agents.graph import _REFERENCE_RE
from workers.parsers import links


def _paper(with_links: bool) -> pymupdf.Document:
    """A two-page paper: body text citing [2] / (2) and a numbered reference list."""
    pdf = pymupdf.open()
    body = pdf.new_page()
    body.insert_text((72, 100), "Residual learning was introduced earlier [2]. See Table 1." if with_links
                     else "Residual learning was introduced earlier (2). See Table 1.", fontsize=11)
    body.insert_text((72, 130), "Code is available online.", fontsize=11)
    refs = pdf.new_page()
    refs.insert_text((72, 80), "References", fontsize=12)
    entries = [
        "[1] A. Author. A first paper. In CVPR, 2015.",
        "[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning",
        "for image recognition. In CVPR, 2016.",
        "[3] C. Third. Another paper. arXiv, 2017.",
    ]
    y = 110
    for line in entries:
        refs.insert_text((90 if line.startswith("for") else 72, y), line, fontsize=10)
        y += 14
    if with_links:
        body = pdf[0]  # page objects go stale once another page is added
        cite = body.search_for("[2]")[0]
        body.insert_link({"kind": pymupdf.LINK_GOTO, "from": cite, "page": 1, "to": pymupdf.Point(72, 124 - 10)})
        url = body.search_for("Code")[0]
        body.insert_link({"kind": pymupdf.LINK_URI, "from": url, "uri": "https://github.com/example/resnet"})
    return pdf


class ReferenceListTests(unittest.TestCase):
    def test_numbered_entries_with_continuation_lines(self) -> None:
        refs = links.parse_references(_paper(with_links=False))
        self.assertEqual(["1", "2", "3"], sorted(refs))
        self.assertEqual("K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.",
                         refs["2"]["text"])
        self.assertEqual(2, refs["2"]["page"])

    def test_no_reference_heading_means_no_entries(self) -> None:
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 100), "1. A numbered list that is not a bibliography.")
        self.assertEqual({}, links.parse_references(pdf))


class ExtractLinksTests(unittest.TestCase):
    def test_citation_and_url_links_are_resolved(self) -> None:
        found = links.extract_links(_paper(with_links=True))
        citation = next(link for link in found if link["label"] == "2")
        self.assertEqual("2", citation["label"])
        self.assertEqual(2, citation["target_page"])
        self.assertTrue(citation["target_text"].startswith("[2] K. He"), citation["target_text"])
        url = next(link for link in found if link["uri"])
        self.assertEqual("url", url["kind"])
        self.assertEqual("Code", url["text"])

    def test_uri_kinds(self) -> None:
        self.assertEqual("doi", links._uri_kind("https://doi.org/10.1109/CVPR.2016.90"))
        self.assertEqual("email", links._uri_kind("mailto:someone@example.org"))
        self.assertEqual("url", links._uri_kind("https://github.com/x/y"))


class RowBuilderTests(unittest.TestCase):
    def test_inferred_citations_when_the_pdf_has_no_links(self) -> None:
        pdf = _paper(with_links=False)
        regions = [
            {"id": 10, "page": 1, "type": "text", "bbox": [0.05, 0.05, 0.95, 0.25],
             "content": "Residual learning was introduced earlier (2). See Table 1."},
            {"id": 20, "page": 2, "type": "text", "bbox": [0.05, 0.08, 0.95, 0.3], "content": "References ..."},
        ]
        rows = links.build_link_rows(pdf, regions)
        reference = next(r for r in rows if r["kind"] == "reference" and r["label"] == "2")
        self.assertEqual(20, reference["region_id"])
        inferred = [r for r in rows if r["kind"] == "citation"]
        self.assertEqual(1, len(inferred))
        self.assertTrue(inferred[0]["inferred"])
        self.assertEqual(10, inferred[0]["region_id"])
        self.assertEqual(20, inferred[0]["target_region_id"])
        self.assertIn("Deep residual learning", inferred[0]["target_text"])

    def test_real_citation_links_suppress_inferred_ones(self) -> None:
        rows = links.build_link_rows(_paper(with_links=True), [
            {"id": 10, "page": 1, "type": "text", "bbox": [0.05, 0.05, 0.95, 0.25], "content": "earlier [2]"}])
        self.assertFalse(any(r["inferred"] for r in rows))
        citation = next(r for r in rows if r["kind"] == "citation")
        self.assertEqual(10, citation["region_id"])
        self.assertIn("for image recognition", citation["target_text"])


class CitationMarkerTests(unittest.TestCase):
    def test_ranges_and_lists_are_expanded(self) -> None:
        self.assertEqual(["4", "27", "28", "29"], links.citation_numbers("as shown [4, 27-29].", None))

    def test_only_known_reference_numbers_count(self) -> None:
        self.assertEqual(["11"], links.citation_numbers("Galvao and Taaffe (11) with n = (300)", {"11", "12"}))

    def test_years_are_not_citations(self) -> None:
        self.assertEqual([], links.citation_numbers("published (2016)", {"1", "2"}))

    def test_link_label(self) -> None:
        self.assertEqual("11", links._label("[11]"))
        self.assertEqual("27", links._label("27,"))
        self.assertIsNone(links._label("Table 2"))


class QuestionReferenceTests(unittest.TestCase):
    def test_reference_numbers_in_questions(self) -> None:
        found = [a or b for a, b in _REFERENCE_RE.findall("Which study is reference 11? And ref. [12], citation 3 or [7]?")]
        self.assertEqual(["11", "12", "3", "7"], found)

    def test_tables_and_years_are_not_references(self) -> None:
        self.assertEqual([], _REFERENCE_RE.findall("What does Table 2 show for 2019?"))


if __name__ == "__main__":
    unittest.main()


class ChartTableCheckTests(unittest.TestCase):
    def test_real_deplot_tables_pass(self) -> None:
        from workers.parsers.charts import is_table
        self.assertTrue(is_table("TITLE |  <0x0A>  | HypertrophySS <0x0A> Single Set | 0.25 <0x0A> Multiple Sets | 0.35"))
        self.assertTrue(is_table("TITLE |  <0x0A> HypertrophyES |  <0x0A> 1 set | 0.24 <0x0A> 2-3 sets | 0.34 <0x0A> 4-6 sets | 0.4"))

    def test_word_loops_from_diagrams_are_rejected(self) -> None:
        from workers.parsers.charts import is_table
        junk = "TITLE |  <0x0A> The<0x0A>Law<0x0A>Law<0x0A>Will<0x0A>will<0x0A>never<0x0A>never<0x0A>be<0x0A>be<0x0A>perfect"
        self.assertFalse(is_table(junk))
        self.assertFalse(is_table(junk, final=False))  # stops generation early too
        self.assertTrue(is_table("TITLE |  <0x0A>  | Strength", final=False))  # too early to judge


class UrlIntentTests(unittest.TestCase):
    def test_questions_asking_for_a_web_address(self) -> None:
        from app.agents.graph import _WANTS_URL_RE
        for q in ("Where can I find the code?", "Is the dataset available online?", "What is the GitHub repo?"):
            self.assertIsNotNone(_WANTS_URL_RE.search(q), q)

    def test_ordinary_questions_do_not_ask_for_links(self) -> None:
        from app.agents.graph import _WANTS_URL_RE
        for q in ("What BLEU score did the big model reach?", "How many studies were included?", "encoded inputs"):
            self.assertIsNone(_WANTS_URL_RE.search(q), q)
