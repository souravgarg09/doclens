import unittest

from workers.parsers.layout import dedupe_regions


def region(rtype, bbox, text, conf=0.9):
    return {"type": rtype, "bbox": bbox, "text": text, "conf": conf}


class DedupeTests(unittest.TestCase):
    def test_identical_twins_keep_one(self):
        a = region("text", [0.1, 0.2, 0.4, 0.25], "JAMES W. KRIEGER Journal", 0.8)
        b = region("text", [0.1, 0.2, 0.4, 0.25], "JAMES W. KRIEGER Journal", 0.9)
        kept = dedupe_regions([a, b])
        self.assertEqual([b], kept)

    def test_parts_inside_a_block_are_dropped(self):
        block = region("text", [0.1, 0.2, 0.4, 0.25], "JAMES W. KRIEGER Journal of Pure Power")
        line1 = region("text", [0.1, 0.2, 0.25, 0.23], "JAMES W. KRIEGER")
        line2 = region("text", [0.1, 0.23, 0.4, 0.25], "Journal of Pure Power")
        self.assertEqual([block], dedupe_regions([block, line1, line2]))

    def test_same_box_as_text_and_title_keeps_one(self):
        a = region("text", [0.1, 0.9, 0.4, 0.92], "1156 Journal of Strength")
        b = region("title", [0.1, 0.9, 0.4, 0.92], "1156 Journal of Strength")
        self.assertEqual(1, len(dedupe_regions([a, b])))

    def test_empty_text_regions_are_dropped_but_visuals_kept(self):
        empty = region("text", [0.0, 0.97, 1.0, 0.99], "   ")
        chart = region("chart", [0.3, 0.6, 0.8, 0.9], "")
        self.assertEqual([chart], dedupe_regions([empty, chart]))

    def test_neighbouring_paragraphs_are_kept(self):
        p1 = region("text", [0.1, 0.1, 0.5, 0.3], "First paragraph about sets.")
        p2 = region("text", [0.1, 0.3, 0.5, 0.5], "Second paragraph about volume.")
        self.assertEqual([p1, p2], dedupe_regions([p1, p2]))

    def test_text_inside_box_but_different_words_is_kept(self):
        big = region("text", [0.1, 0.1, 0.9, 0.9], "Results section text.")
        small = region("text", [0.2, 0.2, 0.3, 0.3], "A different footnote.")
        self.assertEqual(2, len(dedupe_regions([big, small])))


if __name__ == "__main__":
    unittest.main()
