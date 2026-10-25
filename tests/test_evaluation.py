import unittest

from finance_rag.evaluation import normalize, score_hits
from finance_rag.retrieval import Evidence
from dataclasses import replace


def hit(source, text):
    return Evidence("id", text, "publisher", 2025, source, 0, "1", 0.5)


class EvaluationTests(unittest.TestCase):
    def test_two_sources_need_both_passages_for_full_credit(self):
        gold = [{"source": "gs.pdf", "excerpt": "Growth is 3.1%"},
                {"source": "imf.pdf", "excerpt": "Growth is 3.2%"}]
        score = score_hits([hit("other.pdf", "Nothing"), hit("gs.pdf", "Growth is 3.1%")], gold)
        self.assertEqual(score["evidence_recall"], 0.5)
        self.assertFalse(score["all_evidence_found"])
        self.assertEqual(score["reciprocal_rank"], 0.5)
        self.assertEqual(score["required_evidence_ranks"], [2, None])

    def test_same_number_from_wrong_source_does_not_count(self):
        score = score_hits([hit("jpm.pdf", "Growth is 3.1%")], [{"source": "gs.pdf", "excerpt": "Growth is 3.1%"}])
        self.assertEqual(score["evidence_recall"], 0)

    def test_duplicate_hits_do_not_inflate_evidence_recall(self):
        gold = [{"source": "gs.pdf", "excerpt": "Growth is 3.1%"}, {"source": "imf.pdf", "excerpt": "3.2%"}]
        score = score_hits([hit("gs.pdf", "Growth is 3.1%"), hit("gs.pdf", "Growth is 3.1%")], gold)
        self.assertEqual(score["evidence_recall"], 0.5)

    def test_surrounding_evidence_does_not_inflate_original_chunk_score(self):
        anchor = hit("gs.pdf", "Forecast discussion")
        anchor.context_text = "Forecast discussion. Growth is 3.1%"
        gold = [{"source": "gs.pdf", "excerpt": "Growth is 3.1%"}]
        self.assertEqual(score_hits([anchor], gold)["evidence_recall"], 0)
        self.assertEqual(score_hits([replace(anchor, text=anchor.context_text)], gold)["evidence_recall"], 1)

    def test_abstention_cases_are_not_scored_as_failed_retrieval(self):
        self.assertIsNone(score_hits([hit("gs.pdf", "Something")], []))

    def test_pdf_whitespace_and_unicode_ligatures_are_normalized(self):
        self.assertEqual(normalize("In\n ﬂation"), normalize("In flation"))


if __name__ == "__main__":
    unittest.main()
