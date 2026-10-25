import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from finance_rag.retrieval import Evidence, PageContext, focused_query, page_window


class ContextTests(unittest.TestCase):
    def test_focus_retains_geography_year_and_measurement_basis(self):
        query = focused_query("Compare Goldman Sachs January outlook and IMF October outlook forecasts "
                              "for annual global growth in 2025. Account for publication timing.")
        self.assertIn("annual global growth", query)
        self.assertIn("2025", query)
        self.assertNotIn("publication", query)
        self.assertNotIn("goldman", query)
        self.assertIn("not", focused_query("Compare IMF and Goldman Sachs: which growth is not annual in 2026?"))

    def test_window_preserves_anchor_and_obeys_size_cap(self):
        anchor = "important figure " * 60
        page = "before " * 1000 + anchor + " after" * 1000
        window = page_window(page, anchor)
        self.assertIn(anchor, window)
        self.assertLessEqual(len(window), 2200)
        self.assertIsNone(page_window(page, "not on the page"))

    def test_context_does_not_cross_pages_or_mutate_chunk(self):
        context = PageContext(documents=[{"path": "x.pdf"}])
        reader = Mock()
        reader.pages = [Mock(), Mock()]
        reader.pages[0].extract_text.return_value = "Opening. Anchor sentence. Same page completion."
        reader.pages[1].extract_text.return_value = "OTHER PAGE"
        reader.page_labels = ["1", "2"]
        context.readers["x.pdf"] = reader
        hit = Evidence("id", "Anchor sentence.", "IMF", 2025, "x.pdf", 0, "1", .1)
        expanded = context.expand(hit)
        self.assertIn("Same page completion", expanded.context_text)
        self.assertNotIn("OTHER PAGE", expanded.context_text)
        self.assertEqual(expanded.chunk_id, hit.chunk_id)
        self.assertIsNone(hit.context_text)
        hit.page_label = "2"
        self.assertEqual(context.expand(hit).context_status, "page_label_mismatch")

    def test_wrong_pdf_hash_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x.pdf").write_bytes(b"changed PDF")
            context = PageContext(root, [{"path": "x.pdf", "expected_sha256": hashlib.sha256(b"original").hexdigest()}])
            hit = Evidence("id", "Anchor", "IMF", 2025, "x.pdf", 0, "1", .1)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                context.expand(hit)

    def test_missing_pdf_retains_anchor_and_marks_context_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            context = PageContext(Path(directory), [{"path": "x.pdf", "expected_sha256": "unused"}])
            hit = Evidence("id", "Anchor", "IMF", 2025, "x.pdf", 0, "1", .1)
            expanded = context.expand(hit)
            self.assertEqual(expanded.text, "Anchor")
            self.assertIsNone(expanded.context_text)
            self.assertEqual(expanded.context_status, "local_pdf_missing")
