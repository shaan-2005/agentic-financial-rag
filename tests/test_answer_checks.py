import unittest
from unittest.mock import Mock

from finance_rag.answer_checks import check_answer, is_abstention, verify_saved_evidence


def evidence():
    return {"chunk_id": "a", "text": "Global growth is 3.2 percent in 2025.", "publisher": "IMF",
            "year": 2025, "source": "imf.pdf", "page_index": 28, "page_label": "11"}


class AnswerCheckTests(unittest.TestCase):
    def test_valid_citation_and_percent_synonym(self):
        result = check_answer("Growth is 3.20% (IMF, 2025, page 11) [chunk:a]", [evidence()])
        self.assertTrue(result["automated_checks_passed"])
        self.assertTrue(result["semantic_review_required"])

    def test_wrong_publisher_year_and_page_are_not_accepted(self):
        result = check_answer("3.2% (Goldman Sachs, 2024, page 12) [chunk:a]", [evidence()])
        self.assertEqual({e["code"] for e in result["errors"]},
                         {"publisher_mismatch", "year_mismatch", "page_mismatch"})

    def test_invented_id_and_unpaired_reference_are_rejected(self):
        result = check_answer("Claim (IMF, 2025, page 11) [chunk:invented] and [chunk:a]", [evidence()])
        self.assertEqual({e["code"] for e in result["errors"]},
                         {"unknown_chunk_id", "unpaired_chunk_reference"})

    def test_unsupported_percentage_is_flagged(self):
        result = check_answer("Growth is 8.2% (IMF, 2025, page 11) [chunk:a]", [evidence()])
        self.assertTrue(result["citation_metadata_passed"])
        self.assertFalse(result["automated_checks_passed"])
        self.assertEqual(result["warnings"][0]["values"], ["8.2"])

    def test_context_percentage_is_available_but_does_not_prove_meaning(self):
        item = evidence()
        item["context_text"] = item["text"] + " Advanced economies grow 1.6 percent in 2026."
        result = check_answer("Global growth is 1.6% (IMF, 2025, page 11) [chunk:a]", [item])
        # This intentionally passes presence checks despite wrong geography/basis.
        # A structural validator must not pretend it establishes semantic truth.
        self.assertTrue(result["automated_checks_passed"])
        self.assertTrue(result["semantic_review_required"])

    def test_abstention_cannot_hide_an_added_claim(self):
        self.assertTrue(is_abstention("Not found in retrieved documents."))
        self.assertFalse(is_abstention("Not found in retrieved documents. Growth is 9%."))
        self.assertFalse(check_answer("Growth is 3.2%", [evidence()])["citation_metadata_passed"])

    def test_changed_saved_source_text_is_rejected(self):
        item = evidence()
        collection = Mock()
        collection.get.return_value = {"ids": ["a"], "documents": [item["text"]],
            "metadatas": [{"publisher": "IMF", "year": 2025, "source": "imf.pdf", "page": 28, "page_label": "11"}]}
        verify_saved_evidence([item], collection)
        item["text"] = "Invented evidence"
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_saved_evidence([item], collection)

    def test_empty_evidence_graph_can_return_only_an_honest_abstention(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from finance_rag.agent import build_agent
        model = Mock()
        model.bind_tools.return_value = model
        tool_call = AIMessage(content="", tool_calls=[
            {"name": "retriever_tool", "args": {"query": "unanswerable"}, "id": "t1", "type": "tool_call"}])
        model.invoke.side_effect = [tool_call, AIMessage(content="Not found in retrieved documents.")]
        retriever = Mock()
        retriever.search.return_value = []
        result = build_agent(retriever, llm=model).invoke(
            {"messages": [HumanMessage(content="Unanswerable question")], "tool_rounds": 0})
        self.assertTrue(is_abstention(result["messages"][-1].content))
        self.assertEqual(retriever.search.call_count, 1)
        model.invoke.side_effect = [tool_call, AIMessage(content="Not found in retrieved documents. Growth is 9%.")]
        with self.assertRaisesRegex(RuntimeError, "without retrieving"):
            build_agent(retriever, llm=model).invoke(
                {"messages": [HumanMessage(content="Unanswerable question")], "tool_rounds": 0})
        model.invoke.side_effect = [AIMessage(content="Not found in retrieved documents.")]
        with self.assertRaisesRegex(RuntimeError, "without retrieving"):
            build_agent(retriever, llm=model).invoke(
                {"messages": [HumanMessage(content="Answerable question")], "tool_rounds": 0})
