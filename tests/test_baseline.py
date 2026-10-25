import unittest
from unittest.mock import Mock

from finance_rag.ingestion import make_chunk_id


class IngestionIdentityTests(unittest.TestCase):
    def test_same_passage_is_repeatable_and_content_changes_are_distinct(self):
        args = ("doc", "filehash", "pipelinehash", 4, 0, "Growth is 3.1%")
        self.assertEqual(make_chunk_id(*args), make_chunk_id(*args))
        self.assertNotEqual(make_chunk_id(*args), make_chunk_id(*args[:-1], "Growth is 3.2%"))
        self.assertNotEqual(make_chunk_id(*args), make_chunk_id("doc", "filehash", "newpipeline", 4, 0, args[-1]))


class AgentBoundaryTests(unittest.TestCase):
    def test_original_graph_passes_tool_evidence_back_to_model(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from finance_rag.agent import build_agent
        from finance_rag.retrieval import Evidence

        retriever = Mock()
        retriever.search.return_value = [Evidence("id1", "Growth is 3.1%", "Goldman Sachs", 2025,
                                                   "gs_outlook.pdf", 57, "55", 0.1)]
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [AIMessage(content="", tool_calls=[
            {"name": "retriever_tool", "args": {"query": "growth"}, "id": "call1", "type": "tool_call"}]),
            AIMessage(content="Growth is 3.1% (Goldman Sachs, 2025, page 55) [chunk:id1]")]
        result = build_agent(retriever, publisher="Goldman Sachs", llm=model).invoke(
            {"messages": [HumanMessage(content="growth?")], "tool_rounds": 0})
        retriever.search.assert_called_once_with("growth", publisher="Goldman Sachs")
        self.assertEqual(result["tool_rounds"], 1)
        sent = model.invoke.call_args_list[1].args[0]
        tool_results = [m for m in sent if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_results), 1)
        self.assertIn('"chunk_id": "id1"', tool_results[0].content)

    def test_repeated_tool_requests_stop_at_application_limit(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from finance_rag.agent import build_agent

        retriever = Mock()
        retriever.search.return_value = []
        model = Mock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = lambda messages: AIMessage(content="", tool_calls=[
            {"name": "retriever_tool", "args": {"query": "growth"}, "id": str(len(messages)), "type": "tool_call"}])
        with self.assertRaisesRegex(RuntimeError, "Tool-round limit"):
            build_agent(retriever, llm=model).invoke(
                {"messages": [HumanMessage(content="growth?")], "tool_rounds": 0},
                config={"recursion_limit": 12})
        self.assertEqual(retriever.search.call_count, 3)


if __name__ == "__main__":
    unittest.main()
