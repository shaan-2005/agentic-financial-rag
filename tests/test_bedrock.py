import unittest
from unittest.mock import Mock

from finance_rag.bedrock import BedrockChat, clean_answer_text, converse_messages
from finance_rag.settings import read_json


def response(text="Done", tools=None, stop="end_turn"):
    content = [{"text": text}] if text else []
    content.extend({"toolUse": t} for t in (tools or []))
    return {"output": {"message": {"role": "assistant", "content": content}}, "stopReason": stop,
            "usage": {"inputTokens": 100, "outputTokens": 20}}


class BedrockAdapterTests(unittest.TestCase):
    def test_thinking_wrapper_is_removed_but_final_answer_is_retained(self):
        self.assertEqual(clean_answer_text("<thinking>Draft explanation</thinking>\nFinal answer"), "Final answer")
        with self.assertRaisesRegex(RuntimeError, "incomplete thinking"):
            clean_answer_text("<thinking>Incomplete response")

    def test_tool_results_from_one_turn_are_grouped_and_keep_request_ids(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        system, messages = converse_messages([
            SystemMessage(content="Use evidence"), HumanMessage(content="Compare reports"),
            AIMessage(content="", tool_calls=[
                {"name": "retriever_tool", "args": {"query": "IMF"}, "id": "a", "type": "tool_call"},
                {"name": "retriever_tool", "args": {"query": "Goldman"}, "id": "b", "type": "tool_call"}]),
            ToolMessage(content="IMF evidence", tool_call_id="a"),
            ToolMessage(content="Goldman evidence", tool_call_id="b")])
        self.assertEqual(system, [{"text": "Use evidence"}])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual([c["toolResult"]["toolUseId"] for c in messages[-1]["content"]], ["a", "b"])

    def test_bedrock_tool_request_executes_existing_graph_and_records_usage(self):
        from langchain_core.messages import HumanMessage
        from finance_rag.agent import build_agent
        from finance_rag.retrieval import Evidence
        client = Mock()
        client.converse.side_effect = [response(text="", tools=[
            {"toolUseId": "call1", "name": "retriever_tool", "input": {"query": "growth"}}], stop="tool_use"),
            response("Growth is 3.1% (Goldman Sachs, 2025, page 55) [chunk:example]")]
        model = BedrockChat(client=client)
        retriever = Mock()
        retriever.search.return_value = [Evidence("example", "Growth is 3.1%", "Goldman Sachs", 2025,
                                                   "gs_outlook.pdf", 57, "55", 0.1)]
        evidence = {}
        result = build_agent(retriever, llm=model, evidence_store=evidence).invoke(
            {"messages": [HumanMessage(content="growth?")], "tool_rounds": 0})
        self.assertIn("3.1%", result["messages"][-1].content)
        self.assertIn("example", evidence)
        self.assertEqual(model.usage_report()["model_calls"], 2)
        self.assertEqual(model.usage_report()["input_tokens"], 200)
        self.assertAlmostEqual(model.usage_report()["estimated_usd"], 0.0000126)
        sent = client.converse.call_args_list[1].kwargs
        self.assertEqual(sent["messages"][-1]["content"][0]["toolResult"]["toolUseId"], "call1")
        self.assertEqual(sent["modelId"], "amazon.nova-micro-v1:0")

    def test_no_evidence_answer_is_rejected(self):
        from langchain_core.messages import HumanMessage
        from finance_rag.agent import build_agent
        client = Mock()
        client.converse.return_value = response("Invented growth forecast")
        with self.assertRaisesRegex(RuntimeError, "without retrieving"):
            build_agent(Mock(), llm=BedrockChat(client=client)).invoke(
                {"messages": [HumanMessage(content="growth?")], "tool_rounds": 0})

    def test_request_size_limit_prevents_inference(self):
        from langchain_core.messages import HumanMessage
        client = Mock()
        model = BedrockChat(client=client, config=dict(read_json("bedrock.json"), max_request_bytes=100))
        with self.assertRaisesRegex(RuntimeError, "size limit"):
            model.invoke([HumanMessage(content="x" * 200)])
        client.converse.assert_not_called()

    def test_model_call_limit_prevents_extra_inference(self):
        from langchain_core.messages import HumanMessage
        client = Mock()
        client.converse.return_value = response()
        model = BedrockChat(client=client, config=dict(read_json("bedrock.json"), max_model_calls=1))
        model.invoke([HumanMessage(content="hello")])
        with self.assertRaisesRegex(RuntimeError, "call limit"):
            model.invoke([HumanMessage(content="hello")])
        self.assertEqual(client.converse.call_count, 1)

    def test_access_denied_does_not_trigger_an_inference_retry(self):
        from botocore.exceptions import ClientError
        from langchain_core.messages import HumanMessage
        client = Mock()
        client.converse.side_effect = ClientError({"Error": {"Code": "AccessDeniedException"}}, "Converse")
        model = BedrockChat(client=client)
        with self.assertRaisesRegex(RuntimeError, "access denied"):
            model.invoke([HumanMessage(content="hello")])
        self.assertEqual(client.converse.call_count, 1)
        self.assertEqual(model.usage_report()["calls"][0]["status"], "failed")

    def test_truncated_response_is_not_reported_as_a_complete_answer(self):
        from langchain_core.messages import HumanMessage
        client = Mock()
        client.converse.return_value = response("Incomplete", stop="max_tokens")
        model = BedrockChat(client=client)
        with self.assertRaisesRegex(RuntimeError, "before a complete answer"):
            model.invoke([HumanMessage(content="hello")])
        self.assertEqual(model.usage_report()["output_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
