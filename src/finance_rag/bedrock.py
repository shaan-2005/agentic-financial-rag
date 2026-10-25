"""Adapt Bedrock Converse to the existing LangGraph model/tool interface."""
import json
import re
import time

from .settings import read_json


def clean_answer_text(text):
    """Do not display Nova's sometimes-emitted internal-looking thinking wrapper."""
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    if "<thinking>" in cleaned or "</thinking>" in cleaned:
        raise RuntimeError("Model returned an incomplete thinking wrapper; no final answer was accepted.")
    return cleaned


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(not isinstance(block, dict) or block.get("type") != "text" for block in content):
            raise ValueError("This project supports text messages only")
        return "\n".join(block.get("text", "") for block in content)
    raise ValueError("Unsupported message content")


def converse_messages(messages):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    system = []
    conversation = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system.append({"text": message_text(message.content)})
            continue
        if isinstance(message, ToolMessage):
            role = "user"
            content = [{"toolResult": {"toolUseId": message.tool_call_id,
                                       "content": [{"text": message_text(message.content)}]}}]
        elif isinstance(message, HumanMessage):
            role = "user"
            content = [{"text": message_text(message.content)}]
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = []
            text = message_text(message.content)
            if text:
                content.append({"text": text})
            content.extend({"toolUse": {"toolUseId": call["id"], "name": call["name"], "input": call["args"]}}
                           for call in message.tool_calls)
            if not content:
                raise ValueError("Assistant message has no text or tool calls")
        else:
            raise ValueError("Unsupported message type")
        # Bedrock expects all results from one assistant tool request in one user turn.
        if conversation and conversation[-1]["role"] == role:
            conversation[-1]["content"].extend(content)
        else:
            conversation.append({"role": role, "content": content})
    return system, conversation


class BedrockChat:
    def __init__(self, client=None, config=None):
        self.config = config or read_json("bedrock.json")
        self.tools = []
        self.calls = []
        if client is None:
            from botocore.config import Config
            from .aws_access import verified_session
            session, _ = verified_session()
            # No automatic inference retries: a lost response could otherwise be billed twice.
            client = session.client("bedrock-runtime", region_name=self.config["region"],
                                    config=Config(connect_timeout=10, read_timeout=45,
                                                  retries={"mode": "standard", "total_max_attempts": 1}))
        self.client = client

    def bind_tools(self, tools):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        self.tools = []
        for tool in tools:
            spec = convert_to_openai_tool(tool)["function"]
            self.tools.append({"toolSpec": {"name": spec["name"], "description": spec.get("description", ""),
                                             "inputSchema": {"json": spec["parameters"]}}})
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        from botocore.exceptions import BotoCoreError, ClientError

        if len(self.calls) >= self.config["max_model_calls"]:
            raise RuntimeError("Bedrock call limit reached for this answer.")
        system, conversation = converse_messages(messages)
        request = {"modelId": self.config["model_id"], "messages": conversation,
                   "inferenceConfig": {"maxTokens": self.config["max_output_tokens"], "temperature": 0}}
        if system:
            request["system"] = system
        if self.tools:
            request["toolConfig"] = {"tools": self.tools}
        request_bytes = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        if request_bytes > self.config["max_request_bytes"]:
            raise RuntimeError("Bedrock request size limit reached; reduce retrieved context or tool rounds.")
        started = time.perf_counter()
        # Count attempted calls, including errors; do not retry indefinitely.
        record = {"request_bytes": request_bytes, "status": "attempted"}
        self.calls.append(record)
        try:
            response = self.client.converse(**request)
        except ClientError as error:
            record["status"] = "failed"
            code = error.response.get("Error", {}).get("Code", "Unknown")
            if code == "AccessDeniedException":
                raise RuntimeError("Bedrock access denied. Check FinanceRagBedrockAccess; account/model eligibility may also restrict access. Do not upgrade the Free plan to resolve this automatically.") from error
            if code in {"ThrottlingException", "ServiceQuotaExceededException"}:
                raise RuntimeError("Bedrock quota reached. Stop and check the account's model quota before retrying.") from error
            raise RuntimeError(f"Bedrock returned {code}; no automatic inference retry was made.") from error
        except BotoCoreError as error:
            record["status"] = "failed"
            raise RuntimeError("Bedrock network or credential error. No automatic inference retry was made.") from error
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        cost = (input_tokens * self.config["input_usd_per_million_tokens"]
                + output_tokens * self.config["output_usd_per_million_tokens"]) / 1_000_000
        record.update({"status": "succeeded", "input_tokens": input_tokens, "output_tokens": output_tokens,
                       "estimated_usd": round(cost, 8), "latency_seconds": round(time.perf_counter() - started, 3),
                       "stop_reason": response.get("stopReason", "unknown")})
        if response.get("stopReason") in {"max_tokens", "guardrail_intervened", "content_filtered"}:
            raise RuntimeError("Bedrock stopped before a complete answer; token usage was recorded.")
        blocks = response["output"]["message"]["content"]
        text = clean_answer_text("\n".join(block["text"] for block in blocks if "text" in block))
        calls = [{"name": block["toolUse"]["name"], "args": block["toolUse"]["input"],
                  "id": block["toolUse"]["toolUseId"], "type": "tool_call"}
                 for block in blocks if "toolUse" in block]
        if not text and not calls:
            raise RuntimeError("Model returned no final text or tool requests.")
        return AIMessage(content=text, tool_calls=calls,
                         usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                         "total_tokens": input_tokens + output_tokens})

    def usage_report(self):
        return {"provider": "bedrock", "model_id": self.config["model_id"], "region": self.config["region"],
                "model_calls": len(self.calls), "input_tokens": sum(c.get("input_tokens", 0) for c in self.calls),
                "output_tokens": sum(c.get("output_tokens", 0) for c in self.calls),
                "estimated_usd": round(sum(c.get("estimated_usd", 0) for c in self.calls), 8),
                "unreported_usage_possible": any(c["status"] in {"attempted", "failed"} for c in self.calls),
                "calls": self.calls, "pricing_note": self.config["pricing_note"]}


def smoke_test():
    from langchain_core.messages import HumanMessage
    config = dict(read_json("bedrock.json"), max_output_tokens=64, max_model_calls=1)
    model = BedrockChat(config=config)
    answer = model.invoke([HumanMessage(content="Reply with exactly: Bedrock connection works.")])
    return {"answer": answer.content, "usage": model.usage_report()}
