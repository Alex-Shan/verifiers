"""Translate stateless Responses tools to the renderer's function-call format."""

import json
import uuid

from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.dialects.responses import serialize_response
from verifiers.v1.errors import ProviderError
from verifiers.v1.types import AssistantMessage, Message, Response, Tool, ToolCall


def _tool_name(tool: Tool | ToolCall) -> str:
    return f"{tool.namespace}.{tool.name}" if tool.namespace else tool.name


def _custom_arguments(arguments: str) -> str:
    return json.dumps({"input": arguments}, ensure_ascii=False)


class ResponsesRendererAdapter:
    """Keep native tool identities on the trace and function wrappers in model tokens."""

    def __init__(self, body: dict, tools: list[Tool] | None) -> None:
        if body.get("previous_response_id") or body.get("conversation"):
            raise ProviderError(
                "Responses training requires the full history in input; provider-side "
                "conversation state is unsupported.",
                status_code=400,
            )
        items = body.get("input")
        for item in items if isinstance(items, list) else []:
            kind = item.get("type")
            if (
                item.get("encrypted_content")
                or kind is None
                and not item.get("role")
                or kind == "reasoning"
                and item.get("id")
                and not (item.get("summary") or item.get("content"))
                or kind
                not in (
                    None,
                    "message",
                    "reasoning",
                    "function_call",
                    "custom_tool_call",
                    "function_call_output",
                    "custom_tool_call_output",
                )
            ):
                raise ProviderError(
                    f"Responses training cannot render opaque or unsupported input: {kind}.",
                    status_code=400,
                )
        self._tools: dict[str, Tool] = {}
        wire_tools: list[dict] = []
        for tool in tools or []:
            if tool.type not in ("function", "custom"):
                raise ProviderError(
                    f"Responses training only supports function/custom tools, got {tool.type}.",
                    status_code=400,
                )
            name = _tool_name(tool)
            if name in self._tools:
                raise ProviderError(
                    f"Responses renderer tool name collision: {name}.", status_code=400
                )
            self._tools[name] = tool
            description = tool.description
            parameters = tool.parameters
            if tool.type == "custom":
                description += "\nPass this tool's exact raw input in the input string."
                if format_spec := (tool.model_extra or {}).get("format"):
                    description += "\nInput format: " + json.dumps(format_spec)
                parameters = {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                }
            function = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
            if tool.strict is not None:
                function["strict"] = tool.strict
            wire_tools.append({"type": "function", "function": function})
        self.tools = wire_tools or None

    def message(self, message: Message) -> dict:
        if not isinstance(message, AssistantMessage):
            return message_to_wire(message)
        # Responses provider_state contains output items, not chat reasoning_details.
        wire = message_to_wire(
            message.model_copy(update={"provider_state": None, "tool_calls": None})
        )
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": _tool_name(call),
                        "arguments": _custom_arguments(call.arguments)
                        if call.type == "custom"
                        else call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return wire

    def finish(self, response: Response) -> Response:
        calls = []
        for call in response.message.tool_calls or []:
            if call.name not in self._tools:
                raise ProviderError(f"Unknown Responses renderer tool: {call.name}.")
            tool = self._tools[call.name]
            arguments = call.arguments
            if tool.type == "custom":
                try:
                    wrapped = json.loads(arguments)
                except ValueError as exc:
                    raise ProviderError(
                        "Invalid custom tool input wrapper JSON."
                    ) from exc
                if (
                    not isinstance(wrapped, dict)
                    or set(wrapped) != {"input"}
                    or not isinstance(wrapped["input"], str)
                ):
                    raise ProviderError(
                        "Custom tool arguments must contain an input string."
                    )
                arguments = wrapped["input"]
            calls.append(
                ToolCall(
                    id="call_" + uuid.uuid4().hex,
                    type=tool.type,
                    name=tool.name,
                    namespace=tool.namespace,
                    arguments=arguments,
                )
            )
        response.message = response.message.model_copy(
            update={
                "tool_calls": calls or None,
                "reasoning_content": response.message.reasoning_content or None,
            }
        )
        response.raw = serialize_response(response)
        response.message.provider_state = response.raw["output"]
        return response
