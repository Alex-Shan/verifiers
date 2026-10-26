"""Responses renderer conversions preserve native tool semantics and sampled tokens."""

import json

import numpy as np
import pytest
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from verifiers.v1.clients.responses import ResponsesRendererAdapter
from verifiers.v1.dialects import ResponsesDialect
from verifiers.v1.dialects.base import parse_sse_event
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import message_hash
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    Tool,
    ToolCall,
    TurnTokens,
    Usage,
)


def _response(content=None, reasoning="Think first.", calls=None, finish="tool_calls"):
    return Response(
        id="resp_test",
        created=0,
        model="qwen3",
        message=AssistantMessage(
            content=content, reasoning_content=reasoning, tool_calls=calls
        ),
        finish_reason=finish,
        usage=Usage(prompt_tokens=3, completion_tokens=2),
        tokens=TurnTokens(
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5],
            completion_logprobs=[-0.1, -0.2],
        ),
    )


def test_namespaced_and_custom_tools_round_trip_without_changing_tokens():
    tools = [
        Tool(name="run", namespace="functions", parameters={"type": "object"}),
        Tool(name="run", namespace="other", parameters={"type": "object"}),
        Tool(type="custom", name="apply_patch", format={"type": "text"}),
    ]
    adapter = ResponsesRendererAdapter({}, tools)
    assert [t["function"]["name"] for t in adapter.tools] == [
        "functions.run",
        "other.run",
        "apply_patch",
    ]
    assert adapter.tools[2]["function"]["parameters"]["required"] == ["input"]
    patch = "*** Begin Patch\n*** Add File: x\n+你好\n*** End Patch"
    response = _response(
        content="I will edit the file.",
        calls=[
            ToolCall(id="0", name="functions.run", arguments='{"cmd":"pwd"}'),
            ToolCall(
                id="1", name="apply_patch", arguments=json.dumps({"input": patch})
            ),
        ],
    )
    tokens = response.tokens
    adapter.finish(response)
    function, custom = response.message.tool_calls
    assert (function.name, function.namespace, function.type) == (
        "run",
        "functions",
        "function",
    )
    assert (custom.type, custom.arguments) == ("custom", patch)
    assert function.id != custom.id
    assert response.tokens is tokens
    np.testing.assert_allclose(response.tokens.completion_logprobs, [-0.1, -0.2])
    replay = ResponsesDialect().parse_request({"input": response.raw["output"]})
    assert message_hash(replay.messages[0]) == message_hash(response.message)
    wire = adapter.message(replay.messages[0])
    assert wire["reasoning_content"] == "Think first."
    assert "reasoning_details" not in wire
    assert wire["tool_calls"][0]["function"]["name"] == "functions.run"
    assert json.loads(wire["tool_calls"][1]["function"]["arguments"]) == {
        "input": patch
    }
    assert replay.messages[0].tool_calls[1].type == "custom"


@pytest.mark.parametrize("finish", ["stop", "tool_calls", "length"])
def test_responses_stream_validates_and_reassembles_all_output_items(finish):
    adapter = ResponsesRendererAdapter(
        {}, [Tool(name="run"), Tool(type="custom", name="patch", namespace="edits")]
    )
    response = adapter.finish(
        _response(
            content="Working.",
            finish=finish,
            calls=[
                ToolCall(id="0", name="run", arguments="{}"),
                ToolCall(
                    id="1", name="edits.patch", arguments='{"input":"patch text"}'
                ),
            ]
            if finish != "stop"
            else None,
        )
    )
    OpenAIResponse.model_validate(response.raw)
    dialect = ResponsesDialect()
    parser = dialect.stream_parser()
    events = []
    for chunk in dialect.stream_events(response.raw):
        event = parse_sse_event(chunk)
        if not event:
            continue
        TypeAdapter(ResponseStreamEvent).validate_python(event)
        events.append(event)
        parser.feed(chunk)
        if dialect.is_terminal_event(chunk):
            parser.on_done()
    replayed = parser.finish()
    assert message_hash(replayed.message) == message_hash(response.message)
    assert replayed.finish_reason == finish
    assert replayed.usage.total_tokens == 5
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert [
        e["item"] for e in events if e["type"] == "response.output_item.done"
    ] == response.raw["output"]
    assert events[-1]["type"] == (
        "response.incomplete" if finish == "length" else "response.completed"
    )
    if finish != "stop":
        assert any(
            e["type"] == "response.function_call_arguments.delta" for e in events
        )
        assert any(e["type"] == "response.custom_tool_call_input.done" for e in events)


@pytest.mark.parametrize(
    "content,reasoning", [("Done.", None), (None, None), (None, "Thinking.")]
)
def test_text_and_reasoning_only_responses(content, reasoning):
    response = ResponsesRendererAdapter({}, None).finish(
        _response(content=content, reasoning=reasoning, finish="stop")
    )
    dialect = ResponsesDialect()
    parsed = dialect.parse_response(dialect.validate_response(response.raw))
    assert message_hash(parsed.message) == message_hash(response.message)
    assert dialect.stream_events(response.raw)


@pytest.mark.parametrize(
    "arguments", ["not json", "[]", '{"input":3}', '{"input":"x","extra":1}']
)
def test_malformed_custom_input_is_a_provider_error(arguments):
    adapter = ResponsesRendererAdapter({}, [Tool(type="custom", name="patch")])
    with pytest.raises(ProviderError):
        adapter.finish(
            _response(calls=[ToolCall(id="0", name="patch", arguments=arguments)])
        )


@pytest.mark.parametrize(
    "body",
    [
        {"previous_response_id": "resp_prior"},
        {"conversation": "conv_prior"},
        {"input": [{"type": "reasoning", "encrypted_content": "opaque"}]},
        {"input": [{"type": "item_reference", "id": "prior"}]},
        {"input": [{"id": "prior"}]},
        {"input": [{"type": "reasoning", "id": "rs_prior"}]},
        {"input": [{"type": "compaction", "encrypted_content": "opaque"}]},
    ],
)
def test_provider_owned_state_is_rejected_before_generation(body):
    with pytest.raises(ProviderError) as error:
        ResponsesRendererAdapter(body, None)
    assert error.value.status_code == 400


def test_renderer_tool_name_collisions_are_rejected():
    with pytest.raises(ProviderError, match="collision"):
        ResponsesRendererAdapter({}, [Tool(name="a.b"), Tool(namespace="a", name="b")])


def test_provider_executed_tools_are_rejected():
    with pytest.raises(ProviderError, match="function/custom"):
        ResponsesRendererAdapter({}, [Tool(type="web_search", name="web_search")])
