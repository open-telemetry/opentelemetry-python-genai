# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent (``invoke_agent``) instrumentation tests."""

from __future__ import annotations

import inspect
import json
import pathlib
import tempfile
from collections.abc import Generator
from types import GeneratorType, SimpleNamespace
from typing import Any

import pytest
from smolagents import CodeAgent, OpenAIModel, ToolCallingAgent
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)
from smolagents.monitoring import TokenUsage

from opentelemetry.instrumentation.genai.smolagents import (
    patch as patch_module,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .test_utils import (
    BrokenTool,
    FakeCodeModel,
    FakeStreamingCodeModel,
    ImageTool,
    NeverFinishingCodeModel,
    attr,
    data_point_attributes,
    metrics_by_name,
    parse_messages,
    spans_by_operation,
)


def _operations(spans: list[Any]) -> list[str]:
    return [
        (span.attributes or {}).get(GenAI.GEN_AI_OPERATION_NAME)
        for span in spans
    ]


def test_tool_calling_agent_with_image(
    instrument_with_content, span_exporter, vcr
) -> None:
    from PIL import Image

    model = OpenAIModel(
        model_id="gpt-4o",
        api_key="test_openai_api_key",
        api_base="https://api.openai.com/v1",
    )
    image_path = pathlib.Path(__file__).parent / "fixtures" / "img.png"
    agent = ToolCallingAgent(tools=[], model=model, max_steps=3)
    with vcr.use_cassette("agent_with_image.yaml"):
        agent.run(
            "Describe what you see in this image briefly.",
            images=[Image.open(image_path)],
        )

    spans = span_exporter.get_finished_spans()
    (agent_span,) = spans_by_operation(spans, "invoke_agent")
    (tool_span,) = spans_by_operation(spans, "execute_tool")

    # Two operations, because the model call OpenAIModel makes is left to the
    # OpenAI SDK's own instrumentation.
    assert sorted(filter(None, _operations(spans))) == [
        "execute_tool",
        "invoke_agent",
    ]
    assert agent_span.name == "invoke_agent ToolCallingAgent"
    assert tool_span.parent.span_id == agent_span.context.span_id
    # The run itself consumes no tokens, so it reports none.
    assert attr(agent_span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) is None
    assert attr(agent_span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) is None

    # run(task, images=...) is passed positionally.
    agent_inputs = parse_messages(agent_span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert agent_inputs[0]["role"] == "user"
    assert agent_inputs[0]["parts"][0] == {
        "type": "text",
        "content": "Describe what you see in this image briefly.",
    }
    assert agent_inputs[0]["parts"][1]["type"] == "blob"
    assert agent_inputs[0]["parts"][1]["modality"] == "image"


def test_code_agent_non_streaming(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    result = agent.run("Test question")
    assert result == "Test result from CodeAgent"

    spans = span_exporter.get_finished_spans()
    (agent_span,) = spans_by_operation(spans, "invoke_agent")
    (tool_span,) = spans_by_operation(spans, "execute_tool")

    # Executor context propagation: final_answer nests under the agent span.
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert attr(agent_span, GenAI.GEN_AI_REQUEST_MODEL) == "fake-model"
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)
    assert json.loads(attr(agent_span, GenAI.GEN_AI_TOOL_DEFINITIONS)) == [
        {
            "type": "function",
            "name": "final_answer",
            "description": "Provides a final answer to the given problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the problem",
                    }
                },
                "required": ["answer"],
            },
        }
    ]
    # run(task) is positional, so the task is only recorded if the wrapper
    # binds positional arguments to the signature.
    inputs = parse_messages(agent_span, GenAI.GEN_AI_INPUT_MESSAGES)
    assert inputs[0] == {
        "role": "user",
        "parts": [{"type": "text", "content": "Test question"}],
    }
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"][0]["content"] == "Test result from CodeAgent"


def test_code_agent_no_content(instrument_no_content, span_exporter) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)
    assert attr(agent_span, GenAI.GEN_AI_TOOL_DEFINITIONS) is not None
    assert attr(agent_span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None


def test_additional_args_reach_the_recorded_task(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question", additional_args={"city": "Paris"})

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    (part,) = parse_messages(agent_span, GenAI.GEN_AI_INPUT_MESSAGES)[0][
        "parts"
    ]
    assert part["content"].startswith("Test question")
    assert "city" in part["content"]
    assert "Paris" in part["content"]


def test_run_returning_full_result_records_the_final_answer(
    instrument_with_content, span_exporter
) -> None:
    # run(return_full_result=True) returns a RunResult wrapping the answer.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    result = agent.run("Test question", return_full_result=True)
    assert result.output == "Test result from CodeAgent"

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"] == [
        {"type": "text", "content": "Test result from CodeAgent"}
    ]


IMAGE_FINAL_ANSWER = """
Thought: Return the image.
Code:
```py
final_answer(make_image())
```<end_code>
"""


class _ImageAnswerModel(FakeCodeModel):
    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=IMAGE_FINAL_ANSWER,
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def test_image_final_answer_is_recorded_as_a_blob(
    instrument_with_content, span_exporter, monkeypatch
) -> None:
    # smolagents wraps an image answer in AgentImage, whose __str__ saves a PNG
    # into a new temp directory and mutates the object the caller still holds.
    def _no_temp_dirs(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("telemetry wrote an image to disk")

    monkeypatch.setattr(tempfile, "mkdtemp", _no_temp_dirs)

    agent = CodeAgent(
        tools=[ImageTool()], model=_ImageAnswerModel(), max_steps=2
    )
    agent.run("Make an image")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    (part,) = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)[0][
        "parts"
    ]
    assert part["type"] == "blob"
    assert part["modality"] == "image"
    assert part["mime_type"] == "image/png"


def test_code_agent_streaming_is_lazy_until_drained(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)

    # Nothing has finished before the caller drains the stream.
    assert (
        spans_by_operation(span_exporter.get_finished_spans(), "invoke_agent")
        == []
    )

    list(stream)

    spans = span_exporter.get_finished_spans()
    (agent_span,) = spans_by_operation(spans, "invoke_agent")
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"][0]["content"] == "Test result from CodeAgent"
    # The agent span is still open while the caller drains the stream, so the
    # tool call the run makes on the way stays nested under it instead of
    # becoming a root span.
    (tool_span,) = spans_by_operation(spans, "execute_tool")
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id


def test_streaming_run_stays_a_generator(
    instrument_with_content, span_exporter
) -> None:
    # Instrumentation observes, it doesn't change what run() returns. Callers
    # branch on these checks, so a wrapper that fails them changes behavior.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)

    assert isinstance(stream, Generator)
    assert inspect.isgenerator(stream)
    assert stream.__class__ is GeneratorType
    list(stream)


def test_streaming_run_records_no_chunk_metrics(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    # The chunk metrics describe the response stream of a generation call,
    # and a chunk of an agent run is a step object.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    list(agent.run("Test question", stream=True))

    metrics = metrics_by_name(metric_reader)
    assert gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION in metrics
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK
        not in metrics
    )
    assert (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_TIME_PER_OUTPUT_CHUNK
        not in metrics
    )


def test_streaming_run_with_stream_outputs(
    instrument_with_content, span_exporter
) -> None:
    # stream_outputs=True routes the agent's model calls through
    # generate_stream. The run reports the same thing either way.
    agent = CodeAgent(
        tools=[],
        model=FakeStreamingCodeModel(),
        max_steps=3,
        stream_outputs=True,
    )
    list(agent.run("Test question", stream=True))

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"][0]["content"] == "Test result from CodeAgent"


def test_agent_run_metrics(
    instrument_with_content, span_exporter, metric_reader
) -> None:
    # gen_ai.provider.name is required on gen_ai.client.operation.duration and
    # AgentInvocation carries none by default, so the wrapper resolves it from
    # the agent's model.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    metrics = metrics_by_name(metric_reader)
    duration = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION]
    assert {
        GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
        GenAI.GEN_AI_PROVIDER_NAME: "unknown",
        GenAI.GEN_AI_REQUEST_MODEL: "fake-model",
    } in data_point_attributes(duration)
    # Only model calls feed the token histogram, and this package
    # instruments none.
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE not in metrics


@pytest.mark.parametrize("stream", [False, True])
def test_run_that_exhausts_max_steps_is_not_reported_as_a_plain_stop(
    instrument_with_content, span_exporter, stream: bool
) -> None:
    # smolagents doesn't raise when a run runs out of steps: it synthesizes an
    # answer, appends an ActionStep carrying AgentMaxStepsError and returns.
    # Reporting "stop" would make giving up indistinguishable from answering.
    agent = CodeAgent(tools=[], model=NeverFinishingCodeModel(), max_steps=1)
    if stream:
        list(agent.run("Test question", stream=True))
    else:
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == (
        "length",
    )
    # The library returned normally, so the run is not an error.
    assert agent_span.status.status_code == StatusCode.UNSET
    assert attr(agent_span, error_attributes.ERROR_TYPE) is None
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["finish_reason"] == "length"


def test_streaming_run_close_finalizes_once(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)

    # Close before draining: no FinalAnswerStep observed, so no output recorded.
    stream.close()
    stream.close()  # idempotent

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert len(agent_spans) == 1
    assert attr(agent_spans[0], GenAI.GEN_AI_OUTPUT_MESSAGES) is None


@pytest.mark.parametrize("use_context_manager", [True, False])
def test_streaming_run_stopped_midway_finalizes_successfully(
    instrument_with_content, span_exporter, use_context_manager: bool
) -> None:
    # smolagents' _run_stream yields from a finally block, so closing it after
    # the first step makes CPython raise "generator ignored GeneratorExit".
    # Abandoning a run is not a failed run and must not surface an error the
    # caller never wrote.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)
    if use_context_manager:
        with stream:
            for _ in stream:
                break
    else:
        for _ in stream:
            break
        stream.close()

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert attr(agent_span, error_attributes.ERROR_TYPE) is None
    # No FinalAnswerStep was observed, so the run reports no outcome.
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


def test_streaming_run_close_reraises_a_real_cleanup_error(
    instrument_with_content, span_exporter, monkeypatch
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _failing_cleanup(*args: Any, **kwargs: Any):
        try:
            yield SimpleNamespace(token_usage=None)
        finally:
            raise RuntimeError("cleanup exploded")

    monkeypatch.setattr(agent, "_run_stream", _failing_cleanup)
    stream = agent.run("Test question", stream=True)
    for _ in stream:
        break
    with pytest.raises(RuntimeError, match="cleanup exploded"):
        stream.close()

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.ERROR
    assert attr(agent_span, error_attributes.ERROR_TYPE) == "RuntimeError"


def test_caller_error_inside_stream_context(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    with pytest.raises(RuntimeError, match="caller exploded"):
        with agent.run("Test question", stream=True):
            raise RuntimeError("caller exploded")

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    assert attr(agent_spans[0], error_attributes.ERROR_TYPE) == "RuntimeError"


def test_run_failure_before_stream(
    instrument_with_content, span_exporter, monkeypatch
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "_run_stream", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.ERROR
    assert attr(agent_span, error_attributes.ERROR_TYPE) == "RuntimeError"


def test_stream_failure_during_iteration(
    instrument_with_content, span_exporter, monkeypatch
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _bad_stream(*args: Any, **kwargs: Any):
        yield SimpleNamespace(token_usage=None)
        raise ConnectionError("stream died")

    monkeypatch.setattr(agent, "_run_stream", _bad_stream)
    stream = agent.run("Test question", stream=True)
    with pytest.raises(ConnectionError, match="stream died"):
        list(stream)

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    assert (
        attr(agent_spans[0], error_attributes.ERROR_TYPE) == "ConnectionError"
    )


def test_interrupted_run_ends_the_span(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
) -> None:
    # A KeyboardInterrupt is not an Exception. Catching only Exception would
    # leave the span of an interrupted run open for the process's lifetime.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _interrupt(*args: Any, **kwargs: Any):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent, "_run_stream", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    # Interrupting a run is not the agent failing, which is how the util's own
    # ``with invocation`` reads it too.
    assert agent_span.status.status_code == StatusCode.UNSET
    assert lifecycle.leaked == []


def test_a_failed_recording_does_not_break_a_finished_run(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
) -> None:
    # The run is over by the time the answer is recorded, so a conversion that
    # cannot read it drops the content and changes nothing else.
    def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unexpected answer shape")

    monkeypatch.setattr(patch_module, "final_answer_parts", raise_error)

    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    assert agent.run("Test question") == "Test result from CodeAgent"

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert attr(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    assert lifecycle.leaked == []


class _ManagerModel:
    """Manager CodeAgent model: first delegate to the managed agent, then finish."""

    model_id = "manager-model"
    kwargs: dict[str, Any] = {}

    def generate(self, messages, **kwargs) -> ChatMessage:
        if len(messages) < 4:
            content = (
                "Thought: delegate.\nCode:\n```py\n"
                'search_agent("Who is the president?")\n```<end_code>'
            )
        else:
            content = (
                "Thought: finish.\nCode:\n```py\n"
                'final_answer("Final report.")\n```<end_code>'
            )
        return ChatMessage(
            role="assistant",
            content=content,
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        )

    def __call__(self, *args, **kwargs) -> ChatMessage:
        return self.generate(*args, **kwargs)


class _ManagedModel:
    """Managed ToolCallingAgent model: immediately return final_answer."""

    model_id = "managed-model"
    kwargs: dict[str, Any] = {}

    def generate(self, messages, **kwargs) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments="Report on the president",
                    ),
                )
            ],
            token_usage=TokenUsage(input_tokens=4, output_tokens=2),
        )

    def __call__(self, *args, **kwargs) -> ChatMessage:
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        return message


def test_managed_agent_nesting(instrument_with_content, span_exporter) -> None:
    managed = ToolCallingAgent(
        tools=[],
        model=_ManagedModel(),
        max_steps=3,
        name="search_agent",
        description="Runs searches.",
    )
    manager = CodeAgent(
        tools=[],
        model=_ManagerModel(),
        managed_agents=[managed],
        max_steps=4,
    )
    assert manager.run("Fake question.") == "Final report."

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    by_name = {span.name: span for span in agent_spans}
    assert "invoke_agent search_agent" in by_name
    manager_span = next(
        span for name, span in by_name.items() if "search_agent" not in name
    )
    managed_span = by_name["invoke_agent search_agent"]
    assert managed_span.parent.span_id == manager_span.context.span_id
    assert (
        attr(managed_span, GenAI.GEN_AI_AGENT_DESCRIPTION) == "Runs searches."
    )
    assert attr(manager_span, GenAI.GEN_AI_AGENT_DESCRIPTION) is None
    # A managed agent is something the manager's model can call, so it belongs
    # in the manager's tool definitions next to its own tools.
    definitions = json.loads(attr(manager_span, GenAI.GEN_AI_TOOL_DEFINITIONS))
    assert [definition["name"] for definition in definitions] == [
        "final_answer",
        "search_agent",
    ]
    search_agent = definitions[1]
    assert search_agent["description"] == "Runs searches."
    assert "task" in search_agent["parameters"]["properties"]


class _RecoveringModel:
    """ToolCallingAgent model: call the broken tool once, then final_answer."""

    model_id = "recovering-model"
    kwargs: dict[str, Any] = {}

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, **kwargs) -> ChatMessage:
        self.calls += 1
        if self.calls == 1:
            name, arguments = "broken_tool", {"location": "Paris"}
        else:
            name, arguments = "final_answer", "recovered"
        return ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id=f"call_{self.calls}",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=name, arguments=arguments
                    ),
                )
            ],
            token_usage=TokenUsage(input_tokens=3, output_tokens=1),
        )

    def __call__(self, *args, **kwargs) -> ChatMessage:
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        return message


def test_expected_tool_error_is_recorded_and_agent_continues(
    instrument_with_content, span_exporter
) -> None:
    agent = ToolCallingAgent(
        tools=[BrokenTool()], model=_RecoveringModel(), max_steps=4
    )
    assert agent.run("Do the thing") == "recovered"

    tool_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "execute_tool"
    )
    statuses = {
        attr(span, GenAI.GEN_AI_TOOL_NAME): span.status.status_code
        for span in tool_spans
    }
    assert statuses["broken_tool"] == StatusCode.ERROR
    assert statuses["final_answer"] == StatusCode.UNSET
    broken = next(
        span
        for span in tool_spans
        if attr(span, GenAI.GEN_AI_TOOL_NAME) == "broken_tool"
    )
    assert attr(broken, error_attributes.ERROR_TYPE) == "ValueError"
