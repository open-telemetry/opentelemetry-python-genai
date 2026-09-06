# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import json
import pathlib
import tempfile
import weakref
from collections.abc import Generator
from types import GeneratorType, SimpleNamespace
from typing import Any

import pytest
from smolagents import (
    AgentExecutionError,
    CodeAgent,
    OpenAIModel,
    ToolCallingAgent,
)
from smolagents.memory import ActionStep, FinalAnswerStep
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)
from smolagents.monitoring import Timing, TokenUsage

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
    ModelWithoutModelId,
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

    # OpenAI SDK instrumentation is disabled, so only the agent emits a span.
    assert sorted(filter(None, _operations(spans))) == ["invoke_agent"]
    assert agent_span.name == "invoke_agent ToolCallingAgent"

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
    # Binding maps the positional value in ``run(task)`` to the ``task`` parameter.
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


def test_agent_event_only_records_no_inline_content(
    instrument_event_only, span_exporter, log_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert attr(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES) is None
    assert not log_exporter.get_finished_logs()


def test_agent_span_and_event_records_content_on_span_only(
    instrument_span_and_event, span_exporter, log_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    inputs = parse_messages(agent_span, GenAI.GEN_AI_INPUT_MESSAGES)
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert inputs[0]["parts"][0]["content"] == "Test question"
    assert outputs[0]["parts"][0]["content"] == "Test result from CodeAgent"
    assert not log_exporter.get_finished_logs()


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
    # Calling ``AgentImage.__str__`` writes a PNG and mutates the instance.
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


def _first_step_only(agent: CodeAgent) -> None:
    for _ in agent.run("Test question", stream=True):
        break


def test_break_on_final_answer_step_finishes_the_span(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)
    for step in stream:
        if isinstance(step, FinalAnswerStep):
            break

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)


def test_abandoned_streaming_run_finalizes_the_span(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    _first_step_only(agent)
    gc.collect()

    assert (
        len(
            spans_by_operation(
                span_exporter.get_finished_spans(), "invoke_agent"
            )
        )
        == 1
    )


def test_abandoned_streaming_run_does_not_leak_context(
    instrument_with_content, span_exporter, tracer_provider
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    _first_step_only(agent)
    gc.collect()

    with tracer_provider.get_tracer("test").start_as_current_span("unrelated"):
        pass

    (unrelated,) = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "unrelated"
    ]
    assert unrelated.parent is None


def test_code_agent_streaming_is_lazy_until_drained(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)

    assert (
        spans_by_operation(span_exporter.get_finished_spans(), "invoke_agent")
        == []
    )

    list(stream)

    spans = span_exporter.get_finished_spans()
    (agent_span,) = spans_by_operation(spans, "invoke_agent")
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"][0]["content"] == "Test result from CodeAgent"


def test_streaming_run_stays_a_generator(
    instrument_with_content, span_exporter
) -> None:
    # Callers branch on these checks, so a wrapper that fails them changes
    # behavior.
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
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    metrics = metrics_by_name(metric_reader)
    duration = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION]
    assert {
        GenAI.GEN_AI_OPERATION_NAME: "invoke_agent",
        GenAI.GEN_AI_REQUEST_MODEL: "fake-model",
    } in data_point_attributes(duration)
    # A run reports no token counts of its own: each model call records its
    # own, and totalling the steps here would count them a second time.
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE not in metrics
    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_USAGE_INPUT_TOKENS) is None
    assert attr(agent_span, GenAI.GEN_AI_USAGE_OUTPUT_TOKENS) is None


@pytest.mark.parametrize("stream", [False, True])
def test_run_that_exhausts_max_steps_is_not_reported_as_a_plain_stop(
    instrument_with_content, span_exporter, stream: bool
) -> None:
    # At ``max_steps``, smolagents records ``AgentMaxStepsError`` on the final
    # ``ActionStep`` and returns normally. Reporting "stop" would hide that the
    # agent reached the limit.
    agent = CodeAgent(tools=[], model=NeverFinishingCodeModel(), max_steps=1)
    if stream:
        list(agent.run("Test question", stream=True))
    else:
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == (
        "max_steps",
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert attr(agent_span, error_attributes.ERROR_TYPE) is None
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["finish_reason"] == "max_steps"


@pytest.mark.parametrize("stream", [False, True])
def test_terminal_step_error_is_reported_as_error_finish_reason(
    instrument_with_content, span_exporter, monkeypatch, stream: bool
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _terminal_error(
        *_args: Any, **_kwargs: Any
    ) -> Generator[FinalAnswerStep, None, None]:
        agent.memory.steps.append(
            ActionStep(
                step_number=1,
                timing=Timing(start_time=0),
                error=AgentExecutionError("terminal error", agent.logger),
            )
        )
        yield FinalAnswerStep(output="fallback")

    monkeypatch.setattr(agent, "_run_stream", _terminal_error)
    if stream:
        list(agent.run("Test question", stream=True))
    else:
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("error",)
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["finish_reason"] == "error"


def test_streaming_run_close_finalizes_once(
    instrument_with_content, span_exporter
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)

    # No ``FinalAnswerStep`` was observed, so no output is recorded.
    stream.close()
    stream.close()

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert len(agent_spans) == 1
    assert attr(agent_spans[0], GenAI.GEN_AI_OUTPUT_MESSAGES) is None


@pytest.mark.parametrize("use_context_manager", [True, False])
def test_streaming_run_stopped_midway_preserves_close_error(
    instrument_with_content, span_exporter, use_context_manager: bool
) -> None:
    # smolagents yields from a finally block, so its generator rejects an early
    # close. Instrumentation must preserve that exception.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    stream = agent.run("Test question", stream=True)
    with pytest.raises(RuntimeError, match="generator ignored GeneratorExit"):
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
    assert agent_span.status.status_code == StatusCode.ERROR
    assert attr(agent_span, error_attributes.ERROR_TYPE) == "RuntimeError"
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) is None


def test_streaming_run_close_reraises_a_real_cleanup_error(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
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
    assert lifecycle.leaked == []


def test_caller_error_inside_stream_context(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    with pytest.raises(RuntimeError, match="caller exploded"):
        with agent.run("Test question", stream=True) as stream:
            next(stream)
            raise RuntimeError("caller exploded")

    agent_spans = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert len(agent_spans) == 1
    assert agent_spans[0].status.status_code == StatusCode.ERROR
    assert attr(agent_spans[0], error_attributes.ERROR_TYPE) == "RuntimeError"
    assert lifecycle.leaked == []


def test_run_bad_call_records_an_error_span(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    with pytest.raises(TypeError):
        agent.run()

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.ERROR
    assert attr(agent_span, error_attributes.ERROR_TYPE) == "TypeError"
    assert attr(agent_span, GenAI.GEN_AI_INPUT_MESSAGES) is None
    assert lifecycle.leaked == []


def test_run_failure_before_stream(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
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
    assert lifecycle.leaked == []


def test_stream_failure_during_iteration(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
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
    assert lifecycle.leaked == []


def test_interrupted_stream_iteration_ends_the_span(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _interrupt(*args: Any, **kwargs: Any):
        if False:
            yield None
        raise KeyboardInterrupt

    monkeypatch.setattr(agent, "_run_stream", _interrupt)
    stream = agent.run("Test question", stream=True)

    with pytest.raises(KeyboardInterrupt):
        next(stream)

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert lifecycle.leaked == []


def test_interrupted_agent_recording_ends_the_span(
    instrument_with_content, span_exporter, lifecycle, monkeypatch
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    def _interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(patch_module, "to_tool_definitions", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        agent.run("Test question")

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert lifecycle.leaked == []


@pytest.fixture
def signature_calls(monkeypatch) -> list[str]:
    original_signature = patch_module.signature
    calls: list[str] = []

    def _signature(callable_: Any):
        calls.append(getattr(callable_, "__qualname__", repr(callable_)))
        return original_signature(callable_)

    monkeypatch.setattr(patch_module, "_signatures", {})
    monkeypatch.setattr(patch_module, "signature", _signature)
    return calls


def test_agent_run_caches_the_signature(
    instrument_with_content, signature_calls: list[str]
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)

    agent.run("First question")
    agent.run("Second question")

    assert signature_calls == ["MultiStepAgent.run"]


def test_the_signature_cache_does_not_grow_per_agent(
    instrument_with_content, signature_calls: list[str]
) -> None:
    agents = [
        CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
        for _ in range(5)
    ]
    for index, agent in enumerate(agents):
        agent.run(f"Question {index}")

    assert signature_calls == ["MultiStepAgent.run"]
    assert len(patch_module._signatures) == 1


def test_the_signature_cache_does_not_retain_agents(
    instrument_with_content, signature_calls: list[str]
) -> None:
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")
    reference = weakref.ref(agent)

    del agent
    gc.collect()

    assert reference() is None


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
    # Interrupting a run is not the agent failing, so the span ends without an
    # error.
    assert agent_span.status.status_code == StatusCode.UNSET
    assert lifecycle.leaked == []


def test_a_model_without_a_model_id_is_supported(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    # smolagents allows models without ``model_id``. Instrumentation must also
    # allow them.
    agent = CodeAgent(tools=[], model=ModelWithoutModelId(), max_steps=3)
    assert agent.run("Test question") == "Test result from CodeAgent"

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert attr(agent_span, GenAI.GEN_AI_REQUEST_MODEL) is None
    assert lifecycle.leaked == []


class _ManagerModel:
    model_id = "manager-model"
    kwargs: dict[str, Any] = {}

    def generate(self, messages, **kwargs) -> ChatMessage:
        # The first step starts with three messages.
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


def test_managed_agent_run_is_recorded(
    instrument_with_content, span_exporter
) -> None:
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
    # ``CodeAgent`` runs managed agents in a worker thread without caller
    # context. The managed run therefore starts a separate trace.
    assert managed_span.parent is None
    assert (
        attr(managed_span, GenAI.GEN_AI_AGENT_DESCRIPTION) == "Runs searches."
    )
    assert attr(manager_span, GenAI.GEN_AI_AGENT_DESCRIPTION) is None
    # Managed agents are exposed to the manager model as tools.
    definitions = json.loads(attr(manager_span, GenAI.GEN_AI_TOOL_DEFINITIONS))
    assert [definition["name"] for definition in definitions] == [
        "final_answer",
        "search_agent",
    ]
    search_agent = definitions[1]
    assert search_agent["description"] == "Runs searches."
    assert "task" in search_agent["parameters"]["properties"]


class _RecoveringModel:
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


def test_a_tool_error_the_agent_recovers_from_leaves_the_run_successful(
    instrument_with_content, span_exporter
) -> None:
    # smolagents gives tool errors back to the model, so a recovered run stays
    # successful.
    agent = ToolCallingAgent(
        tools=[BrokenTool()], model=_RecoveringModel(), max_steps=4
    )
    assert agent.run("Do the thing") == "recovered"

    (agent_span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "invoke_agent"
    )
    assert agent_span.status.status_code == StatusCode.UNSET
    assert attr(agent_span, error_attributes.ERROR_TYPE) is None
    assert attr(agent_span, GenAI.GEN_AI_RESPONSE_FINISH_REASONS) == ("stop",)
    outputs = parse_messages(agent_span, GenAI.GEN_AI_OUTPUT_MESSAGES)
    assert outputs[0]["parts"][0]["content"] == "recovered"
