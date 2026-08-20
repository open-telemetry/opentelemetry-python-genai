# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool (execute_tool) instrumentation tests."""

from __future__ import annotations

import json
import sys
import tempfile
from types import ModuleType
from typing import Any

import pytest
from smolagents import CodeAgent, Tool, ToolCallingAgent, tool
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)
from smolagents.monitoring import TokenUsage
from smolagents.tools import PipelineTool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .test_utils import (
    BrokenTool,
    FakeCodeModel,
    GetWeatherTool,
    ImageTool,
    attr,
    spans_by_operation,
)


class ForecastTool(Tool):
    """Two inputs, one of them optional, to exercise argument binding."""

    name = "get_forecast"
    description = "Get the forecast for a given city"
    inputs = {
        "location": {"type": "string", "description": "The city"},
        "unit": {
            "type": "string",
            "description": "The temperature unit",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, location: str, unit: str = "C") -> str:
        return f"sunny in {location} ({unit})"


def _tool_span(span_exporter: Any) -> Any:
    (span,) = spans_by_operation(
        span_exporter.get_finished_spans(), "execute_tool"
    )
    return span


def test_tool_returns_string(instrument_with_content, span_exporter) -> None:
    assert GetWeatherTool()("Paris") == "sunny"

    span = _tool_span(span_exporter)
    assert span.name == "execute_tool get_weather"
    assert attr(span, GenAI.GEN_AI_TOOL_NAME) == "get_weather"
    assert attr(span, GenAI.GEN_AI_TOOL_TYPE) == "function"
    assert (
        attr(span, GenAI.GEN_AI_TOOL_DESCRIPTION)
        == "Get the weather for a given city"
    )
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT) == "sunny"
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)) == {
        "location": "Paris"
    }


class ReversedInputsTool(Tool):
    """A tool whose declared ``inputs`` are ordered differently from ``forward``.

    smolagents accepts this: ``inputs`` describes the schema the model sees,
    while positional calls are bound by ``forward``'s signature.
    """

    name = "compare"
    description = "Compare two cities"
    inputs = {
        "second": {"type": "string", "description": "The second city"},
        "first": {"type": "string", "description": "The first city"},
    }
    output_type = "string"

    def forward(self, first: str, second: str) -> str:
        return f"{first} vs {second}"


@pytest.mark.parametrize(
    "tool_factory, args, kwargs, expected",
    [
        # Positional arguments are named after forward()'s parameters...
        (ForecastTool, ("Paris",), {}, {"location": "Paris"}),
        # ...including when only some of them are positional.
        (
            ForecastTool,
            ("Paris",),
            {"unit": "F"},
            {"location": "Paris", "unit": "F"},
        ),
        (ForecastTool, ("Paris", "F"), {}, {"location": "Paris", "unit": "F"}),
        (
            ForecastTool,
            (),
            {"location": "Paris", "unit": "F"},
            {"location": "Paris", "unit": "F"},
        ),
        # A lone dict of declared inputs is forwarded as kwargs by Tool.__call__.
        (ForecastTool, ({"location": "Paris"},), {}, {"location": "Paris"}),
        # The declared inputs order is not the call order: naming arguments
        # after it would record every value under the wrong name.
        (
            ReversedInputsTool,
            ("Paris", "Berlin"),
            {},
            {"first": "Paris", "second": "Berlin"},
        ),
    ],
)
def test_tool_argument_binding(
    instrument_with_content,
    span_exporter,
    tool_factory: type[Tool],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    result = tool_factory()(*args, **kwargs)

    span = _tool_span(span_exporter)
    arguments = json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS))
    assert arguments == expected
    # The tool's own output proves which parameter each value really reached.
    for value in expected.values():
        assert str(value) in result


def test_tool_argument_binding_falls_back_to_declared_inputs(
    instrument_with_content, span_exporter
) -> None:
    # The gradio and LangChain tool wrappers set
    # skip_forward_signature_validation and declare a generic forward, which
    # names nothing; the declared inputs are all there is to go on.
    class GenericForwardTool(Tool):
        skip_forward_signature_validation = True
        name = "generic"
        description = "Wraps a foreign tool"
        inputs = {"query": {"type": "string", "description": "The query"}}
        output_type = "string"

        def forward(self, *args: Any, **kwargs: Any) -> str:
            return f"ran {args}{kwargs}"

    GenericForwardTool()("weather")

    span = _tool_span(span_exporter)
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)) == {
        "query": "weather"
    }


def test_tool_argument_binding_drops_extra_positional(
    instrument_with_content, span_exporter
) -> None:
    # smolagents rejects the extra argument itself; the arguments recorded
    # before the call must not name it after an input it doesn't belong to.
    with pytest.raises(TypeError):
        ForecastTool()("Paris", "F", "extra")

    span = _tool_span(span_exporter)
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)) == {
        "location": "Paris",
        "unit": "F",
    }


def test_tool_returns_dict(instrument_with_content, span_exporter) -> None:
    class WeatherDictTool(Tool):
        name = "get_weather"
        description = "Get detailed weather"
        inputs = {"location": {"type": "string", "description": "city"}}
        output_type = "object"

        def forward(self, location: str) -> dict[str, Any]:
            return {"condition": "sunny", "temperature": 25}

    assert WeatherDictTool()("Paris") == {
        "condition": "sunny",
        "temperature": 25,
    }

    span = _tool_span(span_exporter)
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT)) == {
        "condition": "sunny",
        "temperature": 25,
    }


def test_tool_returns_tuple(instrument_with_content, span_exporter) -> None:
    @tool
    def get_population(location: str) -> tuple[str, str]:
        """Get population and location type.

        Args:
            location: the location
        """
        return f"Population in {location} is 10 million", "City"

    assert get_population("Paris") == (
        "Population in Paris is 10 million",
        "City",
    )

    span = _tool_span(span_exporter)
    assert attr(span, GenAI.GEN_AI_TOOL_NAME) == "get_population"
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT)) == [
        "Population in Paris is 10 million",
        "City",
    ]


def test_tool_returning_a_side_effecting_string_is_not_stringified(
    instrument_with_content, span_exporter
) -> None:
    # AgentText and AgentAudio subclass str and inherit AgentType.__str__, which
    # calls to_string(); for audio that writes a .wav into a temp directory.
    # Reading the underlying str keeps telemetry side-effect free.
    class SideEffectingText(str):
        def __str__(self) -> str:
            raise AssertionError("telemetry called __str__")

    @tool
    def transcribe(path: str) -> str:
        """Transcribe a recording.

        Args:
            path: the recording
        """
        return SideEffectingText("transcript")

    transcribe("clip.wav")

    span = _tool_span(span_exporter)
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT) == "transcript"


def test_tool_returning_an_image_records_its_type(
    instrument_with_content, span_exporter, monkeypatch
) -> None:
    # util-genai falls back to str() for values it can't serialize, which for a
    # PIL image is a heap address and for AgentImage is a PNG written to disk.
    def _no_temp_dirs(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("telemetry wrote an image to disk")

    monkeypatch.setattr(tempfile, "mkdtemp", _no_temp_dirs)

    ImageTool()()

    span = _tool_span(span_exporter)
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT) == "Image"


def test_tool_content_capture_disabled(
    instrument_no_content, span_exporter
) -> None:
    assert GetWeatherTool()("Paris") == "sunny"

    span = _tool_span(span_exporter)
    assert attr(span, GenAI.GEN_AI_TOOL_NAME) == "get_weather"
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS) is None
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT) is None


def test_tool_error_reraises_and_records(
    instrument_with_content, span_exporter
) -> None:
    with pytest.raises(ValueError, match="tool exploded"):
        BrokenTool()("Paris")

    span = _tool_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "ValueError"


def test_interrupted_tool_call_ends_the_span(
    instrument_with_content, span_exporter, lifecycle
) -> None:
    # A KeyboardInterrupt is not an Exception, and must still end the span.
    class InterruptingTool(Tool):
        name = "interrupting_tool"
        description = "Interrupts."
        inputs = {
            "location": {"type": "string", "description": "The location."}
        }
        output_type = "string"

        def forward(self, location: str) -> str:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        InterruptingTool()("Paris")

    span = _tool_span(span_exporter)
    assert span.status.status_code == StatusCode.UNSET
    assert lifecycle.leaked == []


class _ToolCallingModel:
    """Replays a scripted list of tool calls, one message per step."""

    model_id = "scripted-model"
    kwargs: dict[str, Any] = {}

    def __init__(self, steps: list[list[tuple[str, str, Any]]]) -> None:
        self._steps = steps
        self.calls = 0

    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        step = self._steps[min(self.calls, len(self._steps) - 1)]
        self.calls += 1
        return ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id=call_id,
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name=name, arguments=arguments
                    ),
                )
                for call_id, name, arguments in step
            ],
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> ChatMessage:
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        return message


def _call_ids_by_tool(span_exporter: Any) -> dict[str, Any]:
    return {
        attr(span, GenAI.GEN_AI_TOOL_NAME): attr(
            span, GenAI.GEN_AI_TOOL_CALL_ID
        )
        for span in spans_by_operation(
            span_exporter.get_finished_spans(), "execute_tool"
        )
    }


def test_tool_call_id_is_recorded(
    instrument_with_content, span_exporter
) -> None:
    # Tool.__call__ gets only the argument values, so the id comes from the
    # ToolCall objects the step published.
    agent = ToolCallingAgent(
        tools=[GetWeatherTool()],
        model=_ToolCallingModel(
            [
                [("call_weather", "get_weather", {"location": "Paris"})],
                [("call_final", "final_answer", "sunny in Paris")],
            ]
        ),
        max_steps=3,
    )
    agent.run("What is the weather in Paris?")

    assert _call_ids_by_tool(span_exporter) == {
        "get_weather": "call_weather",
        "final_answer": "call_final",
    }


def test_parallel_tool_calls_each_record_their_own_id(
    instrument_with_content, span_exporter
) -> None:
    # Two calls in one message run in worker threads. Each tool claims the
    # pending call that matches its name.
    agent = ToolCallingAgent(
        tools=[GetWeatherTool(), ForecastTool()],
        model=_ToolCallingModel(
            [
                [
                    ("call_weather", "get_weather", {"location": "Paris"}),
                    ("call_forecast", "get_forecast", {"location": "Rome"}),
                ],
                [("call_final", "final_answer", "done")],
            ]
        ),
        max_steps=3,
    )
    agent.run("Compare Paris and Rome.")

    call_ids = _call_ids_by_tool(span_exporter)
    assert call_ids["get_weather"] == "call_weather"
    assert call_ids["get_forecast"] == "call_forecast"


def test_repeated_calls_to_one_tool_in_a_step_record_no_id(
    instrument_with_content, span_exporter
) -> None:
    # execute_tool_call rewrites state-variable arguments, so two calls to one
    # tool in a step cannot be told apart.
    agent = ToolCallingAgent(
        tools=[GetWeatherTool()],
        model=_ToolCallingModel(
            [
                [
                    ("call_paris", "get_weather", {"location": "Paris"}),
                    ("call_rome", "get_weather", {"location": "Rome"}),
                ],
                [("call_final", "final_answer", "done")],
            ]
        ),
        max_steps=3,
    )
    agent.run("Compare Paris and Rome.")

    weather_ids = {
        attr(span, GenAI.GEN_AI_TOOL_CALL_ID)
        for span in spans_by_operation(
            span_exporter.get_finished_spans(), "execute_tool"
        )
        if attr(span, GenAI.GEN_AI_TOOL_NAME) == "get_weather"
    }
    assert weather_ids == {None}


def test_code_agent_tool_span_records_no_call_id(
    instrument_with_content, span_exporter
) -> None:
    # A CodeAgent's model writes code, so there is no tool call id to record.
    agent = CodeAgent(tools=[], model=FakeCodeModel(), max_steps=3)
    agent.run("Test question")

    assert _call_ids_by_tool(span_exporter) == {"final_answer": None}


class _WeatherCodeModel(FakeCodeModel):
    """Drives a CodeAgent to call ``get_weather`` from generated code."""

    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=(
                "Thought: check the weather.\nCode:\n```py\n"
                'final_answer(get_weather("Rome"))\n```<end_code>'
            ),
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def test_a_managed_agents_tool_does_not_claim_the_managers_call_id(
    instrument_with_content, span_exporter
) -> None:
    # The manager's step calls a managed CodeAgent and its own get_weather at
    # once, and the managed agent calls a get_weather of its own from code. A
    # CodeAgent publishes no tool calls, so unless the nested run clears the
    # manager's, its tool would claim the manager's id.
    managed = CodeAgent(
        tools=[GetWeatherTool()],
        model=_WeatherCodeModel(),
        max_steps=2,
        name="search_agent",
        description="Runs searches.",
    )
    manager = ToolCallingAgent(
        tools=[GetWeatherTool()],
        model=_ToolCallingModel(
            [
                [
                    ("call_delegate", "search_agent", {"task": "Look it up"}),
                    ("call_weather", "get_weather", {"location": "Paris"}),
                ],
                [("call_final", "final_answer", "done")],
            ]
        ),
        managed_agents=[managed],
        max_steps=2,
    )
    manager.run("Delegate and check Paris.")

    weather_ids = sorted(
        str(attr(span, GenAI.GEN_AI_TOOL_CALL_ID))
        for span in spans_by_operation(
            span_exporter.get_finished_spans(), "execute_tool"
        )
        if attr(span, GenAI.GEN_AI_TOOL_NAME) == "get_weather"
    )
    # The manager's call keeps its id; the managed agent's does not take it.
    assert weather_ids == ["None", "call_weather"]


class _EchoPipelineTool(PipelineTool):
    """A ``PipelineTool`` that echoes its prompt.

    ``PipelineTool.__init__`` needs torch and accelerate; the test only needs
    the inherited ``__call__``.
    """

    name = "echo_pipeline"
    description = "Echoes its input through a pipeline"
    inputs = {"prompt": {"type": "string", "description": "The prompt"}}
    output_type = "string"

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        self.is_initialized = True
        self.device = "cpu"

    def encode(self, raw_inputs: Any) -> dict[str, Any]:
        return {"prompt": raw_inputs}

    def forward(self, inputs: dict[str, Any]) -> str:
        return f"echo: {inputs['prompt']}"

    def decode(self, outputs: Any) -> Any:
        return outputs


@pytest.fixture
def fake_torch_and_accelerate(monkeypatch):
    """Satisfy the imports ``PipelineTool.__call__`` does at call time."""
    torch = ModuleType("torch")
    # A real class, because __call__ splits its inputs with isinstance().
    setattr(torch, "Tensor", type("Tensor", (), {}))
    accelerate = ModuleType("accelerate")
    accelerate_utils = ModuleType("accelerate.utils")
    setattr(accelerate_utils, "send_to_device", lambda obj, _device: obj)
    setattr(accelerate, "utils", accelerate_utils)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)


def test_pipeline_tool_emits_one_span(
    instrument_with_content, span_exporter, fake_torch_and_accelerate
) -> None:
    # PipelineTool overrides Tool.__call__ without delegating to it. Patching
    # the defining class shadows Tool's patch, so a call emits one span.
    assert _EchoPipelineTool()("hello") == "echo: hello"

    span = _tool_span(span_exporter)
    assert attr(span, GenAI.GEN_AI_TOOL_NAME) == "echo_pipeline"
    assert attr(span, GenAI.GEN_AI_TOOL_TYPE) == "function"
    assert attr(span, GenAI.GEN_AI_TOOL_CALL_RESULT) == "echo: hello"
    assert json.loads(attr(span, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS)) == {
        "prompt": "hello"
    }


def test_pipeline_tool_error_reraises_and_records(
    instrument_with_content, span_exporter, fake_torch_and_accelerate
) -> None:
    class _BrokenPipelineTool(_EchoPipelineTool):
        def forward(self, inputs: dict[str, Any]) -> str:
            raise ValueError("pipeline exploded")

    with pytest.raises(ValueError, match="pipeline exploded"):
        _BrokenPipelineTool()("hello")

    span = _tool_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert attr(span, error_attributes.ERROR_TYPE) == "ValueError"
