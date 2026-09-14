# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers, tools, and model stubs for smolagents instrumentation tests.

The in-process model helpers cannot be recorded with VCR because they make no
HTTP requests. Each factory bypasses ``__init__`` to avoid loading model weights
and stubs the runtime code that the real ``generate`` method calls. The tests
therefore exercise smolagents' ``generate`` method and its wrapper.

Agent fakes emit no ``chat`` spans.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from smolagents import Tool
from smolagents.models import ChatMessage, ChatMessageStreamDelta
from smolagents.monitoring import TokenUsage

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

CODE_FINAL_ANSWER = """
Thought: Return the final answer.
Code:
```py
final_answer("Test result from CodeAgent")
```<end_code>
"""

CODE_NO_OP = """
Thought: Keep going.
Code:
```py
x = 1
```<end_code>
"""


class FakeCodeModel:
    """Drive a CodeAgent without emitting a ``chat`` span."""

    def __init__(self, model_id: str = "fake-model") -> None:
        self.model_id = model_id
        self.kwargs: dict[str, Any] = {}

    def generate(
        self,
        messages: list[Any],
        stop_sequences: list[str] | None = None,
        response_format: Any = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=CODE_FINAL_ANSWER,
            token_usage=TokenUsage(input_tokens=11, output_tokens=7),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> ChatMessage:
        return self.generate(*args, **kwargs)


class ModelWithoutModelId(FakeCodeModel):
    def __init__(self) -> None:
        super().__init__()
        del self.model_id


class FakeStreamingCodeModel(FakeCodeModel):
    def generate_stream(
        self,
        messages: list[Any],
        stop_sequences: list[str] | None = None,
        response_format: Any = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        yield ChatMessageStreamDelta(
            content=CODE_FINAL_ANSWER,
            token_usage=TokenUsage(input_tokens=11, output_tokens=7),
        )


class NeverFinishingCodeModel(FakeCodeModel):
    """Trigger smolagents' synthesized max-step answer."""

    def generate(self, messages: list[Any], **kwargs: Any) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=CODE_NO_OP,
            token_usage=TokenUsage(input_tokens=3, output_tokens=5),
        )


class ImageTool(Tool):
    name = "make_image"
    description = "Make a tiny image"
    inputs: dict[str, Any] = {}
    output_type = "image"

    def forward(self) -> Any:
        return Image.new("RGB", (4, 4), color="red")


class BrokenTool(Tool):
    name = "broken_tool"
    description = "A tool that always fails"
    inputs = {"location": {"type": "string", "description": "ignored"}}
    output_type = "string"

    def forward(self, location: str) -> str:
        raise ValueError("tool exploded")


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a given city"
    inputs = {
        "location": {
            "type": "string",
            "description": "The city to get the weather for",
        }
    }
    output_type = "string"

    def forward(self, location: str) -> str:
        return "sunny"


# The in-process runtimes flatten a message's content as text, so the content
# has to be a list of parts rather than a bare string.
MESSAGES: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Where is the Louvre?"}],
    }
]


class _PromptTokens:
    """The part of a torch tensor that ``TransformersModel.generate`` uses.

    It reads ``inputs.shape[1]`` for the prompt length and slices the generated
    tail out of the model's output with ``out[0, prompt_length:]``. Moving to a
    device and the ``input_ids`` unwrap are part of the same path.
    """

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.ids))

    def to(self, _device: Any) -> _PromptTokens:
        return self

    def __getitem__(self, key: Any) -> list[int]:
        _row, columns = key
        return self.ids[columns]


def transformers_model(
    prompt_ids: list[int] | None = None,
    generated_ids: list[int] | None = None,
    text: str = "In Paris",
    error: Exception | None = None,
    stream_chunks: list[str] | None = None,
    **model_kwargs: Any,
) -> Any:
    """A ``TransformersModel`` whose tokenizer and weights are stubbed.

    ``generate`` builds the prompt through ``self.tokenizer
    .apply_chat_template``, counts its tokens, calls ``self.model.generate`` and
    decodes the tail. ``generate_stream`` instead runs ``self.model.generate``
    on a thread and iterates ``self.streamer``, so the stubbed streamer is what
    yields the deltas.
    """
    from smolagents.models import TransformersModel

    prompt_ids = prompt_ids or [1, 2, 3]
    generated_ids = generated_ids or [4, 5]

    def generate(**_: Any) -> Any:
        if error is not None:
            raise error
        return _PromptTokens(prompt_ids + generated_ids)

    model = object.__new__(TransformersModel)
    model.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model.kwargs = dict(model_kwargs)
    model.flatten_messages_as_text = True
    model.apply_chat_template_kwargs = {}
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_: _PromptTokens(prompt_ids),
        decode=lambda ids, **_: text,
    )
    model.model = SimpleNamespace(device="cpu", generate=generate)
    model.streamer = iter(stream_chunks or [])
    return model


def mlx_model(text: str = "In Paris", **model_kwargs: Any) -> Any:
    """An ``MLXModel`` whose ``mlx_lm`` pieces are stubbed.

    ``MLXModel.generate`` imports nothing itself; it drives ``stream_generate``
    over ``self.model`` and ``self.tokenizer``, which ``__init__`` loads from
    ``mlx_lm``. Bypassing ``__init__`` is therefore enough to run the real
    ``generate`` without the runtime installed.
    """
    from smolagents.models import MLXModel

    model = object.__new__(MLXModel)
    model.model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    model.kwargs = dict(model_kwargs)
    model.flatten_messages_as_text = True
    model.apply_chat_template_kwargs = {}
    model.model = object()
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, tools=None, **_: [1, 2, 3]
    )
    # ``generate`` counts one output token per delta, so the deltas are the
    # words of ``text`` and the output token count is the word count.
    words = text.split(" ")
    deltas = [
        SimpleNamespace(text=word if index == 0 else f" {word}")
        for index, word in enumerate(words)
    ]
    model.stream_generate = lambda *_, **__: iter(deltas)
    return model


def vllm_model(
    monkeypatch: pytest.MonkeyPatch,
    text: str = "In Paris",
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    **model_kwargs: Any,
) -> Any:
    """A ``VLLMModel`` with ``vllm`` itself stubbed.

    ``VLLMModel.generate`` imports ``SamplingParams`` and
    ``StructuredOutputsParams`` from ``vllm`` when it runs, and no test env
    installs vllm, so both modules are faked for the duration of the test.
    Everything the wrapper reads still comes from the real ``generate``.
    """
    from smolagents.models import VLLMModel

    def fake_params(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    vllm = ModuleType("vllm")
    sampling_params = ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", fake_params)
    setattr(sampling_params, "StructuredOutputsParams", fake_params)
    setattr(vllm, "sampling_params", sampling_params)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    completion = SimpleNamespace(
        prompt_token_ids=prompt_token_ids or [1, 2, 3, 4],
        outputs=[
            SimpleNamespace(text=text, token_ids=output_token_ids or [5, 6])
        ],
    )
    model = object.__new__(VLLMModel)
    model.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model.kwargs = dict(model_kwargs)
    model.flatten_messages_as_text = True
    model._is_vlm = False
    model.apply_chat_template_kwargs = {}
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_: "prompt"
    )
    model.model = SimpleNamespace(generate=lambda *_, **__: [completion])
    return model


def spans_by_operation(
    spans: list[ReadableSpan], operation: str
) -> list[ReadableSpan]:
    return [
        span
        for span in spans
        if (span.attributes or {}).get(GenAI.GEN_AI_OPERATION_NAME)
        == operation
    ]


def attr(span: ReadableSpan, name: str) -> Any:
    return (span.attributes or {}).get(name)


def parse_messages(span: ReadableSpan, name: str) -> list[dict[str, Any]]:
    raw = attr(span, name)
    return json.loads(raw) if isinstance(raw, str) else []


def part_types(messages: list[dict[str, Any]]) -> list[str]:
    return [part["type"] for message in messages for part in message["parts"]]


def metrics_by_name(metric_reader: Any) -> dict[str, Any]:
    data = metric_reader.get_metrics_data()
    if data is None:
        return {}
    return {
        metric.name: metric
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


def data_point_attributes(metric: Any) -> list[dict[str, Any]]:
    return [dict(point.attributes) for point in metric.data.data_points]


class LifecycleRecorder(SpanProcessor):
    """Records span starts and ends.

    The exporter only sees a span once it ends, so an unfinished span reads
    there as no span at all.
    """

    def __init__(self) -> None:
        self.started: list[str] = []
        self.ended: list[str] = []

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        self.started.append(span.name)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span.name)

    @property
    def leaked(self) -> list[str]:
        return self.started[len(self.ended) :]
