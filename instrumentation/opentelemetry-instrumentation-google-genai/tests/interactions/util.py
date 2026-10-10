# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest.mock
from typing import Any

try:
    from pydantic import TypeAdapter

    try:
        # Google GenAI < 2.9.0
        from google.genai._interactions.types.interaction_sse_event import (
            InteractionSSEEvent,
        )
    except ImportError:
        # Google GenAI >= 2.9.0
        from google.genai._gaos.types.interactions import InteractionSSEEvent

    _HAS_SSE_EVENT = True
except ImportError:
    _HAS_SSE_EVENT = False


def parse_sse_events(payloads: list[dict[str, Any]]) -> list[Any]:
    """Real SDK event models, so model-only behaviour is exercised.

    A dict double cannot surface the SDK's nested models -- a search result set
    arrives as a list of models, which has to be serializable by the time it
    lands on a span.
    """
    if not _HAS_SSE_EVENT:
        raise RuntimeError("google-genai interactions types are unavailable")
    adapter = TypeAdapter(InteractionSSEEvent)
    return [adapter.validate_python(payload) for payload in payloads]


def create_request_parameters() -> tuple[dict[str, object], dict[str, object]]:
    try:
        from google.genai._interactions.types.generation_config import (
            GenerationConfig,
        )
        from google.genai._interactions.types.text_response_format import (
            TextResponseFormat,
        )
    except ImportError:
        from google.genai._gaos.types.interactions import (
            GenerationConfig,
            TextResponseFormat,
        )

    config: dict[str, object] = {
        "max_output_tokens": 2048,
        "seed": 0,
        "stop_sequences": ["<END>"],
    }
    expected_attributes: dict[str, object] = {
        "gen_ai.request.max_tokens": 2048,
        "gen_ai.request.seed": 0,
        "gen_ai.request.stop_sequences": ["<END>"],
        "gen_ai.output.type": "text",
    }
    for name, value in (("temperature", 0.0), ("top_p", 0.75)):
        if name in GenerationConfig.model_fields:
            config[name] = value
            expected_attributes[f"gen_ai.request.{name}"] = value

    return (
        {
            "generation_config": GenerationConfig(**config),
            "response_format": TextResponseFormat(
                type="text", mime_type="text/plain"
            ),
        },
        expected_attributes,
    )


def create_mock_interaction(
    interaction_id: str = "test-id",
    model_name: str = "test-model",
    input_text: str = "user input",
    output_text: str = "model output",
    input_tokens: int = 10,
    output_tokens: int = 20,
    input_tokens_by_modality: list[Any] | None = None,
    output_tokens_by_modality: list[Any] | None = None,
    cached_tokens_by_modality: list[Any] | None = None,
) -> Any:
    mock_usage = unittest.mock.MagicMock()
    mock_usage.total_input_tokens = input_tokens
    mock_usage.total_output_tokens = output_tokens
    mock_usage.total_thought_tokens = 0
    mock_usage.total_cached_tokens = 0
    mock_usage.input_tokens_by_modality = input_tokens_by_modality
    mock_usage.output_tokens_by_modality = output_tokens_by_modality
    mock_usage.cached_tokens_by_modality = cached_tokens_by_modality

    mock_user_step = unittest.mock.MagicMock()
    mock_user_step.type = "user_input"
    mock_user_part = unittest.mock.MagicMock()
    mock_user_part.text = input_text
    mock_user_step.content = [mock_user_part]

    mock_model_step = unittest.mock.MagicMock()
    mock_model_step.type = "model_output"
    mock_model_part = unittest.mock.MagicMock()
    mock_model_part.text = output_text
    mock_model_step.content = [mock_model_part]

    mock_interaction = unittest.mock.MagicMock()
    mock_interaction.id = interaction_id
    mock_interaction.model = model_name
    mock_interaction.usage = mock_usage
    mock_interaction.steps = [mock_user_step, mock_model_step]
    mock_interaction.output_text = output_text

    return mock_interaction


def create_mock_content_event() -> Any:
    event = unittest.mock.MagicMock()
    event.event_type = "content"
    event.interaction = None
    return event


def create_mock_completed_event(interaction: Any) -> Any:
    event = unittest.mock.MagicMock()
    event.event_type = "interaction_completed"
    event.interaction = interaction
    return event


def create_mock_fetched_interaction(
    interaction_id: str = "fetched-id",
    model_name: str = "gemini-2.5-flash",
    status: str = "completed",
    output_text: str = "model output",
    system_instruction: str | None = None,
    tools: list[Any] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> Any:
    interaction = create_mock_interaction(
        interaction_id=interaction_id,
        model_name=model_name,
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    # Explicit, since a MagicMock would otherwise hand back a truthy mock for
    # every attribute the fetch mapping reads.
    interaction.status = status
    interaction.system_instruction = system_instruction
    interaction.tools = tools
    return interaction


class FakeStream:
    """Minimal stand-in for the SDK's sync ``Stream``."""

    def __init__(
        self,
        events: list[Any],
        error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._error = error
        self._close_error = close_error
        self.closed = False

    def __iter__(self) -> Any:
        yield from self._events
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class FakeAsyncStream:
    """Minimal stand-in for the SDK's ``AsyncStream``."""

    def __init__(
        self,
        events: list[Any],
        error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._error = error
        self._close_error = close_error
        self.closed = False

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


def create_mock_sse_completed_event(interaction: Any) -> Any:
    """A completion event spelled the way the SDK actually parses it."""
    event = unittest.mock.MagicMock()
    event.event_type = "interaction.completed"
    event.interaction = interaction
    return event


def create_mock_error_event(
    message: str = "The model is currently experiencing high demand.",
    code: str = "service_unavailable",
) -> Any:
    """An in-band stream error, as the provider reports over a 200 response."""
    event = unittest.mock.MagicMock()
    event.event_type = "error"
    event.interaction = None
    event.error = {"message": message, "code": code}
    return event


def create_mock_step_event(
    event_type: str,
    index: int = 0,
    step_type: str | None = None,
    text: str | None = None,
    step: dict[str, Any] | None = None,
    delta: dict[str, Any] | None = None,
) -> Any:
    """A step.start / step.delta / step.stop event, as streams emit them.

    ``step.start`` carries the whole step and ``step.delta`` a fragment of it,
    so ``step`` / ``delta`` take full payloads where a tool call needs one.
    """
    event = unittest.mock.MagicMock()
    event.event_type = event_type
    event.interaction = None
    event.index = index
    if step is None and step_type is not None:
        step = {"type": step_type}
    if delta is None and text is not None:
        delta = {"text": text}
    event.step = step
    event.delta = delta
    return event
