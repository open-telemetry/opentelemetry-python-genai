# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest.mock
from typing import Any


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
