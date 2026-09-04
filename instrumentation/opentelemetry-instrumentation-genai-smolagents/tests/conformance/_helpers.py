# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from smolagents import Tool


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


MESSAGES: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Where is the Louvre?"}],
    },
]


class _PromptTokens:
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


def vllm_model(
    text: str = "In Paris",
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    **model_kwargs: Any,
) -> Any:
    from smolagents.models import VLLMModel

    def fake_params(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    vllm = ModuleType("vllm")
    sampling_params = ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", fake_params)
    setattr(sampling_params, "StructuredOutputsParams", fake_params)
    setattr(vllm, "sampling_params", sampling_params)
    sys.modules["vllm"] = vllm
    sys.modules["vllm.sampling_params"] = sampling_params

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
