# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Pipeline.run`` / ``Pipeline.run_async`` -> ``invoke_workflow``."""

import pytest
from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .test_utils import assert_chat_span_attributes


@pytest.mark.vcr
def test_pipeline_run_produces_workflow_and_chat_spans(
    span_exporter, instrument_with_content
):
    """A prompt-builder + chat-generator pipeline yields exactly the spans this
    migration supports: one for the classified generator, one for the
    pipeline itself. ``ChatPromptBuilder`` has no util-genai invocation type
    (see MIGRATION_REPORT.md) and produces no span of its own."""
    pipeline = Pipeline()
    prompt_builder = ChatPromptBuilder()
    llm = OpenAIChatGenerator(model="gpt-4o")
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    messages = [
        ChatMessage.from_system("Answer concisely in one sentence."),
        ChatMessage.from_user("What country is {{location}} in?"),
    ]
    pipeline.run(
        data={
            "prompt_builder": {
                "template_variables": {"location": "Berlin"},
                "template": messages,
            }
        }
    )

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "chat gpt-4o",
        "invoke_workflow Pipeline",
    ]

    chat_span, workflow_span = spans
    assert chat_span.status.is_ok
    assert_chat_span_attributes(
        chat_span,
        request_model="gpt-4o",
        provider=GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        response_model="gpt-4o-2024-05-13",
        input_tokens=25,
        output_tokens=2,
        finish_reasons=("stop",),
    )
    assert chat_span.parent is not None
    assert chat_span.parent.span_id == workflow_span.context.span_id

    assert workflow_span.status.is_ok
    workflow_attributes = workflow_span.attributes or {}
    assert (
        workflow_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == "invoke_workflow"
    )


@pytest.mark.vcr
async def test_pipeline_run_async_produces_workflow_and_chat_spans(
    span_exporter, instrument_with_content
):
    pipeline = Pipeline()
    llm = OpenAIChatGenerator(model="gpt-4o")
    pipeline.add_component("llm", llm)

    messages = [
        ChatMessage.from_system("Answer user questions succinctly"),
        ChatMessage.from_assistant("What can I help you with?"),
        ChatMessage.from_user(
            "Who won the World Cup in 2022? Answer in one word."
        ),
    ]
    await pipeline.run_async(data={"llm": {"messages": messages}})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "chat gpt-4o",
        "invoke_workflow Pipeline",
    ]

    chat_span, workflow_span = spans
    assert_chat_span_attributes(
        chat_span,
        request_model="gpt-4o",
        response_model="gpt-4o-2024-08-06",
        input_tokens=42,
        output_tokens=2,
    )
    assert chat_span.parent.span_id == workflow_span.context.span_id
