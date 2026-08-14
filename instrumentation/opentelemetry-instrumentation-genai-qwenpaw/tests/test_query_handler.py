# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert ``invoke_agent`` telemetry for ``AgentRunner.query_handler``."""

from __future__ import annotations

import json

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import SpanKind, StatusCode

from .harness import (
    failing_run_command_path,
    fake_run_command_path,
    make_request,
    patched_command_path,
    user_command_msgs,
)

pytest.importorskip("agentscope.message")

# LoongSuite-only extension attributes that must NOT survive the migration.
_REMOVED_EXTENSION_ATTRIBUTES = (
    "gen_ai.span.kind",
    "gen_ai.session.id",
    "gen_ai.user.id",
    "gen_ai.response.time_to_first_token",
    "qwenpaw.agent_id",
    "qwenpaw.channel",
)


async def _drain(stream):
    items = []
    async for item in stream:
        items.append(item)
    return items


@pytest.mark.asyncio
async def test_query_handler_emits_invoke_agent_span(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        items = await _drain(
            runner.query_handler(user_command_msgs(), make_request())
        )

    assert len(items) == 1
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.kind == SpanKind.INTERNAL
    assert span.status.status_code == StatusCode.UNSET

    attributes = dict(span.attributes or {})
    assert (
        attributes[GenAI.GEN_AI_OPERATION_NAME]
        == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    # qwenpaw's `agent_id` is a local config key (e.g. "default"), not a
    # provider-assigned stable identifier, so gen_ai.agent.id must not be
    # recorded per its semconv guidance.
    assert GenAI.GEN_AI_AGENT_ID not in attributes
    assert attributes[GenAI.GEN_AI_CONVERSATION_ID] == "sess-1"
    assert isinstance(attributes[GenAI.GEN_AI_CONVERSATION_ID], str)

    # `agent_name` was added during the qwenpaw 1.1.x line, so the attribute
    # (and span-name suffix) is optional on the oldest supported version.
    agent_name = getattr(runner, "agent_name", None)
    if agent_name:
        assert span.name == f"invoke_agent {agent_name}"
        assert attributes[GenAI.GEN_AI_AGENT_NAME] == agent_name
        assert isinstance(attributes[GenAI.GEN_AI_AGENT_NAME], str)
    else:
        assert span.name == "invoke_agent"
        assert GenAI.GEN_AI_AGENT_NAME not in attributes

    for removed in _REMOVED_EXTENSION_ATTRIBUTES:
        assert removed not in attributes


@pytest.mark.asyncio
async def test_query_handler_kwargs_call_form(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        await _drain(
            runner.query_handler(
                msgs=user_command_msgs(),
                request=make_request(session_id="sess-kw"),
            )
        )

    (span,) = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[GenAI.GEN_AI_CONVERSATION_ID] == "sess-kw"


@pytest.mark.asyncio
async def test_query_handler_without_request_omits_conversation_id(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        await _drain(runner.query_handler(user_command_msgs()))

    (span,) = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert GenAI.GEN_AI_CONVERSATION_ID not in attributes
    assert GenAI.GEN_AI_AGENT_ID not in attributes


@pytest.mark.asyncio
async def test_query_handler_captures_messages_in_span_only_mode(
    instrument_with_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(
        runner_module, fake_run_command_path("hello-output")
    ):
        await _drain(
            runner.query_handler(user_command_msgs("/stop"), make_request())
        )

    (span,) = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})

    assert isinstance(attributes[GenAI.GEN_AI_INPUT_MESSAGES], str)
    input_messages = json.loads(attributes[GenAI.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert input_messages[0]["parts"][0]["content"] == "/stop"

    assert isinstance(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES], str)
    output_messages = json.loads(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["parts"][0]["content"] == "hello-output"


@pytest.mark.asyncio
async def test_query_handler_omits_messages_without_content_capture(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        await _drain(runner.query_handler(user_command_msgs(), make_request()))

    (span,) = span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert GenAI.GEN_AI_INPUT_MESSAGES not in attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attributes


@pytest.mark.asyncio
async def test_stream_side_error_reraises_and_fails_span(
    instrument_with_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(
        runner_module, failing_run_command_path(ConnectionError("boom"))
    ):
        with pytest.raises(ConnectionError, match="boom"):
            await _drain(
                runner.query_handler(user_command_msgs(), make_request())
            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attributes = dict(span.attributes or {})
    assert attributes[error_attributes.ERROR_TYPE] == "ConnectionError"
    assert isinstance(attributes[error_attributes.ERROR_TYPE], str)
    # Output captured before the failure must survive on the error span.
    output_messages = json.loads(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["parts"][0]["content"] == "partial"


@pytest.mark.asyncio
async def test_caller_side_error_reraises_and_fails_span(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        with pytest.raises(ValueError, match="caller bug"):
            async with runner.query_handler(
                user_command_msgs(), make_request()
            ):
                raise ValueError("caller bug")

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attributes = dict(span.attributes or {})
    assert attributes[error_attributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_caller_side_error_closes_the_underlying_generator(
    instrument_no_content,
    runner_module,
    span_exporter,
):
    """Leaving the ``async with`` block early must not leave the turn running."""
    del span_exporter
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(
        runner_module, fake_run_command_path("early-output")
    ):
        stream = runner.query_handler(user_command_msgs(), make_request())
        with pytest.raises(ValueError, match="caller bug"):
            async with stream:
                await anext(stream)
                raise ValueError("caller bug")

    # A suspended async generator keeps a frame; closing it clears it.
    assert getattr(stream, "__wrapped__").ag_frame is None


@pytest.mark.asyncio
async def test_early_close_finalizes_span_with_partial_output(
    instrument_with_content,
    runner_module,
    span_exporter,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(
        runner_module, fake_run_command_path("early-output")
    ):
        stream = runner.query_handler(user_command_msgs(), make_request())
        first = await anext(stream)
        assert first is not None
        await stream.aclose()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    attributes = dict(span.attributes or {})
    output_messages = json.loads(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert output_messages[0]["parts"][0]["content"] == "early-output"


@pytest.mark.asyncio
async def test_operation_duration_metric_recorded(
    instrument_no_content,
    runner_module,
    metric_reader,
):
    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        await _drain(runner.query_handler(user_command_msgs(), make_request()))

    metrics_data = metric_reader.get_metrics_data()
    all_metrics = [
        metric
        for resource_metric in metrics_data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    ]
    # The INTERNAL invoke_agent invocation is not a streamed client call,
    # so only the operation duration is recorded — no streamed-call metrics
    # such as time_to_first_chunk.
    assert {metric.name for metric in all_metrics} == {
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    }
    duration_metrics = [
        metric
        for metric in all_metrics
        if metric.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ]
    assert len(duration_metrics) == 1
    data_points = list(duration_metrics[0].data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    assert point.count == 1
    assert point.sum >= 0
    assert (
        point.attributes[GenAI.GEN_AI_OPERATION_NAME]
        == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
    )


@pytest.mark.asyncio
async def test_uninstrument_restores_query_handler(
    runner_module,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    from opentelemetry.instrumentation.genai.qwenpaw import (
        QwenPawInstrumentor,
    )

    instrumentor = QwenPawInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()

    runner = runner_module.AgentRunner(agent_id="entry-agent")
    with patched_command_path(runner_module, fake_run_command_path("ok")):
        items = await _drain(
            runner.query_handler(user_command_msgs(), make_request())
        )

    assert len(items) == 1
    assert span_exporter.get_finished_spans() == ()
