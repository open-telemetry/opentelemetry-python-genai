# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock

from opentelemetry._logs import Logger
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.semconv.aws import AWS_BEDROCK_GUARDRAIL_ID
from opentelemetry.util.genai.semconv.gen_ai import (
    GenAiOperationName,
    GenAiTokenType,
)
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    EmbeddingsClientOperation,
    ExecuteToolInternalOperation,
    FetchResponseClientOperation,
    InferenceClientOperation,
    InvokeAgentInternalOperation,
    InvokeWorkflowInternalOperation,
    RetrievalClientOperation,
)
from opentelemetry.util.genai.semconv.openai import OpenAIApiType
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    InputMessage,
    TextPart,
)


class TestOperations(TestBase):
    def test_inference_client_operation_lifecycle(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        logger = MagicMock(spec=Logger)

        op = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
            request_model="model",
            server_address="example.com",
            attributes={"custom.attribute": "value"},
            metric_attributes={"metric.attribute": "custom"},
        )
        op.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="hello")])
        ]
        op.request_temperature = 0.7
        op.response_finish_reasons = ["custom-finish-reason"]
        op.usage_input_tokens = 10
        op.usage_output_tokens = 20

        op.start("chat model")
        op.record_token_usage(10, token_type=GenAiTokenType.INPUT)
        op.record_token_usage(20, token_type=GenAiTokenType.OUTPUT)
        op.finish(duration_s=1.25)

        (recorded,) = self.get_finished_spans()
        assert recorded.kind is SpanKind.CLIENT
        assert recorded.attributes is not None
        assert recorded.attributes["gen_ai.provider.name"] == "custom-provider"
        assert recorded.attributes["gen_ai.operation.name"] == "chat"
        assert recorded.attributes["gen_ai.request.model"] == "model"
        assert recorded.attributes["server.address"] == "example.com"
        assert recorded.attributes["gen_ai.request.temperature"] == 0.7
        assert recorded.attributes["gen_ai.response.finish_reasons"] == (
            "custom-finish-reason",
        )
        assert recorded.attributes["gen_ai.usage.input_tokens"] == 10
        assert recorded.attributes["gen_ai.usage.output_tokens"] == 20
        assert recorded.attributes["custom.attribute"] == "value"
        assert (
            json.loads(recorded.attributes["gen_ai.input.messages"])[0]["role"]
            == "user"
        )

        logger.emit.assert_called_once()
        (event_record,) = logger.emit.call_args[0]
        event_attrs = event_record.attributes
        assert event_attrs is not None
        assert event_attrs["gen_ai.operation.name"] == "chat"
        assert event_attrs["gen_ai.provider.name"] == "custom-provider"
        assert event_attrs["gen_ai.request.model"] == "model"
        assert event_attrs["gen_ai.request.temperature"] == 0.7
        assert event_attrs["custom.attribute"] == "value"

        metrics = {
            metric.name: list(metric.data.data_points)
            for metric in self.get_sorted_metrics() or []
        }
        duration_points = metrics["gen_ai.client.operation.duration"]
        assert len(duration_points) == 1
        assert duration_points[0].sum == 1.25
        assert duration_points[0].attributes["gen_ai.operation.name"] == "chat"
        assert duration_points[0].attributes["gen_ai.provider.name"] == (
            "custom-provider"
        )
        assert duration_points[0].attributes["gen_ai.request.model"] == "model"
        assert duration_points[0].attributes["metric.attribute"] == "custom"

        token_points = metrics["gen_ai.client.token.usage"]
        assert len(token_points) == 2
        input_token_pt = next(
            p
            for p in token_points
            if p.attributes.get("gen_ai.token.type") == "input"
        )
        assert input_token_pt.sum == 10
        output_token_pt = next(
            p
            for p in token_points
            if p.attributes.get("gen_ai.token.type") == "output"
        )
        assert output_token_pt.sum == 20

    def test_content_capture_controls_span_and_event(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        # NO_CONTENT
        logger = MagicMock(spec=Logger)
        op = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
        )
        op.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="secret")])
        ]
        op.start("chat")
        op.finish(
            duration_s=0.5,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
        )
        (recorded,) = self.get_finished_spans()
        assert "gen_ai.input.messages" not in recorded.attributes
        logger.emit.assert_called_once()
        (event_record,) = logger.emit.call_args[0]
        assert "gen_ai.input.messages" not in event_record.attributes

        # SPAN_ONLY
        self.memory_exporter.clear()
        logger.reset_mock()
        op2 = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
        )
        op2.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="secret")])
        ]
        op2.start("chat")
        op2.finish(
            duration_s=0.5,
            content_capturing_mode=ContentCapturingMode.SPAN_ONLY,
        )
        (recorded2,) = self.get_finished_spans()
        assert "gen_ai.input.messages" in recorded2.attributes
        logger.emit.assert_called_once()
        (event_record2,) = logger.emit.call_args[0]
        assert "gen_ai.input.messages" not in event_record2.attributes

        # EVENT_ONLY
        self.memory_exporter.clear()
        logger.reset_mock()
        op3 = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
        )
        op3.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="secret")])
        ]
        op3.start("chat")
        op3.finish(
            duration_s=0.5,
            content_capturing_mode=ContentCapturingMode.EVENT_ONLY,
        )
        (recorded3,) = self.get_finished_spans()
        assert "gen_ai.input.messages" not in recorded3.attributes
        logger.emit.assert_called_once()
        (event_record3,) = logger.emit.call_args[0]
        assert "gen_ai.input.messages" in event_record3.attributes

    def test_content_capture_in_init(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        logger = MagicMock(spec=Logger)

        op = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
        )
        assert not op.should_capture_content
        op.input_messages = [
            InputMessage(role="user", parts=[TextPart(content="secret")])
        ]
        op.start("chat")
        op.finish(duration_s=0.5)

        (recorded,) = self.get_finished_spans()
        assert "gen_ai.input.messages" not in recorded.attributes
        logger.emit.assert_called_once()
        (event_record,) = logger.emit.call_args[0]
        assert "gen_ai.input.messages" not in event_record.attributes

    def test_emit_event_false(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        logger = MagicMock(spec=Logger)

        op = InferenceClientOperation(
            tracer,
            meter,
            logger,
            provider_name="custom-provider",
            operation_name=GenAiOperationName.CHAT,
        )
        op.start("chat")
        op.finish(duration_s=0.5, emit_event=False)

        logger.emit.assert_not_called()

    def test_context_manager_records_error_and_reraises(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        error = ValueError("bad request")

        op = RetrievalClientOperation(tracer, meter)
        op.start("retrieval")
        with self.assertRaises(ValueError) as raised:
            with op:
                raise error

        assert raised.exception is error
        (recorded,) = self.get_finished_spans()
        assert recorded.status.status_code is StatusCode.ERROR
        assert recorded.status.description == "bad request"
        assert recorded.attributes is not None
        assert recorded.attributes["error.type"] == "ValueError"
        assert len(recorded.events) == 0

    def test_operation_error_type_resolver(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        op = RetrievalClientOperation(
            tracer, meter, error_type_resolver=lambda exc: "429"
        )
        op.start("retrieval")
        op.finish(error=ValueError("rate limit"))

        (recorded,) = self.get_finished_spans()
        assert recorded.status.status_code is StatusCode.ERROR
        assert recorded.status.description == "rate limit"
        assert recorded.attributes is not None
        assert recorded.attributes["error.type"] == "429"
        assert len(recorded.events) == 0

    def test_operation_finish_with_error_object(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        op = RetrievalClientOperation(tracer, meter)
        op.start("retrieval")
        op.finish(error=Error(message="custom msg", type="custom_type"))

        (recorded,) = self.get_finished_spans()
        assert recorded.status.status_code is StatusCode.ERROR
        assert recorded.status.description == "custom msg"
        assert recorded.attributes is not None
        assert recorded.attributes["error.type"] == "custom_type"
        assert len(recorded.events) == 0

    def test_operation_finish_with_error_object_and_exception(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        exc = ValueError("orig")

        op = RetrievalClientOperation(tracer, meter)
        op.start("retrieval")
        op.finish(
            error=Error(
                message="custom msg", type="custom_type", exception=exc
            )
        )

        (recorded,) = self.get_finished_spans()
        assert recorded.status.status_code is StatusCode.ERROR
        assert recorded.status.description == "custom msg"
        assert recorded.attributes is not None
        assert recorded.attributes["error.type"] == "custom_type"
        assert len(recorded.events) == 0

    def test_context_manager_makes_span_current(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        op = RetrievalClientOperation(tracer, meter)
        op.start("retrieval")
        with op:
            with tracer.start_as_current_span("child"):
                pass

        child, retrieval = self.get_finished_spans()
        assert child.parent is not None
        assert retrieval.context is not None
        assert child.parent.span_id == retrieval.context.span_id

    def test_attribute_only_namespaces(self) -> None:
        assert AWS_BEDROCK_GUARDRAIL_ID == "aws.bedrock.guardrail.id"
        assert OpenAIApiType.RESPONSES == "responses"

    def test_span_name_templates(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        inference_op = InferenceClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.CHAT,
            request_model="gpt-4",
        )
        assert inference_op.span_name == "chat gpt-4"

        inference_op._request_model = None
        assert inference_op.span_name == "chat"

        inference_op._request_model = ""
        assert inference_op.span_name == "chat"

        inference_op._request_model = "_OTHER"
        assert inference_op.span_name == "chat"

        inference_op._operation_name = None
        assert inference_op.span_name == "gen_ai.inference.client"

        retrieval_op = RetrievalClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.RETRIEVAL,
            data_source_id="my-source",
        )
        assert retrieval_op.span_name == "retrieval my-source"
        retrieval_op.data_source_id = None
        assert retrieval_op.span_name == "retrieval"

        fetch_op = FetchResponseClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.FETCH_RESPONSE,
        )
        assert fetch_op.span_name == "fetch_response"

        tool_op = ExecuteToolInternalOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.EXECUTE_TOOL,
            tool_name="calculator",
        )
        assert tool_op.span_name == "execute_tool calculator"

        agent_op = InvokeAgentInternalOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.INVOKE_AGENT,
            agent_name="support-agent",
        )
        assert agent_op.span_name == "invoke_agent support-agent"

        workflow_op = InvokeWorkflowInternalOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.INVOKE_WORKFLOW,
            workflow_name="main-flow",
        )
        assert workflow_op.span_name == "invoke_workflow main-flow"

    def test_span_start_default_name(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)

        op1 = InferenceClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.CHAT,
            request_model="gpt-4o",
        )
        span1 = op1.start()
        assert span1.name == "chat gpt-4o"
        op1.finish()

        op2 = InferenceClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.CHAT,
            request_model="gpt-4o",
        )
        span2 = op2.start("custom-span-name")
        assert span2.name == "custom-span-name"
        op2.finish()

    def test_completion_hook_on_operation(self) -> None:
        tracer = self.tracer_provider.get_tracer(__name__)
        meter = self.meter_provider.get_meter(__name__)
        hook = MagicMock(spec=CompletionHook)

        op = InferenceClientOperation(
            tracer,
            meter,
            completion_hook=hook,
            operation_name=GenAiOperationName.CHAT,
            request_model="gpt-4o",
            input_messages=[
                InputMessage(role="user", parts=[TextPart(content="hello")])
            ],
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
        )
        assert op.should_capture_content is True
        op.start()
        op.finish()

        hook.on_completion.assert_called_once()
        kwargs = hook.on_completion.call_args.kwargs
        assert len(kwargs["inputs"]) == 1
        assert kwargs["span"] is not None

        embeddings_op = EmbeddingsClientOperation(
            tracer,
            meter,
            operation_name=GenAiOperationName.EMBEDDINGS,
        )
        assert not hasattr(embeddings_op, "completion_hook")
