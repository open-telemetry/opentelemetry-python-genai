# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ambient ``gen_ai.agent.name`` propagation onto inference metrics."""

from __future__ import annotations

import asyncio

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import Error


class _SimpleStream:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)

    def __iter__(self) -> _SimpleStream:
        return self

    def __next__(self) -> str:
        if self._chunks:
            return self._chunks.pop(0)
        raise StopIteration

    def close(self) -> None:
        pass


class _TestStreamWrapper(SyncStreamWrapper[str]):
    def _process_chunk(self, chunk: str) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_invocation.input_tokens = 10
        self._self_invocation.output_tokens = 20
        self._self_invocation.stop()

    def _on_stream_error(self, error: Error) -> None:
        self._self_invocation.fail(error)


class _SimpleAsyncStream:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _SimpleAsyncStream:
        return self

    async def __anext__(self) -> str:
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        pass


class _TestAsyncStreamWrapper(AsyncStreamWrapper[str]):
    def _process_chunk(self, chunk: str) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_invocation.input_tokens = 10
        self._self_invocation.output_tokens = 20
        self._self_invocation.stop()

    def _on_stream_error(self, error: Error) -> None:
        self._self_invocation.fail(error)


class TestAgentNameInferenceMetrics(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
            logger_provider=logger_provider,
        )

    def tearDown(self) -> None:
        self.log_exporter.clear()
        super().tearDown()

    def _harvest_metrics(self) -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {}
        for metric in self.get_sorted_metrics() or []:
            points = result.setdefault(metric.name, [])
            for point in metric.data.data_points or []:
                points.append(dict(point.attributes or {}))
        return result

    def test_inference_inside_local_agent_records_agent_name_on_metrics_only(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="triage"):
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ) as inference:
                inference.input_tokens = 12
                inference.output_tokens = 8

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            self.assertIn(metric_name, metrics)
            for attrs in metrics[metric_name]:
                self.assertEqual(attrs.get(GenAI.GEN_AI_AGENT_NAME), "triage")

        chat_spans = [
            s
            for s in self.get_finished_spans()
            if s.name == "chat gpt-4o-mini"
        ]
        self.assertEqual(len(chat_spans), 1)
        self.assertNotIn(
            GenAI.GEN_AI_AGENT_NAME, chat_spans[0].attributes or {}
        )

        logs = self.log_exporter.get_finished_logs()
        details_events = [
            log.log_record
            for log in logs
            if log.log_record.event_name
            == "gen_ai.client.inference.operation.details"
        ]
        self.assertEqual(len(details_events), 1)
        self.assertNotIn(
            GenAI.GEN_AI_AGENT_NAME, details_events[0].attributes or {}
        )

    def test_failed_inference_inside_local_agent_records_agent_name_on_metrics_only(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="triage"):
            invocation = self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            )
            invocation.fail(RuntimeError("boom"))

        metrics = self._harvest_metrics()
        points = metrics["gen_ai.client.operation.duration"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].get(GenAI.GEN_AI_AGENT_NAME), "triage")

        logs = self.log_exporter.get_finished_logs()
        events_by_name = {
            log.log_record.event_name: log.log_record for log in logs
        }
        for event_name in (
            "gen_ai.client.inference.operation.details",
            "gen_ai.client.inference.exception",
        ):
            self.assertIn(event_name, events_by_name)
            self.assertNotIn(
                GenAI.GEN_AI_AGENT_NAME,
                events_by_name[event_name].attributes or {},
            )

    def test_standalone_inference_omits_agent_name(self) -> None:
        with self.handler.inference(
            provider="openai", request_model="gpt-4o-mini"
        ) as inference:
            inference.input_tokens = 5
            inference.output_tokens = 3

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            for attrs in metrics[metric_name]:
                self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, attrs)

    def test_unattached_local_agent_with_activate(self) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="triage", _attach_to_context=False
        )
        try:
            with agent.activate():
                with self.handler.inference(
                    provider="openai", request_model="model-inside"
                ):
                    pass
            with self.handler.inference(
                provider="openai", request_model="model-outside"
            ):
                pass
        finally:
            agent.stop()

        metrics = self._harvest_metrics()
        by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in metrics["gen_ai.client.operation.duration"]
        }
        self.assertEqual(
            by_model,
            {"model-inside": "triage", "model-outside": None},
        )

    def test_unattached_local_agent_as_context_manager(self) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="triage", _attach_to_context=False
        )
        with agent:
            with self.handler.inference(
                provider="openai", request_model="model-inside"
            ):
                pass
        with self.handler.inference(
            provider="openai", request_model="model-after"
        ):
            pass

        metrics = self._harvest_metrics()
        by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in metrics["gen_ai.client.operation.duration"]
        }
        self.assertEqual(
            by_model,
            {"model-inside": "triage", "model-after": None},
        )

    def test_unattached_local_agent_with_explicit_context(self) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="triage", _attach_to_context=False
        )
        try:
            with self.handler.inference(
                provider="openai",
                request_model="gpt-4o-mini",
                context=agent.context,
            ):
                pass
        finally:
            agent.stop()

        metrics = self._harvest_metrics()
        points = metrics["gen_ai.client.operation.duration"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].get(GenAI.GEN_AI_AGENT_NAME), "triage")

    def test_unattached_inference_captures_agent_name_at_creation_time(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="triage", _attach_to_context=False
        )
        inference = self.handler.inference(
            provider="openai",
            request_model="gpt-4o-mini",
            context=agent.context,
            _attach_to_context=False,
        )
        agent.stop()
        inference.input_tokens = 12
        inference.output_tokens = 8
        inference.stop()

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            self.assertIn(metric_name, metrics)
            for attrs in metrics[metric_name]:
                self.assertEqual(attrs.get(GenAI.GEN_AI_AGENT_NAME), "triage")

    def test_unattached_local_agent_without_activation_omits_agent_name(
        self,
    ) -> None:
        agent = self.handler.invoke_local_agent(
            agent_name="triage", _attach_to_context=False
        )
        try:
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ):
                pass
        finally:
            agent.stop()

        metrics = self._harvest_metrics()
        points = metrics["gen_ai.client.operation.duration"]
        self.assertEqual(len(points), 1)
        self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, points[0])

    def test_nested_local_agents_attribute_each_inference_to_active_agent(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="outer"):
            with self.handler.inference(
                provider="openai", request_model="model-1"
            ):
                pass
            with self.handler.invoke_local_agent(agent_name="inner"):
                with self.handler.inference(
                    provider="openai", request_model="model-2"
                ):
                    pass
            with self.handler.inference(
                provider="openai", request_model="model-3"
            ):
                pass

        metrics = self._harvest_metrics()
        by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in metrics["gen_ai.client.operation.duration"]
        }
        self.assertEqual(
            by_model,
            {
                "model-1": "outer",
                "model-2": "inner",
                "model-3": "outer",
            },
        )

    def test_unnamed_or_empty_nested_local_agent_does_not_leak_outer_name(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="outer"):
            with self.handler.invoke_local_agent(agent_name=None):
                with self.handler.inference(
                    provider="openai", request_model="model-none"
                ):
                    pass
            with self.handler.invoke_local_agent(agent_name=""):
                with self.handler.inference(
                    provider="openai", request_model="model-empty"
                ):
                    pass
            with self.handler.inference(
                provider="openai", request_model="model-outer"
            ):
                pass

        metrics = self._harvest_metrics()
        by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in metrics["gen_ai.client.operation.duration"]
        }
        self.assertEqual(
            by_model,
            {
                "model-none": None,
                "model-empty": None,
                "model-outer": "outer",
            },
        )

    def test_explicit_metric_attribute_wins_over_ambient_agent_name(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="ambient"):
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ) as inference:
                inference.input_tokens = 4
                inference.output_tokens = 2
                inference.metric_attributes[GenAI.GEN_AI_AGENT_NAME] = (
                    "override"
                )

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            self.assertIn(metric_name, metrics)
            for attrs in metrics[metric_name]:
                self.assertEqual(
                    attrs.get(GenAI.GEN_AI_AGENT_NAME), "override"
                )

    def test_outer_ambient_agent_name_wins_over_suppressed_inner_metric_attribute(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="outer"):
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ):
                with self.handler.inference(
                    provider="openai", request_model="gpt-4o-mini"
                ) as inner:
                    inner.input_tokens = 7
                    inner.metric_attributes[GenAI.GEN_AI_AGENT_NAME] = (
                        "inner-raw"
                    )

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
        ):
            self.assertIn(metric_name, metrics)
            self.assertEqual(len(metrics[metric_name]), 1)
            self.assertEqual(
                metrics[metric_name][0].get(GenAI.GEN_AI_AGENT_NAME), "outer"
            )

    def test_remote_agent_does_not_publish_or_inherit_agent_name(self) -> None:
        with self.handler.invoke_remote_agent(
            provider="openai",
            request_model="remote-standalone",
            agent_name="remote",
        ):
            with self.handler.inference(
                provider="openai", request_model="model-standalone"
            ):
                pass

        with self.handler.invoke_local_agent(agent_name="caller"):
            with self.handler.invoke_remote_agent(
                provider="openai",
                request_model="remote-unnamed",
                agent_name=None,
            ):
                with self.handler.inference(
                    provider="openai", request_model="model-under-unnamed"
                ):
                    pass
            with self.handler.invoke_remote_agent(
                provider="openai",
                request_model="remote-named",
                agent_name="remote",
            ):
                with self.handler.inference(
                    provider="openai", request_model="model-under-named"
                ):
                    pass

        unnamed_remote_spans = [
            s for s in self.get_finished_spans() if s.name == "invoke_agent"
        ]
        self.assertEqual(len(unnamed_remote_spans), 1)
        self.assertNotIn(
            GenAI.GEN_AI_AGENT_NAME, unnamed_remote_spans[0].attributes or {}
        )

        metrics = self._harvest_metrics()
        duration_points = metrics["gen_ai.client.operation.duration"]
        remote_points = [
            attrs
            for attrs in duration_points
            if attrs.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
        ]
        self.assertEqual(len(remote_points), 3)
        for attrs in remote_points:
            self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, attrs)

        inference_by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in duration_points
            if attrs.get(GenAI.GEN_AI_OPERATION_NAME)
            == GenAI.GenAiOperationNameValues.CHAT.value
        }
        self.assertEqual(
            inference_by_model,
            {
                "model-standalone": None,
                "model-under-unnamed": "caller",
                "model-under-named": "caller",
            },
        )

    def test_streaming_inference_retains_agent_name_after_agent_exits(
        self,
    ) -> None:
        with self.handler.invoke_local_agent(agent_name="streamer"):
            invocation = self.handler.inference(
                provider="openai",
                request_model="gpt-4o-mini",
                _attach_to_context=False,
            )
            stream = _TestStreamWrapper(
                _SimpleStream(["hello", " world"]), invocation
            )

        self.assertEqual(list(stream), ["hello", " world"])

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
            "gen_ai.client.operation.time_to_first_chunk",
            "gen_ai.client.operation.time_per_output_chunk",
        ):
            self.assertIn(metric_name, metrics)
            for attrs in metrics[metric_name]:
                self.assertEqual(
                    attrs.get(GenAI.GEN_AI_AGENT_NAME), "streamer", metric_name
                )

    def test_async_streaming_inference_retains_agent_name_after_agent_exits(
        self,
    ) -> None:
        async def _run() -> list[str]:
            with self.handler.invoke_local_agent(
                agent_name="async-streamer", _attach_to_context=False
            ):
                invocation = self.handler.inference(
                    provider="openai",
                    request_model="gpt-4o-mini",
                    _attach_to_context=False,
                )
                stream = _TestAsyncStreamWrapper(
                    _SimpleAsyncStream(["hello", " world"]), invocation
                )
            return [chunk async for chunk in stream]

        self.assertEqual(asyncio.run(_run()), ["hello", " world"])

        metrics = self._harvest_metrics()
        for metric_name in (
            "gen_ai.client.operation.duration",
            "gen_ai.client.token.usage",
            "gen_ai.client.operation.time_to_first_chunk",
            "gen_ai.client.operation.time_per_output_chunk",
        ):
            self.assertIn(metric_name, metrics)
            for attrs in metrics[metric_name]:
                self.assertEqual(
                    attrs.get(GenAI.GEN_AI_AGENT_NAME),
                    "async-streamer",
                    metric_name,
                )

    def test_concurrent_asyncio_tasks_isolate_agent_names(self) -> None:
        async def _run_agent(agent_name: str, model: str) -> None:
            agent = self.handler.invoke_local_agent(
                agent_name=agent_name, _attach_to_context=False
            )
            with agent:
                await asyncio.sleep(0)
                with self.handler.inference(
                    provider="openai",
                    request_model=model,
                    _attach_to_context=False,
                ):
                    await asyncio.sleep(0)

        async def _run() -> None:
            await asyncio.gather(
                _run_agent("agent-a", "model-a"),
                _run_agent("agent-b", "model-b"),
            )

        asyncio.run(_run())

        metrics = self._harvest_metrics()
        by_model = {
            attrs[GenAI.GEN_AI_REQUEST_MODEL]: attrs.get(
                GenAI.GEN_AI_AGENT_NAME
            )
            for attrs in metrics["gen_ai.client.operation.duration"]
        }
        self.assertEqual(
            by_model,
            {"model-a": "agent-a", "model-b": "agent-b"},
        )
