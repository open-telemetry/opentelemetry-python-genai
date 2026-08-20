# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle, entry-point, completion-hook, and restoration tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import smolagents
from smolagents.tools import PipelineTool
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
    _model_classes_defining,
    _tool_classes_defining_call,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.genai.completion_hook import CompletionHook

from .test_utils import MESSAGES, transformers_model


class RecordingHook(CompletionHook):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _generate_a_chat_span() -> None:
    transformers_model().generate(messages=MESSAGES)


def test_entrypoint_loads_instrumentor() -> None:
    (entrypoint,) = entry_points(
        group="opentelemetry_instrumentor", name="smolagents"
    )
    instrumentor = entrypoint.load()()
    assert isinstance(instrumentor, SmolagentsInstrumentor)


def test_instrumentation_dependencies() -> None:
    dependencies = SmolagentsInstrumentor().instrumentation_dependencies()
    assert "smolagents >= 1.24.0" in dependencies


def test_instrument_uninstrument_restores_originals(
    tracer_provider, logger_provider, meter_provider
) -> None:
    original_generate = smolagents.TransformersModel.generate
    original_mlx_generate = smolagents.MLXModel.generate
    original_generate_stream = smolagents.TransformersModel.generate_stream
    original_run = smolagents.MultiStepAgent.run
    original_tool_call = smolagents.Tool.__call__
    original_pipeline_call = PipelineTool.__dict__["__call__"]
    original_process_tool_calls = smolagents.ToolCallingAgent.__dict__[
        "process_tool_calls"
    ]
    original_timeout = smolagents.local_python_executor.timeout

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    assert smolagents.TransformersModel.generate is not original_generate
    assert smolagents.MLXModel.generate is not original_mlx_generate
    assert (
        smolagents.TransformersModel.generate_stream
        is not original_generate_stream
    )
    assert smolagents.MultiStepAgent.run is not original_run
    assert PipelineTool.__dict__["__call__"] is not original_pipeline_call
    assert (
        smolagents.ToolCallingAgent.__dict__["process_tool_calls"]
        is not original_process_tool_calls
    )
    assert smolagents.local_python_executor.timeout is not original_timeout

    instrumentor.uninstrument()

    assert smolagents.TransformersModel.generate is original_generate
    assert smolagents.MLXModel.generate is original_mlx_generate
    assert (
        smolagents.TransformersModel.generate_stream
        is original_generate_stream
    )
    assert smolagents.MultiStepAgent.run is original_run
    assert smolagents.Tool.__call__ is original_tool_call
    assert PipelineTool.__dict__["__call__"] is original_pipeline_call
    assert (
        smolagents.ToolCallingAgent.__dict__["process_tool_calls"]
        is original_process_tool_calls
    )
    assert smolagents.local_python_executor.timeout is original_timeout


@pytest.mark.parametrize(
    "model_class",
    [
        "Model",
        "OpenAIModel",
        "AzureOpenAIModel",
        "AmazonBedrockModel",
        "InferenceClientModel",
        "LiteLLMModel",
        "LiteLLMRouterModel",
    ],
)
def test_api_backed_classes_are_left_alone(
    tracer_provider, logger_provider, meter_provider, model_class: str
) -> None:
    # Each of these calls a client library that carries its own instrumentation,
    # so patching them here would emit a second chat span for one model call and
    # count the token-usage and duration metrics twice.
    original = getattr(smolagents, model_class).__dict__.get("generate")

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        assert (
            getattr(smolagents, model_class).__dict__.get("generate")
            is original
        )
    finally:
        instrumentor.uninstrument()


def test_uninstrument_through_a_new_constructor_call(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor is a per-class singleton, so the documented
    # SmolagentsInstrumentor().instrument() / SmolagentsInstrumentor()
    # .uninstrument() form must restore everything even though the second
    # constructor call re-runs __init__ on the live instance.
    original_generate = smolagents.TransformersModel.generate
    original_run = smolagents.MultiStepAgent.run
    original_timeout = smolagents.local_python_executor.timeout

    SmolagentsInstrumentor().instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    SmolagentsInstrumentor().uninstrument()

    assert smolagents.TransformersModel.generate is original_generate
    assert smolagents.MultiStepAgent.run is original_run
    assert smolagents.local_python_executor.timeout is original_timeout


@pytest.mark.parametrize("method", ["generate", "generate_stream"])
def test_model_classes_defining(method: str) -> None:
    classes = _model_classes_defining(method)

    assert len(classes) == len(set(classes))
    # Every entry owns the method, so no class is wrapped for one it only
    # inherits.
    for model_cls in classes:
        assert method in model_cls.__dict__


def test_only_transformers_defines_generate_stream() -> None:
    # All three in-process runtimes define generate, but only TransformersModel
    # streams.
    assert set(_model_classes_defining("generate")) == {
        smolagents.TransformersModel,
        smolagents.VLLMModel,
        smolagents.MLXModel,
    }
    assert _model_classes_defining("generate_stream") == [
        smolagents.TransformersModel
    ]


def test_tool_classes_defining_call() -> None:
    classes = _tool_classes_defining_call()

    assert len(classes) == len(set(classes))
    for tool_cls in classes:
        assert "__call__" in tool_cls.__dict__
    # PipelineTool overrides Tool.__call__ and calls forward directly instead of
    # delegating, so both classes need patching. smolagents doesn't export
    # PipelineTool, so it is reached through the MRO of SpeechToTextTool.
    assert set(classes) == {smolagents.Tool, PipelineTool}
    # A tool that only inherits __call__ is covered by the class it inherits it
    # from, and must not be wrapped again.
    assert smolagents.FinalAnswerTool not in classes
    assert smolagents.SpeechToTextTool not in classes


def test_repeated_instrument_uninstrument(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # BaseInstrumentor returns a per-class singleton, so the wrapped-class
    # bookkeeping has to survive being filled and drained more than once.
    original_generate = smolagents.TransformersModel.generate
    original_run = smolagents.MultiStepAgent.run
    original_tool_call = smolagents.Tool.__call__

    instrumentor = SmolagentsInstrumentor()
    for _ in range(2):
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        assert smolagents.TransformersModel.generate is not original_generate
        assert smolagents.MultiStepAgent.run is not original_run
        assert smolagents.Tool.__call__ is not original_tool_call
        instrumentor.uninstrument()
        assert smolagents.TransformersModel.generate is original_generate
        assert smolagents.MultiStepAgent.run is original_run
        assert smolagents.Tool.__call__ is original_tool_call


def test_uninstrument_without_instrument() -> None:
    # BaseInstrumentor.uninstrument() short-circuits, but _uninstrument() must
    # also be a no-op on unpatched attributes: the rollback in _instrument()
    # calls it after a partial patch.
    original_generate = smolagents.TransformersModel.generate
    original_run = smolagents.MultiStepAgent.run
    original_tool_call = smolagents.Tool.__call__
    original_timeout = smolagents.local_python_executor.timeout

    SmolagentsInstrumentor().uninstrument()
    SmolagentsInstrumentor()._uninstrument()

    assert smolagents.TransformersModel.generate is original_generate
    assert smolagents.MultiStepAgent.run is original_run
    assert smolagents.Tool.__call__ is original_tool_call
    assert smolagents.local_python_executor.timeout is original_timeout


def test_instrument_with_no_providers() -> None:
    # Without providers the handler falls back to the globals; instrumenting
    # must not require a caller to pass them.
    original_generate = smolagents.TransformersModel.generate
    original_run = smolagents.MultiStepAgent.run

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument()
    try:
        assert smolagents.TransformersModel.generate is not original_generate
        assert smolagents.MultiStepAgent.run is not original_run
    finally:
        instrumentor.uninstrument()

    assert smolagents.TransformersModel.generate is original_generate
    assert smolagents.MultiStepAgent.run is original_run


def _originals() -> dict[tuple[type, str], Any]:
    """Every attribute ``_instrument`` patches, before it is patched."""
    patched: dict[tuple[type, str], Any] = {
        (model_cls, "generate"): model_cls.__dict__["generate"]
        for model_cls in _model_classes_defining("generate")
    }
    patched.update(
        {
            (model_cls, "generate_stream"): model_cls.__dict__[
                "generate_stream"
            ]
            for model_cls in _model_classes_defining("generate_stream")
        }
    )
    patched.update(
        {
            (tool_cls, "__call__"): tool_cls.__dict__["__call__"]
            for tool_cls in _tool_classes_defining_call()
        }
    )
    patched[(smolagents.MultiStepAgent, "run")] = (
        smolagents.MultiStepAgent.__dict__["run"]
    )
    patched[(smolagents.ToolCallingAgent, "process_tool_calls")] = (
        smolagents.ToolCallingAgent.__dict__["process_tool_calls"]
    )
    return patched


@pytest.mark.parametrize("fail_on_call", [2, None])
def test_failed_instrument_rolls_back_partial_patches(
    tracer_provider, logger_provider, meter_provider, fail_on_call: int | None
) -> None:
    # A failure part-way through must leave nothing patched, because
    # uninstrument() cannot clean up after a failed _instrument(). fail_on_call=2
    # fails after the first patch, None fails on the last one (timeout), so the
    # rollback is exercised from both ends.
    originals = _originals()
    assert len(originals) > 1, "the rollback needs more than one patch"
    original_timeout = smolagents.local_python_executor.timeout

    real_wrap = wrap_function_wrapper
    calls = 0

    def failing_wrap(target: Any, name: str, wrapper: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_on_call or (
            fail_on_call is None and name == "timeout"
        ):
            raise RuntimeError("boom")
        real_wrap(target, name, wrapper)

    with patch(
        "opentelemetry.instrumentation.genai.smolagents.wrap_function_wrapper",
        failing_wrap,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            SmolagentsInstrumentor().instrument(
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
            )

    assert calls > 1, "at least one attribute was expected to be patched"
    for (target, name), original in originals.items():
        assert target.__dict__[name] is original, (
            f"{target.__name__}.{name} was not restored"
        )
    assert smolagents.local_python_executor.timeout is original_timeout


def test_a_user_subclass_inherits_the_patched_generate(
    tracer_provider, logger_provider, meter_provider
) -> None:
    # A subclass that doesn't override generate is instrumented through the base
    # it inherits it from, and must not be wrapped a second time.
    class TenantMLXModel(smolagents.MLXModel):
        pass

    assert "generate" not in TenantMLXModel.__dict__

    instrumentor = SmolagentsInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        # wrapt returns a fresh bound wrapper per attribute access, so the
        # wrapper objects differ; the underlying wrapped function is shared,
        # proving the subclass inherits the single wrapped generate.
        assert (
            TenantMLXModel.generate.__wrapped__
            is smolagents.MLXModel.generate.__wrapped__
        )
    finally:
        instrumentor.uninstrument()


def test_explicit_completion_hook_takes_precedence(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    explicit_hook = RecordingHook()
    with patch(
        "opentelemetry.instrumentation.genai.smolagents.load_completion_hook"
    ) as load_hook:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
            completion_hook=explicit_hook,
        ):
            _generate_a_chat_span()

    load_hook.assert_not_called()
    assert explicit_hook.calls, "explicit completion hook was not invoked"


def test_env_completion_hook_used_when_no_explicit_hook(
    tracer_provider, logger_provider, meter_provider
) -> None:
    env_hook = RecordingHook()
    with patch(
        "opentelemetry.instrumentation.genai.smolagents.load_completion_hook",
        return_value=env_hook,
    ) as load_hook:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            _generate_a_chat_span()

    load_hook.assert_called_once()
    assert env_hook.calls, "env-resolved completion hook was not invoked"
