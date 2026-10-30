# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests configuration module."""

import logging
import os
from collections import Counter
from collections.abc import Iterator

import boto3
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.callbacks import CallbackManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
    _StabilityMode,
)
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import (
    scrub_response_headers,
    scrub_response_headers_overwrite,
)
from opentelemetry.util.genai.types import ContentCapturingMode

# Loading the ``vcr`` module also activates the ``vcr_cassette_name`` override,
# which strips the ``[parametrize]`` suffix so all ``content_capture`` matrix
# cells share one cassette per test (the HTTP request is identical across
# cells).
pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]

_DURATION_METRICS = {
    "execute_tool": "gen_ai.execute_tool.duration",
    "invoke_agent": "gen_ai.invoke_agent.duration",
    "invoke_workflow": "gen_ai.invoke_workflow.duration",
    "retrieval": gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
    "chat": gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
}


@pytest.fixture
def instrument_langchain(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    content_capture: ContentCapturingMode,
    caplog: pytest.LogCaptureFixture,
) -> Iterator[None]:
    with (
        caplog.at_level(
            logging.WARNING, logger="langchain_core.callbacks.manager"
        ),
        instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture=content_capture.name,
        ),
    ):
        (handler,) = [
            handler
            for handler in CallbackManager([]).handlers
            if isinstance(handler, OpenTelemetryLangChainCallbackHandler)
        ]
        yield

        # Pytest resets caplog.records between setup, call, and teardown.
        warnings = [
            record.getMessage()
            for phase in ("setup", "call", "teardown")
            for record in caplog.get_records(phase)
            if record.name == "langchain_core.callbacks.manager"
            and record.levelno >= logging.WARNING
        ]
        assert not warnings, warnings
        assert not handler._invocation_manager._invocations

    expected = Counter(
        _DURATION_METRICS[operation]
        for span in span_exporter.get_finished_spans()
        if span.attributes is not None
        and (
            operation := span.attributes.get(
                gen_ai_attributes.GEN_AI_OPERATION_NAME
            )
        )
        in _DURATION_METRICS
    )
    assert expected
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    actual = {
        metric.name: sum(point.count for point in metric.data.data_points)
        for resource in metrics_data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name in _DURATION_METRICS.values()
    }
    assert actual == expected


@pytest.fixture(scope="function", name="chat_openai_gpt_3_5_turbo_model")
def fixture_chat_openai_gpt_3_5_turbo_model():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_sequences=["\n", "Human:", "AI:"],
        seed=100,
    )
    yield llm


@pytest.fixture(scope="function", name="chat_openai_reasoning")
def fixture_chat_openai_reasoning():
    llm = ChatOpenAI(model="gpt-5-mini")
    yield llm


@pytest.fixture(scope="function", name="chat_openai_legacy_functions")
def fixture_chat_openai_legacy_functions():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    yield llm


@pytest.fixture(scope="function", name="us_amazon_nova_lite_v1_0")
def fixture_us_amazon_nova_lite_v1_0():
    llm_model_value = "us.amazon.nova-lite-v1:0"
    llm = ChatBedrock(
        model_id=llm_model_value,
        client=boto3.client(
            "bedrock-runtime",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2",
            aws_account_id="test_account",
        ),
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-west-2",
        provider="amazon",
        temperature=0.1,
        max_tokens=100,
    )
    yield llm


@pytest.fixture(scope="function", name="chat_anthropic_claude_sonnet")
def fixture_chat_anthropic_claude_sonnet():
    llm_model_value = "claude-sonnet-4-5"
    llm = ChatAnthropic(
        model=llm_model_value,
        api_key="test_key",
        temperature=0.1,
        max_tokens=1024,
    )
    yield llm


@pytest.fixture(scope="function", name="gemini")
def fixture_gemini():
    llm_model_value = "gemini-2.5-pro"
    llm = ChatGoogleGenerativeAI(
        model=llm_model_value, api_key="test_key", transport="rest"
    )
    yield llm


@pytest.fixture(scope="function", name="chat_openai_vision")
def fixture_chat_openai_image():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=100,
    )
    yield llm


@pytest.fixture(scope="function", name="chat_openai_responses_vision")
def fixture_chat_openai_responses_vision():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=100,
        use_responses_api=True,
        include_response_headers=True,
    )
    yield llm


@pytest.fixture(scope="function")
def start_instrumentation(
    tracer_provider,
    meter_provider,
    logger_provider,
):
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_openai_api_key"


@pytest.fixture(autouse=True)
def reset_semconv_stability(monkeypatch: pytest.MonkeyPatch):
    """Ensure the semconv stability singleton re-reads env vars for each test.

    _get_opentelemetry_stability_opt_in_mode does not call _initialize() itself,
    so we patch it to call _initialize() first, making it pick up any env var
    changes applied via monkeypatch.setenv within the test body.
    """
    original = _OpenTelemetrySemanticConventionStability._get_opentelemetry_stability_opt_in_mode

    @classmethod  # type: ignore[misc]
    def _reinitializing_get(cls, signal_type):
        cls._initialized = False
        cls._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING = {}
        cls._initialize()
        return cls._OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING.get(
            signal_type, _StabilityMode.DEFAULT
        )

    monkeypatch.setattr(
        _OpenTelemetrySemanticConventionStability,
        "_get_opentelemetry_stability_opt_in_mode",
        _reinitializing_get,
    )
    yield
    monkeypatch.setattr(
        _OpenTelemetrySemanticConventionStability,
        "_get_opentelemetry_stability_opt_in_mode",
        original,
    )


_overwrite_response_headers = scrub_response_headers_overwrite(
    {
        "openai-organization": "test_openai_org_id",
        "openai-project": "test_openai_project_id",
        "Set-Cookie": "test_set_cookie",
    }
)
# Dropped rather than overwritten: the overwrite helper adds absent headers to
# every response, and too many of them break replay of other providers.
_drop_response_headers = scrub_response_headers(
    ["anthropic-organization-id", "anthropic-workspace-id"]
)


def _scrub_response(response):
    return _drop_response_headers(_overwrite_response_headers(response))


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
            ("x-api-key", "test_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": _scrub_response,
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
        "ignore_hosts": ["169.254.169.254"],
    }
