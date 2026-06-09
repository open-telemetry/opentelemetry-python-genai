# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests configuration module."""

import os

import boto3
import pytest
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation._semconv import (
    _OpenTelemetrySemanticConventionStability,
    _StabilityMode,
)
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.test_util_genai.vcr import scrub_response_headers_overwrite

# Loading the ``vcr`` module also activates the ``vcr_cassette_name`` override,
# which strips the ``[parametrize]`` suffix so all ``content_capture`` matrix
# cells share one cassette per test (the HTTP request is identical across
# cells).
pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


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


@pytest.fixture(scope="function", name="gemini")
def fixture_gemini():
    llm_model_value = "gemini-2.5-pro"
    llm = ChatGoogleGenerativeAI(model=llm_model_value, api_key="test_key")
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


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "openai-organization": "test_openai_org_id",
                "openai-project": "test_openai_project_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
        "ignore_hosts": ["169.254.169.254"],
    }
