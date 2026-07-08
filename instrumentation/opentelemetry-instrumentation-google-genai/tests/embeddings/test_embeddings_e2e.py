# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

import google.genai
import pytest
import yaml
from google.genai import types

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import StatusCode

# Disable mTLS client certificates to prevent workstation-specific OpenSSL/cryptography dependencies
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

from ..common.otel_mocker import OTelMocker

# Real key needed to re-record VCR against real gemini API.
_FAKE_API_KEY = "FAKE_KEY"


class _PrettyPrintJSONBody:
    """Makes request and response body recordings more readable in cassettes."""

    @staticmethod
    def serialize(cassette_dict):
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.safe_load(cassette_string)


@pytest.fixture(name="fully_initialized_vcr", scope="module", autouse=True)
def setup_vcr(vcr):
    vcr.register_serializer("yaml", _PrettyPrintJSONBody)
    vcr.serializer = "yaml"
    return vcr


@pytest.fixture(name="setup_instrumentation", scope="module", autouse=True)
def fixture_setup_instrumentation():
    instrumentor = GoogleGenAiSdkInstrumentor()
    instrumentor.instrument()
    yield
    instrumentor.uninstrument()


@pytest.fixture(name="otel_mocker", autouse=True)
def fixture_otel_mocker():
    result = OTelMocker()
    result.install()
    yield result
    result.uninstall()


def _before_record_request(request):
    if request.method:
        request.method = request.method.upper()
    return request


@pytest.fixture(name="vcr_config", scope="module")
def fixture_vcr_config():
    return {
        "filter_query_parameters": ["key", "apiKey"],
        "filter_headers": ["x-goog-api-key", "authorization"],
        "before_record_request": _before_record_request,
    }


@pytest.fixture(name="client")
def fixture_client():
    client = google.genai.Client(
        vertexai=False,
        api_key=_FAKE_API_KEY,
        http_options=types.HttpOptions(
            timeout=10.0,
            headers={
                "accept-encoding": "identity",
                "connection": "close",
            },
        ),
    )
    yield client
    try:
        client.close()
    except Exception:
        pass


@pytest.mark.vcr
def test_embeddings_e2e(client, otel_mocker: OTelMocker):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents="hello world",
    )

    assert response is not None

    spans = otel_mocker.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "embeddings gemini-embedding-2"
    assert span.status.status_code == StatusCode.UNSET

    attrs = span.attributes
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "gemini"
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gemini-embedding-2"
    assert attrs[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 3072
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 2
