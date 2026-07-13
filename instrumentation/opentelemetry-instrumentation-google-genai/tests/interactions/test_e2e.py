# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import yaml
from google.genai import Client
from google.genai.types import HttpOptions

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)

from ..common.otel_mocker import OTelMocker

pytestmark = pytest.mark.skipif(
    not _HAS_INTERACTIONS,
    reason="Interactions are not supported in this version of google-genai",
)

# Switch to a real key when running the VCR tests against the real API.
_FAKE_API_KEY = "GEMINI_API_KEY"


class _LiteralBlockScalar(str):
    """Formats the string as a literal block scalar, preserving whitespace and
    without interpreting escape characters"""


def _literal_block_scalar_presenter(dumper, data):
    """Represents a scalar string as a literal block, via '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


@pytest.fixture(
    name="internal_setup_yaml_pretty_formatting", scope="module", autouse=True
)
def fixture_setup_yaml_pretty_formatting():
    yaml.add_representer(_LiteralBlockScalar, _literal_block_scalar_presenter)


def _process_string_value(string_value):
    """Pretty-prints JSON or returns long strings as a LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return _LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return _LiteralBlockScalar(string_value)
    return string_value


def _convert_body_to_literal(data):
    """Searches the data for body strings, attempting to pretty-print JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                string_val = value["string"]
                if isinstance(string_val, bytes):
                    try:
                        string_val = string_val.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                if isinstance(string_val, str):
                    value["string"] = _process_string_value(string_val)

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = _process_string_value(value)
            elif key == "body" and isinstance(value, bytes):
                try:
                    data[key] = _process_string_value(value.decode("utf-8"))
                except UnicodeDecodeError:
                    pass

            else:
                _convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = _convert_body_to_literal(choice)

    return data


class _PrettyPrintJSONBody:
    """This makes request and response body recordings more readable."""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(name="fully_initialized_vcr", scope="module", autouse=True)
def setup_vcr(vcr):
    vcr.register_serializer("yaml", _PrettyPrintJSONBody)
    vcr.serializer = "yaml"
    return vcr


@pytest.fixture(name="vcr_config", scope="module")
def fixture_vcr_config():
    return {
        "filter_query_parameters": [
            "key",
            "apiKey",
            "quotaUser",
            "userProject",
            "token",
            "access_token",
            "accessToken",
            "refesh_token",
            "refreshToken",
            "authuser",
            "bearer",
            "bearer_token",
            "bearerToken",
            "userIp",
        ],
        "filter_headers": [
            "x-goog-api-key",
            "authorization",
            "server",
            "Server",
            "Server-Timing",
            "Date",
        ],
        "ignore_hosts": [
            "oauth2.googleapis.com",
            "iam.googleapis.com",
        ],
        "decode_compressed_response": True,
    }


@pytest.fixture(name="instrumentor")
def fixture_instrumentor():
    return GoogleGenAiSdkInstrumentor()


@pytest.fixture(name="setup_instrumentation", autouse=True)
def fixture_setup_instrumentation(instrumentor):
    instrumentor.instrument()
    yield
    instrumentor.uninstrument()


@pytest.fixture(name="otel_mocker", autouse=True)
def fixture_otel_mocker():
    result = OTelMocker()
    result.install()
    yield result
    result.uninstall()


@pytest.fixture(name="client")
def fixture_client():
    return Client(
        api_key=_FAKE_API_KEY,
        vertexai=False,
        http_options=HttpOptions(headers={"accept-encoding": "identity"}),
    )


@pytest.mark.vcr
def test_sync_interactions_create(client, otel_mocker: OTelMocker):
    os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
        "SPAN_AND_EVENT"
    )

    response = client.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how can you help me today?",
    )

    assert response is not None
    assert response.id is not None

    span = otel_mocker.get_span_named("interactions.create gemini-2.5-flash")
    assert span is not None
    assert span.attributes["gen_ai.provider.name"] == "gemini"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.operation.name"] == "interactions.create"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_interactions_create(client, otel_mocker: OTelMocker):
    os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
        "SPAN_AND_EVENT"
    )

    response = await client.aio.interactions.create(
        model="gemini-2.5-flash",
        input="Hello, how can you help me today?",
    )

    assert response is not None
    assert response.id is not None

    span = otel_mocker.get_span_named("interactions.create gemini-2.5-flash")
    assert span is not None
    assert span.attributes["gen_ai.provider.name"] == "gemini"
    assert span.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.operation.name"] == "interactions.create"
