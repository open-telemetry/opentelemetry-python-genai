# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for the AgentScope instrumentation package."""

from __future__ import annotations

import asyncio
import inspect
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]

# Set up DASHSCOPE_API_KEY before any dashscope modules are imported. The
# dashscope SDK reads environment variables at import time and caches them.
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_api_key"

# vcrpy's aiohttp stub still references a mixin removed by newer aiohttp
# releases. Patch it so replay works during marker setup.
try:
    import aiohttp.streams  # type: ignore[import-not-found]

    if not hasattr(aiohttp.streams, "AsyncStreamReaderMixin"):
        aiohttp.streams.AsyncStreamReaderMixin = object
except ImportError:
    pass

try:
    import aiohttp  # type: ignore[import-not-found]
    import vcr.stubs.aiohttp_stubs as aiohttp_stubs

    if (
        "stream_writer"
        in inspect.signature(aiohttp.ClientResponse.__init__).parameters
    ):

        class _CompatStreamWriter:
            output_size = 0

        class _CompatMockClientResponse(aiohttp_stubs.MockClientResponse):
            def __init__(self, method, url, request_info=None):
                aiohttp.ClientResponse.__init__(
                    self,
                    method=method,
                    url=url,
                    writer=None,
                    continue100=None,
                    timer=None,
                    request_info=request_info,
                    traces=None,
                    loop=asyncio.get_event_loop(),
                    session=None,
                    stream_writer=_CompatStreamWriter(),
                )

        aiohttp_stubs.MockClientResponse = _CompatMockClientResponse
except ImportError:
    pass

from opentelemetry.instrumentation.genai.agentscope import (  # noqa: E402
    AgentScopeInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import (  # noqa: E402
    instrument as _instrument,
)

_V2_TEST_FILE = "test_v2_instrumentation.py"
# Files that must also be collected under agentscope v2. test_conformance.py
# carries its own module-level skips for non-conformance envs.
_V2_EXTRA_TEST_FILES = frozenset({"test_conformance.py"})


def _agentscope_major() -> int:
    try:
        installed_version = version("agentscope")
    except PackageNotFoundError:
        return 1
    try:
        return int(installed_version.split(".", 1)[0])
    except ValueError:
        return 1


def pytest_configure(config: pytest.Config) -> None:
    config.option.asyncio_mode = "auto"
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    config.option.api_key = os.environ["DASHSCOPE_API_KEY"]


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    path = Path(str(collection_path))
    if not path.name.startswith("test_") or path.suffix != ".py":
        return None

    major = _agentscope_major()
    if major >= 2:
        if path.name in _V2_EXTRA_TEST_FILES:
            return None
        return path.name != _V2_TEST_FILE
    if path.name == _V2_TEST_FILE:
        return True
    return None


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (  # noqa: PLC0415
        scrub_response_headers,
    )

    return {
        "filter_headers": [
            ("authorization", "<redacted>"),
            ("api-key", "<redacted>"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(["set-cookie"]),
    }


@pytest.fixture
def dashscope_model(request):
    from agentscope.model import DashScopeChatModel  # noqa: PLC0415

    return DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )


@pytest.fixture
def instrument(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        AgentScopeInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        AgentScopeInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        AgentScopeInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content_and_events(
    tracer_provider, logger_provider, meter_provider
):
    with _instrument(
        AgentScopeInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_AND_EVENT",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
