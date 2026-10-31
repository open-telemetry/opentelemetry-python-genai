# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OllamaInstrumentor lifecycle."""

from __future__ import annotations

import ollama

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor


def test_instrumentation_dependencies():
    assert OllamaInstrumentor().instrumentation_dependencies() == (
        "ollama >= 0.4.0, < 1",
    )


def test_uninstrument_restores_clients_and_module_functions(
    tracer_provider, logger_provider, meter_provider
):
    originals = {
        "Client.chat": ollama.Client.chat,
        "Client.embed": ollama.Client.embed,
        "AsyncClient.chat": ollama.AsyncClient.chat,
        "AsyncClient.embed": ollama.AsyncClient.embed,
    }
    module_chat, module_embed = ollama.chat, ollama.embed
    instrumentor = OllamaInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        assert ollama.Client.chat is not originals["Client.chat"]
        assert ollama.AsyncClient.embed is not originals["AsyncClient.embed"]
        assert ollama.chat is not module_chat
        assert ollama.embed is not module_embed
    finally:
        instrumentor.uninstrument()

    assert ollama.Client.chat is originals["Client.chat"]
    assert ollama.Client.embed is originals["Client.embed"]
    assert ollama.AsyncClient.chat is originals["AsyncClient.chat"]
    assert ollama.AsyncClient.embed is originals["AsyncClient.embed"]
    assert ollama.chat is module_chat
    assert ollama.embed is module_embed
