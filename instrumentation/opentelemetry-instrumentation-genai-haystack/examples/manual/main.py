# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from opentelemetry import trace
from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# 1. Setup OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# 2. Instrument Haystack
# Set OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=True to capture message content
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "True"
HaystackInstrumentor().instrument()


# 3. Use Haystack
def main():
    generator = OpenAIChatGenerator(model="gpt-4o-mini")
    messages = [
        ChatMessage.from_user("Tell me a quick joke about observability.")
    ]
    response = generator.run(messages=messages)
    print("\nResponse:")
    print(response["replies"][0].text)


if __name__ == "__main__":
    main()
