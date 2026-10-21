# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

# NOTE: OpenTelemetry Python Logs API is in beta
from opentelemetry import _logs, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LogRecordProcessor,
    ReadWriteLogRecord,
)
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class DropGenAiEventsProcessor(LogRecordProcessor):
    """LogRecordProcessor wrapper that suppresses GenAI inference operation details events.

    This processor filters out the ``gen_ai.client.inference.operation.details`` event:
    1. Early drop: ``enabled()`` returns ``False`` when checking the event name,
       allowing the SDK and instrumentation to avoid constructing the event payload.
    2. Fallback: ``on_emit()`` drops the record if it is emitted.
    """

    def __init__(self, processor: LogRecordProcessor) -> None:
        self._processor = processor

    def enabled(self, *, event_name: str | None = None, **kwargs: Any) -> bool:
        if event_name == "gen_ai.client.inference.operation.details":
            return False
        return self._processor.enabled(event_name=event_name, **kwargs)

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        if not self.enabled(event_name=log_record.log_record.event_name):
            return
        self._processor.on_emit(log_record)

    def shutdown(self) -> None:
        self._processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._processor.force_flush(timeout_millis)


# Configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# Configure logging with the custom DropGenAiEventsProcessor
logger_provider = LoggerProvider()
base_log_processor = BatchLogRecordProcessor(OTLPLogExporter())
logger_provider.add_log_record_processor(
    DropGenAiEventsProcessor(base_log_processor)
)
_logs.set_logger_provider(logger_provider)

# Instrument OpenAI
OpenAIInstrumentor().instrument()


def main() -> None:
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
