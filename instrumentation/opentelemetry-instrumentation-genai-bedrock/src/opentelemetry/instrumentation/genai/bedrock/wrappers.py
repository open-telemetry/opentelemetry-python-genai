# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
#
# Based on the Bedrock extension in opentelemetry-python-contrib by @xrmx:
# https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3161
# https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3258

"""Stream wrappers for AWS Bedrock ConverseStream instrumentation."""

from __future__ import annotations

from typing import Any

from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
)

from .extractors import normalize_finish_reason


class ConverseStreamWrapper:
    """Wrapper around a Bedrock ConverseStream response that accumulates
    chunks and finalizes the invocation when the stream completes.

    The Bedrock ConverseStream response has a ``stream`` key containing an
    EventStream iterator. This wrapper proxies the original response dict
    but replaces the ``stream`` value with an instrumented iterator.
    """

    def __init__(
        self,
        response: dict[str, Any],
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        self._response = response
        self._invocation = invocation
        self._capture_content = capture_content
        self._finalized = False

        # Wrap the event stream iterator
        original_stream = response.get("stream")
        if original_stream is not None:
            response["stream"] = _StreamEventIterator(
                original_stream,
                self._on_complete,
                self._on_fail,
                capture_content,
            )

    def __getitem__(self, key: str) -> Any:
        return self._response[key]

    def __contains__(self, key: str) -> bool:
        return key in self._response

    def get(self, key: str, default: Any = None) -> Any:
        return self._response.get(key, default)

    def _on_complete(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        finish_reason: str | None,
        parts: list[MessagePart],
    ) -> None:
        if self._finalized:
            return
        self._finalized = True

        if input_tokens is not None:
            self._invocation.input_tokens = input_tokens
        if output_tokens is not None:
            self._invocation.output_tokens = output_tokens

        normalized = normalize_finish_reason(finish_reason)
        if normalized:
            self._invocation.finish_reasons = [normalized]

        if self._capture_content and parts:
            self._invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=normalized or "",
                )
            ]

        self._invocation.stop()

    def _on_fail(self, exc: BaseException) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._invocation.fail(exc)


class _StreamEventIterator:
    """Iterates over Bedrock ConverseStream events, accumulating content."""

    def __init__(
        self,
        event_stream: Any,
        on_complete: Any,
        on_fail: Any,
        capture_content: bool,
    ) -> None:
        self._event_stream = event_stream
        self._on_complete = on_complete
        self._on_fail = on_fail
        self._capture_content = capture_content

        # Accumulated state
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._finish_reason: str | None = None
        self._parts: list[MessagePart] = []
        self._current_text: str = ""
        self._current_tool_use: dict[str, Any] | None = None

    def __iter__(self) -> _StreamEventIterator:
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._event_stream)
        except StopIteration:
            self._finalize_current_block()
            self._on_complete(
                self._input_tokens,
                self._output_tokens,
                self._finish_reason,
                self._parts,
            )
            raise
        except Exception as exc:
            self._on_fail(exc)
            raise

        self._process_event(event)
        return event

    def _process_event(self, event: dict[str, Any]) -> None:
        """Process a single stream event and update accumulated state."""
        if "contentBlockStart" in event:
            self._handle_content_block_start(event["contentBlockStart"])
        elif "contentBlockDelta" in event:
            self._handle_content_block_delta(event["contentBlockDelta"])
        elif "contentBlockStop" in event:
            self._finalize_current_block()
        elif "messageStop" in event:
            stop = event["messageStop"]
            self._finish_reason = stop.get("stopReason")
        elif "metadata" in event:
            metadata = event["metadata"]
            usage = metadata.get("usage") or {}
            input_tokens = usage.get("inputTokens")
            output_tokens = usage.get("outputTokens")
            if input_tokens is not None:
                self._input_tokens = input_tokens
            if output_tokens is not None:
                self._output_tokens = output_tokens

    def _handle_content_block_start(self, block_start: dict[str, Any]) -> None:
        """Handle the start of a new content block."""
        self._finalize_current_block()
        start = block_start.get("start") or {}
        if "toolUse" in start:
            tool_use = start["toolUse"]
            self._current_tool_use = {
                "toolUseId": tool_use.get("toolUseId"),
                "name": tool_use.get("name", ""),
                "input_json": "",
            }
        else:
            # Text block (default)
            self._current_text = ""

    def _handle_content_block_delta(self, block_delta: dict[str, Any]) -> None:
        """Handle a content block delta."""
        delta = block_delta.get("delta") or {}
        if "text" in delta:
            if self._capture_content:
                self._current_text += delta["text"]
        elif "toolUse" in delta:
            if self._current_tool_use is not None and self._capture_content:
                tool_delta = delta["toolUse"]
                self._current_tool_use["input_json"] += tool_delta.get(
                    "input", ""
                )

    def _finalize_current_block(self) -> None:
        """Finalize the current content block and add it to parts."""
        if not self._capture_content:
            self._current_text = ""
            self._current_tool_use = None
            return

        if self._current_tool_use is not None:
            import json  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

            input_json = self._current_tool_use.get("input_json", "")
            arguments: Any = None
            if input_json:
                try:
                    arguments = json.loads(input_json)
                except ValueError:
                    arguments = input_json
            self._parts.append(
                ToolCallRequest(
                    arguments=arguments,
                    name=self._current_tool_use.get("name", ""),
                    id=self._current_tool_use.get("toolUseId"),
                )
            )
            self._current_tool_use = None
        elif self._current_text:
            self._parts.append(Text(content=self._current_text))
            self._current_text = ""
