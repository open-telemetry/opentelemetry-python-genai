---
name: write-conformance-tests
description: Author GenAI conformance-test scenarios for an OpenTelemetry instrumentation package — Scenario subclasses under tests/conformance/, the test_conformance.py runner, declared gaps, lib-specific assertions, and weaver live-check policies. Use when adding or updating conformance tests for any instrumentation, whether greenfield or ported.
---

# Write GenAI conformance tests

Conformance tests validate that an instrumentation package emits telemetry
matching the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
via Weaver live-check. They apply to **any** instrumentation package —
greenfield or ported — and don't depend on how the package was built.

This skill covers authoring the `tests/conformance/<scenario>.py` modules and
the `tests/test_conformance.py` runner. For the always-on rules that hold even
without this skill loaded, see the **Conformance tests** section of
[AGENTS.md](../../../AGENTS.md).

## One scenario per operation

Put one scenario per emitted semconv operation under `tests/conformance/`.
Write a scenario for **every** semconv operation the library emits, even one
currently blocked by a util-genai or semconv gap. Skipping the scenario hides
the gap; writing it records the gap (see [Declared gaps](#declared-gaps)). 
**Never** drop a scenario file because it would fail today.

## Recommended scenarios

Cover the scenarios below that apply to the library. Skip a row only when the
library genuinely can't perform that operation (e.g. an inference-only
client has no `embeddings` scenario).

**LLM client instrumentations:**

| Scenario | File | Covers |
|---|---|---|
| Inference | `inference.py` | A `chat` operation. |
| Tool calling | `tool_calling.py` | A `chat` turn where the model returns tool calls and a follow-up turn feeds tool results back. Asserts tool calls and tool results are present on input/output **messages**. weaver will validate the format. Do **not** expect `execute_tool` spans unless the client library itself instruments tool execution — most don't; tool execution is the caller's code. |
| Multimodal content | `multimodal.py` | A `chat` turn carrying the **non-text parts** the client accepts (inline image/audio bytes, media URLs, file refs, …), asserting each round-trips onto the messages. Cover only the part types the library emits — see [Message-part coverage](#message-part-coverage). |
| Reasoning | `reasoning.py` | A `chat` turn against a reasoning model where the response carries reasoning/thinking content, asserting a `reasoning` part lands on an output message (and `gen_ai.usage.output_tokens` / reasoning-token attributes if the library records them). Only when the client surfaces reasoning content — see [Message-part coverage](#message-part-coverage). |
| Embeddings | `embedding.py` | An `embeddings` operation. |

**Agent / orchestration instrumentations:**

| Scenario | File | Covers |
|---|---|---|
| Agent invocation with tooling | `invoke_agent.py` | An `invoke_agent` run that calls at least one tool — expects `invoke_agent` plus the nested `execute_tool` / `chat` spans the framework emits. |
| Multi-agent orchestration | `multi_agent.py` | One agent handing off to / invoking another — expects nested `invoke_agent` spans under the orchestrator. |
| Workflows | `invoke_workflow.py` | An `invoke_workflow` run wrapping the agent/tool spans it drives. |

## Message-part coverage

Weaver validates a part's *shape*, not *which* part types a scenario
exercised — a text-only scenario leaves the package's image/audio/file/tool
mapping unverified. So exercise **every non-text part type the library can
emit** and assert it landed on a message. Cover only what the package
instruments: walk its wrappers (the step-6 mapping for a port) for which
`opentelemetry.util.genai.types` parts they produce.

| Part `type` | util-genai type | Emitted when the library accepts… |
|---|---|---|
| `text` | `Text` | plain text (always) |
| `tool_call` / `tool_call_response` | `ToolCallRequest` / `ToolCallResponse` | function/tool calling — covered by `tool_calling.py` |
| `server_tool_call` / `server_tool_call_response` | `ServerToolCall` / `ServerToolCallResponse` | vendor server-side tools (web_search, code_interpreter, …) |
| `reasoning` | `Reasoning` | reasoning / thinking items |
| `blob` | `Blob` | inline image/audio/video **bytes** (`modality` distinguishes them) |
| `uri` | `Uri` | an external media **URL** (`modality`) |
| `file` | `File` | a **file reference** / id (`modality`) |
| `generic` | `GenericPart` | a provider item with no semconv mapping — flag, don't drop |

Group by shared turn/cassette — typically one `multimodal.py` for the
image/audio/file/url inputs the client accepts, `tool_calling.py` for tool
parts, and `reasoning.py` for `reasoning` parts (a reasoning model emits
those on output messages, not input). `type` alone gives `blob` / `uri` /
`file`; to tell image from
audio from video, read the part's `modality` with a `_part_fields` helper
returning `(type, modality)` tuples (defined alongside `_part_types` in
[Lib-specific assertions](#lib-specific-assertions)):

```python
    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        chat_spans = [
            entry["span"] for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        input_parts = {
            (t, m) for span in chat_spans
            for t, m in _part_fields(_attr(span, "gen_ai.input.messages"))
        }
        # e.g. an inline image + an audio URL were sent
        assert ("blob", "image") in input_parts, f"no image blob, saw {input_parts}"
        assert ("uri", "audio") in input_parts, f"no audio uri, saw {input_parts}"
```

If a part type the library accepts can't round-trip yet (a util-genai/semconv
gap), still write the scenario and record it as a
[declared gap](#declared-gaps) — never silently omit the part.

## Scenario modules

Each scenario module defines a subclass of `Scenario` from
`opentelemetry.test_util_genai.conformance`. It sets the `expected_spans` /
`expected_metrics` ClassVars and implements
`run(self, *, tracer_provider, meter_provider, logger_provider, vcr)`.
Drive instrumentation through the shared `instrument` context manager (not
`instr.instrument()` / `trace.set_tracer_provider`). The runner injects an
already-configured `vcr`, so a cassette-based scenario just calls
`vcr.use_cassette(...)`:

```python
# tests/conformance/inference.py
from typing import Any

from opentelemetry.instrumentation.genai.<lib> import <Lib>Instrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = ("chat",)
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            <Lib>Instrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("inference.yaml"):
                ...  # call the patched API
```

One operation per scenario. No env vars, no logging config.

**VCR cassettes are not required — a transport mock works too.** Mock HTTP
however the package's **unit** tests already do, and use the **same pattern
across every scenario in that package** (don't mix cassettes and transport
mocks within one lib). If the package mocks the transport (e.g.
`httpx.MockTransport`, `respx`) instead of replaying cassettes, build the
client with that transport inside `run()` and ignore the injected `vcr`:

```python
    def run(self, *, tracer_provider, meter_provider, logger_provider, vcr) -> None:
        with instrument(
            <Lib>Instrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            client = <Lib>(transport=httpx.MockTransport(_handler))  # canned response
            client.<method>(...)  # call the patched API
```

`vcr` stays in the signature either way (the runner always passes it).

## Declared gaps

**Declared gaps** go in the `expected_violations` ClassVar (a tuple of
`ExpectedViolation`), not `xfail`. `run_conformance` fails on *undeclared*
weaver violations and on declared violations weaver no longer reports — so
a known util-genai/semconv gap is recorded as an `expected_violation` that
fails loudly the moment it's fixed.

When the gap is too big to capture as individual `expected_violations` — the
whole operation can't run yet — skip the entire scenario instead, via
`pytest.mark.xfail` / `skip` on the parametrize entry in
`test_conformance.py`. Don't invent a one-off `reason=` string: **ask the
user to file a tracking bug** and update the skip `reason=` with that issue
(e.g. `reason="blocked by open-telemetry/...#1234"`) so the skip is traceable
and gets revisited when the bug is fixed. **Never** drop the scenario file
itself — a skipped scenario still records that the operation exists; a
deleted one hides it.

## Lib-specific assertions

**Lib-specific assertions** go in a `validate(self, report)` override on the
scenario (call `super().validate(report)` first) — there is no
`_local_assertions.py` / `LocalAssertions` hook. Common lib-specific shapes:

- **Vendor server-tool payloads** (`code_interpreter`, `web_search`, …).
- **Vendor-specific finish reasons** outside semconv's enum (`stop`,
  `length`, `content_filter`, `tool_call`, `error`).
- **Provider-specific `error.type`** — exception class names from the
  underlying SDK.

`validate` receives the weaver `LiveCheckReport`. Read span samples from
`report["samples"]` (each `entry["span"]["attributes"]` is a list of
`{"name", "value"}`) and seen metrics from
`report["statistics"]["seen_registry_metrics"]`. Always call
`super().validate(report)` first so the `expected_spans` / `expected_metrics`
checks still run:

```python
# tests/conformance/tool_calling.py
from __future__ import annotations

import json
from typing import Any

from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario


class ToolCallingScenario(Scenario):
    expected_spans = ("chat",)
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(self, *, tracer_provider, meter_provider, logger_provider, vcr) -> None:
        ...  # drive a tool-calling turn — see "Scenario modules" above

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)  # keep the expected_spans / _metrics checks

        # Lib-specific: weaver validates the *shape* of each message part, but
        # not that a tool call actually round-tripped. Assert the model's
        # tool_call landed on an output message and the tool result was fed
        # back on an input message (across the two chat turns).
        chat_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert chat_spans, "no chat span emitted"

        output_part_types = {
            t for span in chat_spans
            for t in _part_types(_attr(span, "gen_ai.output.messages"))
        }
        input_part_types = {
            t for span in chat_spans
            for t in _part_types(_attr(span, "gen_ai.input.messages"))
        }
        assert "tool_call" in output_part_types, (
            f"expected a tool_call part on an output message, saw {output_part_types}"
        )
        assert "tool_call_response" in input_part_types, (
            f"expected a tool_call_response part on an input message, saw {input_part_types}"
        )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    # gen_ai.{input,output}.messages is a JSON string of
    # [{"role": ..., "parts": [{"type": ..., ...}]}].
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]


def _part_fields(messages_json: str | None) -> list[tuple[str, str | None]]:
    # Like _part_types, but keeps modality so image/audio/video are
    # distinguishable on blob/uri/file parts (None for parts without one).
    messages = json.loads(messages_json) if messages_json else []
    return [
        (part["type"], part.get("modality"))
        for message in messages
        for part in message["parts"]
    ]
```

## The test_conformance.py runner

`tests/test_conformance.py` guards collection with
`pytest.importorskip("opentelemetry.test.weaver_live_check")`, parametrizes
the scenario *instances*, and calls `run_conformance(scenario, vcr=vcr,
weaver=weaver_live_check)` — it builds its own providers from the weaver
OTLP endpoint, so don't pass providers/exporters:

```python
import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    Scenario,
    run_conformance,
)

from .conformance.embedding import EmbeddingScenario
from .conformance.inference import InferenceScenario


@pytest.mark.parametrize(
    "scenario",
    [InferenceScenario(), EmbeddingScenario()],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, vcr, weaver_live_check: WeaverLiveCheck
) -> None:
    run_conformance(scenario, vcr=vcr, weaver=weaver_live_check)
```

Do not write a separate `test_weaver_live.py`; weaver is already wired
through `run_conformance`. The `weaver_live_check` fixture skips the test
(rather than passing without the gate) only when it can't start (unsupported
platform / network) — never wrap it in a try/except, which would silently
disable the gate.

## Weaver policies

`weaver_live_check` enforces `policies/genai_content_validation.rego`
(content-attribute JSON shape) and `policies/genai_span_validation.rego`
(span name format, per-op expected attributes) — read these policy files
when authoring scenarios; they're authoritative.

`weaver_live_check` downloads the pinned weaver binary on first use (cached
under `~/.cache/otel-conformance/weaver/`).

## Recorded HTTP (cassettes or transport mock)

A scenario replays one HTTP interaction per operation. Use whichever
mechanism the package's unit tests use, consistently across all of its
scenarios:

- **VCR cassette** — `vcr.use_cassette("<scenario>.yaml")`, one committed
  cassette per operation under `tests/cassettes/`.
- **Transport mock** — build the SDK client with an `httpx.MockTransport`
  (or `respx`) returning a canned response; no cassette file needed.

Pick the one the lib already follows and don't mix the
two within a package.

**AI-generated cassettes.** Lacking provider access, you may synthesize a
cassette from the provider's API reference via AI. Start the cassette with a
`# TODO: this is generated by AI, re-record` comment, mention it in the PR,
and open a follow-up issue to re-record it against the real provider in CI.

## Running

```sh
uv run tox -e py312-test-instrumentation-genai-<lib>-conformance
```

The `*-conformance` tox envs target `tests/test_conformance.py` directly; the
regular `*-{oldest,latest}` envs `--ignore` it so they don't need the
OTLP/gRPC exporter or `weaver_live_check`.

