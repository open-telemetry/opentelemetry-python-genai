---
name: port-from-openinference
description: Port an openinference-instrumentation-* package from https://github.com/open-telemetry/donation-openinference into this repo as a new package. Use when a user asks to migrate a package from OpenInference.
---

# Port an OpenInference `instrumentation-*` package

Migrate an `openinference-instrumentation-<source>` package from
https://github.com/open-telemetry/donation-openinference into this
repo as a new package under `instrumentation/`. This is a **new implementation** 
that emits OTel GenAI semantic conventions through `opentelemetry-util-genai`.

The major
work items are: rewriting the patcher to method-level (step 5), mapping every
request/response shape into OTel `InputMessage`/`OutputMessage` parts
(step 6), and migrating the unit-test corpus while filtering openinference-framework
plumbing tests out (step 7).

## Inputs

User specifies the source, e.g. `openinference-instrumentation-crewai`.

- **Source**: `https://github.com/open-telemetry/donation-openinference/tree/main/python/instrumentation/<source>/`.
  Fetch a fresh shallow clone if you don't already have one locally:
  ```sh
  git clone --depth=1 https://github.com/open-telemetry/donation-openinference.git /tmp/openinference
  ```
  and use `/tmp/openinference/python/instrumentation/<source>/` as the
  source path in step 1.

User may also provide ***target package name**. If not provided: derive it from the source name:  
- drop the leading `openinference-instrumentation-`. Remaining part should match the instrumented library name as it appears on PyPI. If it's not the case, flag it.
- The target package name should be `opentelemetry-instrumentation-genai-<lib>` where `<lib>` is the instrumented library name (e.g. `openai`, `anthropic`, `bedrock`). For example:
  - `openinference-instrumentation-openai` → `opentelemetry-instrumentation-genai-openai`
  - `openinference-instrumentation-anthropic` → `opentelemetry-instrumentation-genai-anthropic`
  Confirm the chosen name with the user.

## Reference material

- **OTel GenAI spans**: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-spans.md> — authoritative attribute names and operation enum.
- **OpenInference → OTel attribute mapping** (Arize-maintained): <https://github.com/Arize-ai/openinference/blob/main/spec/genai/README.md>. Use as a quick lookup for what an OpenInference attribute *roughly* corresponds to in OTel; when the mapping disagrees with the official semconv, **the official semconv wins**.
- **Message JSON schemas**:
  - input messages: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-input-messages.json>
  - output messages: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-output-messages.json>
  - system instructions: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-system-instructions.json>
  - tool definitions: <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-tool-definitions.json>
  - retrieval documents: <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-retrieval-documents.json>

- **Code for above models**: <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/non-normative/models.py>.

## Non-negotiable rules

The repo-wide rules in [AGENTS.md](../../../AGENTS.md) already apply
(telemetry through `opentelemetry-util-genai` public surface only, no
`type: ignore`, semconv enums over string literals, re-raise caught
exceptions). The rules below are the ones the port is most likely to
violate:

1. **Zero OpenInference dependencies.** No `openinference-instrumentation`,
   no `openinference-semantic-conventions`, no `openinference-*` anywhere
   in the port's `src/` or `tests/`. Verify with
   `rg openinference instrumentation/<target>` — output must be empty.
2. **Public util-genai surface only.** Beyond the AGENTS.md rule, the port
   must not import any `opentelemetry.util.genai._*` module — the allowed
   modules are enumerated in step 4.
3. **Ignore all other OpenInference instrumentations during the port.** The only
   instrumentation code to read is the OpenInference package being migrated
   plus `opentelemetry-util-genai`. Build
   from first principles: original OpenInference code + util-genai public API +
   official semconv spec.
4. **Never work around gaps.** If util-genai or the GenAI semconv is
   missing something, flag it and fail the test intentionally
5. **Do not make OTel API calls.** **Exception:**
   semconv attributes that exist in the registry but have no named property
   on `InferenceInvocation` (e.g. `gen_ai.usage.cache_creation.input_tokens`,
   `gen_ai.usage.cache_read.input_tokens`) may be set via
   `invocation.attributes[KEY] = …` — that's still going through the
   util-genai extension point. Import the key from
   `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`. Do
   not invent attribute names that aren't in the semconv.
6. **Reuse VCR cassettes.** Reuse cassettes from the OpenInference
   tests when possible.
7. **Conformance tests must never be silently skipped.** 
   When instrumentation can't be made conformant due to missing
   information, gap in semantic conventions, or in util-genai, still 
   write the scenario and let it fail. 

   Ask user to decide if they want to mark the scenario as skipped with a reason, or
   add an `expected_violation` in the scenario that covers the missing piece.
   All these must be documented in `MIGRATION_REPORT.md` as well, with links to the skipped scenario and the expected violation.

8. **Do not modify weaver policies.**

## Migration flow

### 1. Create the target package

```sh
cp -R <source-path>/ instrumentation/<target>/
cd instrumentation/<target>
rm -rf .pytest_cache .tox .venv venv .vscode .DS_Store .claude .ruff_cache
find . -name __pycache__ -type d -exec rm -rf {} +
rm -f CHANGELOG.md  # OpenInference's per-package history doesn't apply here
```

Keep `LICENSE` (Apache-2.0). Don't carry over `examples/` directories or
OpenInference's `README.md` — both are rewritten below. Per-package
`CHANGELOG.md` is towncrier-generated at release time; don't carry one over,
and add a fragment for the new package per the **Changelog** section of
[AGENTS.md](../../../AGENTS.md).

### 2. Rename the Python module

OpenInference packages live at `src/openinference/instrumentation/<lib>/`.
Move that tree under the OTel GenAI namespace, update path according to the target package name:

```sh
mkdir -p src/opentelemetry/instrumentation/genai
mv src/openinference/instrumentation/<lib> src/opentelemetry/instrumentation/genai/<lib>
rm -rf src/openinference
```

Update every import. Verify zero `openinference` references remain in
`src/`, `tests/`, README. The instrumentor class typically renames from
`<Lib>Instrumentor` (kept as-is — same name is fine).

### 3. Update `pyproject.toml`, `version.py`, and `README.rst`

- `[project] name` → new package name.
- `[project.entry-points.opentelemetry_instrumentor]` → un-prefixed lib name
  pointing at the new module path
  (`<lib> = "opentelemetry.instrumentation.genai.<lib>:<Lib>Instrumentor"`).
- Hatch version path, project URLs, classifiers → new repo paths.
- **Strip every `openinference-*` dependency.** OpenInference packages typically depend on
  `openinference-instrumentation`, `openinference-semantic-conventions`, and
  sometimes `opentelemetry-instrumentation` for the `BaseInstrumentor` mixin
  — keep only the last one. Replace with `opentelemetry-instrumentation` (for
  `BaseInstrumentor`) and the underlying SDK (`openai`, `anthropic`, …) at
  the same range OpenInference was using. 
- `__version__` in `version.py` should equal the value in
  `opentelemetry-util-genai/src/opentelemetry/util/genai/version.py` — all
  workspace packages share one version. Verify:

  ```sh
  diff <(grep ^__version__ instrumentation/<target>/src/opentelemetry/instrumentation/genai/<lib>/version.py) \
       <(grep ^__version__ opentelemetry-util-genai/src/opentelemetry/util/genai/version.py)
  ```

- Hatchling builds **require a `README.md` or `README.rst`**. Rewrite it to
  point at the new repo URLs and module path; drop OpenInference links, OpenInference badges,
  `using_attributes(...)` examples, `OpenInferenceTracer` / `TraceConfig`
  configuration, and any "OpenInference semconv" links. Include a usage snippet
  importing from `opentelemetry.instrumentation.genai.<lib>`, a pointer to
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, and a pointer to
  `tests/conformance/` (no `examples/`).

### 4. Drop OpenInference plumbing

OpenInference ships a framework that's incompatible with this repo's util-genai model
— excise it before touching the patcher.

```sh
rg 'openinference|OpenInferenceTracer|TraceConfig|using_attributes|using_session|using_user|using_metadata|using_tags|SpanAttributes\.|OpenInferenceMimeTypeValues|OpenInferenceSpanKindValues|safe_json_dumps' src/ tests/
```

Drop every match. The mappings:

- **`OpenInferenceTracer` / `TraceConfig`** — replaced by `TelemetryHandler` from
  `opentelemetry.util.genai.handler`. Instrumentation code does not
  construct or pass tracers.
- **`using_attributes(session_id=…, user_id=…, …)` / `using_session` /
  `using_metadata` / `using_tags`** — there is **no OTel GenAI equivalent**
  context-propagation API. Drop the calls and the tests that exercise them
  (those go in the test "skip with reason" bucket — see step 7).
- **`OpenInferenceSpanKindValues` / `OpenInferenceMimeTypeValues`** — drop;
  span kind is set by util-genai based on the invocation type.
- **`SpanAttributes.LLM_*` / `SpanAttributes.INPUT_*`** — flat-string
  OpenInference-semconv attributes. Replaced by typed `InputMessage` / `OutputMessage`
  payloads serialized into `gen_ai.input.messages` / `gen_ai.output.messages`
  by util-genai. The conversion is in step 6.
- **`safe_json_dumps`** — drop; util-genai serializes message payloads.

**Any import of `opentelemetry.util.genai._<anything>` from instrumentation
`src/` is a violation.** Public surface only:

- `opentelemetry.util.genai.handler` — `TelemetryHandler`
- `opentelemetry.util.genai.invocation` — `InferenceInvocation`,
  `EmbeddingInvocation`, `ToolInvocation`, `WorkflowInvocation`,
  `AgentInvocation`, `Error`, `GenAIInvocation`
- `opentelemetry.util.genai.types` — `InputMessage`, `OutputMessage`,
  `Text`, `ToolCallRequest`, `ToolCallResponse`, `Reasoning`,
  `ServerToolCall`, `ServerToolCallResponse`, `GenericPart`, `Blob`,
  `File`, `Uri`, `Modality`
- `opentelemetry.util.genai.completion_hook`
- `opentelemetry.util.genai.environment_variables`

```sh
rg 'from opentelemetry\.util\.genai\._' instrumentation/<target>/src/
```

Output must be empty.

### 5. Rewrite patching: transport → API method level

This is the largest behavioral change. OpenInference typically patches at
the HTTP-transport layer (`OpenAI.request`, `AsyncOpenAI.request`,
`HTTPClient._send_request`, etc.) and dispatches by `cast_to` response type.
**That pattern does not survive the port.** util-genai's
`InferenceInvocation` model needs the request kwargs (`model`, `messages`,
`tools`, `stream`, …) at call time, which only the API methods see.

Replace every transport-level wrapper with method-level
`wrap_function_wrapper` calls — one per public API endpoint — using
**positional args only** (newer wrapt versions reject keyword args):

```python
from wrapt import wrap_function_wrapper
from opentelemetry.instrumentation.utils import unwrap

wrap_function_wrapper(
    "openai.resources.chat.completions",   # module     (positional)
    "Completions.create",                  # name       (positional)
    chat_completions_wrapper,              # wrapper    (positional)
)
```

For uninstrument, use `opentelemetry.instrumentation.utils.unwrap` (matching
positional module + name).

**Patch ALL API methods OpenInference instruments.** Walk OpenInference's
`_instrumentor.py` / `instrumentor.py` plus the dispatch table in the
transport accumulator and enumerate every endpoint OpenInference emits a span for —
including ones with only generic attribute extraction (assistants, threads,
files, fine-tuning, vector stores, batches, uploads, moderations, …). Each
becomes one `wrap_function_wrapper` call. Dropping coverage for an endpoint
because it's "legacy" or "rarely used" is a regression — OTel GenAI semconv
applies to every inference and embedding API regardless of vintage. If a
specific API has no util-genai invocation type yet, that's a gap for the
review report (see step 11), not a reason to drop the patch.

### 6. Map every request and response shape into OTel GenAI types

For each wrapped method, walk OpenInference's input-parsing branch by branch and
ensure each branch has a corresponding mapping in the new wrapper. Same
for output parsing. A wrapper that handles `str` input but not `list`
input (when the original SDK accepts both) is incomplete and must not
ship.

Mapping cheat sheet — OpenInference source shape on the left, OTel construct on the
right (all types from `opentelemetry.util.genai.types` unless noted):

| Source request item | OTel construct |
|---|---|
| User / assistant / system text message | `Input/OutputMessage(role=…, parts=[Text(content=…)])` |
| Assistant message containing a tool/function call | `Message(role="assistant", parts=[ToolCallRequest(name=…, id=…, arguments=…)])` |
| Tool/function result message | `Message(role="tool", parts=[ToolCallResponse(id=…, response=…)])` |
| Reasoning / thinking item | `Message(role="assistant", parts=[Reasoning(content=…)])` |
| Server-side tool call (web_search, file_search, code_interpreter, …) | `Message(parts=[ServerToolCall(name=…, server_tool_call=…, id=…)])` |
| Server-side tool call result | `Message(parts=[ServerToolCallResponse(server_tool_call_response=…, id=…)])` |
| Inline image / audio / video bytes | `Blob(mime_type=…, modality="image"\|"audio"\|"video", content=b"…")` |
| External media URL | `Uri(mime_type=…, modality=…, uri="https://…")` |
| File reference (e.g. OpenAI `file_id`) | `File(mime_type=…, modality=…, file_id="file-…")` |
| Provider-specific item with no semconv mapping | `GenericPart(value=…)` — never silently drop. Flag those in the review report. |

Output messages mirror the input mapping — `OutputMessage` serializes with
a `parts` array (not `content`); each part has a `type` field. When
asserting on `gen_ai.output.messages`, parse the JSON and check
`msg["parts"]`.

`Modality` is `Literal["image", "video", "audio"]`. `error.type` and span
status come from `invocation.fail(exc)` — do not emit a separate span
exception event.

### 7. Restructure tests

```text
tests/
  cassettes/<scenario>.yaml
  conformance/
    inference.py / embedding.py / ...   # see step 8
  conftest.py
  test_<existing>.py                    # unit tests; refactor onto helpers
  requirements.{oldest,latest}.txt
```

**Categorize OpenInference tests before migrating.** List every test function in the
OpenInference package and bucket each one:

- ✅ **Migrate** — exercises a patched API method. Rewrite assertions
  from flat OpenInference semconv attributes (`SpanAttributes.LLM_INPUT_MESSAGES_…`)
  to OTel constructs: assert on `span.attributes[GenAIAttributes.GEN_AI_…]`
  (semconv constants), and parse `gen_ai.input.messages` / `gen_ai.output.messages`
  JSON to check the `parts` arrays.
- ✅ **Migrate (rewrite)** — unit test for an OpenInference-internal helper
  (attribute extractor, message parser, etc.) where the helper is gone but
  the **parsing scenario** still applies. Rewrite as an integration test
  that feeds the same response shape through VCR and asserts the
  resulting OTel telemetry. This is the bucket most likely to be
  mis-categorized — anything covering tool-call objects, refusals,
  reasoning items, multi-content messages, token-usage breakdowns, image
  inputs, raw-response variants (e.g. `with_raw_response`) is parsing
  domain logic and **must** be migrated. The wrapper has to do the
  parsing; the test has to verify it.
- ❌ **Skip with reason** — exercises OpenInference framework plumbing with no OTel
  equivalent: `using_attributes()` context propagation, `TraceConfig`
  masking, `OpenInferenceTracer` behavior, OpenInference flat-attribute naming format,
  `OpenInferenceSpanKindValues` checks. Document the reason in a comment
  in the test file (or in `MIGRATION_REPORT.md` later — see step 11).
  Do **not** skip a test because the API it exercises is "legacy" or
  "new" — semconv applies to all inference APIs.

Decision rule for the ✅ rewrite vs ❌ skip split: ask whether the test
covers a *response shape* from the instrumented library, or *OpenInference framework
behavior*. A test that constructs a tool-call object with `arguments` /
`call_id` / `name` and verifies it's extracted into attributes covers a
response shape — migrate it. A test that checks
`using_attributes(session_id=…)` propagates into span attributes covers OpenInference
framework behavior — skip it.

**Sanity check before committing step 7.** Count source vs ported tests:

```sh
rg -c '^\s*(async )?def test_' <source-path>/tests/
rg -c '^\s*(async )?def test_' instrumentation/<target>/tests/
```

A port that drops from 80 tests to 5 is a regression — go back to the
"migrate" and "migrate (rewrite)" buckets and finish them.

**Replace conftest boilerplate.** OpenInference conftests duplicate the
exporter/provider/VCR plumbing that lives here in
`opentelemetry-test-util-genai`. Don't copy OpenInference's — mirror an
existing package's conftest (e.g.
`instrumentation/opentelemetry-instrumentation-genai-openai/tests/conftest.py`),
which registers the shared fixtures as plugins:

```python
pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]
```

The lib-specific conftest then adds only: `vcr_config` (per-package
`filter_headers` and `before_record_response`), an `environment` autouse
for the lib's API-key env var, library-client fixtures (e.g.
`openai_client` / `async_openai_client`), and the `instrument_*` fixtures
(`instrument_no_content`, `instrument_with_content`, `instrument_event_only`)
built on the shared `instrument` context manager from
`opentelemetry.test_util_genai.instrumentor` (see [AGENTS.md](../../../AGENTS.md) Tests).

**Assertions — no shared helper module.** There is no
`opentelemetry.test_util_genai.assertions`. Assert directly on
`span.attributes[GenAIAttributes.GEN_AI_…]` using the semconv constants
from `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`, and
on metric/log records from the in-memory exporters. Factor repeated checks
into a per-package `tests/test_utils.py` (existing packages have helpers
like `assert_all_attributes`, `assert_completion_attributes`,
`assert_messages_attribute`, plus weather-tool fixtures). OpenInference
helpers (`_check_llm_attributes`, etc.) map onto these — rewrite them on
top of OTel semconv constants and parsed `gen_ai.*.messages` JSON, and keep
tiny constants (`DEFAULT_MODEL`, sample prompts) inline or in
`tests/test_utils.py`.

**Required unit-test coverage per wrapped method.** Apply the repo test
matrix (sync/async × happy/error, plus streaming × happy/error where the
method streams — see [AGENTS.md](../../../AGENTS.md) Tests section)
to **every** method patched in step 5. For the port these are blockers for
the migration PR, not follow-up. The error variants must verify the
original exception is re-raised, `error.type` is recorded, and span status
is ERROR.

**`tests/requirements.{latest,oldest}.txt`** — OpenInference typically has its own
pin file; keep only the third-party version pins (`openai==`,
`anthropic==`, `pydantic==`, `pytest==`, …). Drop every `-e <path>` line
— this repo's uv workspace already installs all members editable via
`uv sync --all-packages`. Drop `requirements.pydantic1.txt`-style
side-channel files entirely.

### 8. Conformance scenarios

Author conformance scenarios using the **`write-conformance-tests`** skill —
it's the generic procedure (scenario modules, the `test_conformance.py`
runner, declared gaps, lib-specific assertions, weaver policies) and applies
to any instrumentation. Port-specific notes on top of that skill:

- Drop OpenInference's `examples/` tree — its end-to-end demos are replaced
  by conformance scenarios, not ported.
- For an operation blocked by a util-genai/semconv gap, point the
  `expected_violations` / `xfail` `reason=` at the gap row in
  `MIGRATION_REPORT.md`.

### 9. Cassettes

- Copy cassettes from OpenInference's `tests/cassettes/` (or wherever the OpenInference package
  parks them) into the port's `tests/cassettes/`. Reuse names so existing
  unit tests keep loading them.
- Reuse existing cassettes for conformance scenarios when they are applicable.

### 10. Workspace integration

Wire the new package into the workspace, `tox.ini`, and pyright per the
**Adding a package to the workspace** section of [AGENTS.md](../../../AGENTS.md)
— it applies to any new package, not just ports. Port-specific note on top:

- **Leave the package out of `[tool.pyright] include`.** A port over untyped
  `wrapt` boundaries (`wrapped, instance, args, kwargs`) and vendor SDK members
  produces hundreds of strict-mode errors, so don't add it to `include` until
  typing lands — track that as a follow-up.

### 11. Local checks, review, and PR

Run the pre-PR checks from the **Commands** section of
[AGENTS.md](../../../AGENTS.md) — `tox -e precommit`, `tox -e typecheck`, and
the package's `-{oldest,latest}` (and `-conformance`) test envs.

Open the PR with the `migration:openinference` label. Run the
`review-ported` skill locally to generate `MIGRATION_REPORT.md`; iterate
until §4 (test coverage) is clean. The review skill compares the port
against OpenInference (or any upstreams you name), so coverage gaps
surface in one report.

## See also

- [AGENTS.md](../../../AGENTS.md) — general repo rules that already apply to the port.
- `util/opentelemetry-util-genai/AGENTS.md` — util-genai usage rules.
- `.github/skills/write-conformance-tests/SKILL.md` — generic conformance-scenario authoring (step 8).
- `.github/skills/review-ported/SKILL.md` — sister review skill (writes `MIGRATION_REPORT.md`).
