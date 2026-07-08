---
name: migrate-from-openinference
description: Migrate an openinference-instrumentation-* package from https://github.com/open-telemetry/donation-openinference into this repo. Creates a new package, or — when a package for the library already exists in the repo — augments it with the coverage OpenInference adds on top. Use when a user asks to migrate or port a package from OpenInference.
---

# Migrate an OpenInference `instrumentation-*` package

Migrate an `openinference-instrumentation-<source>` package from
https://github.com/open-telemetry/donation-openinference into this
repo. The result emits OTel GenAI semantic conventions through
`opentelemetry-util-genai`.

Two modes, decided by the "Before you start" gate below:

- **Greenfield migration** — no package for the library exists yet. Create a
  **new implementation** under `instrumentation/`. The default, and what the
  bulk of this skill describes.
- **Augment an existing package** — the repo already ships
  `opentelemetry-instrumentation-genai-<lib>`. Don't re-create it; inventory
  what it covers, diff against OpenInference, and add **only the missing
  parts**. See [Augment mode](#augment-mode-the-package-already-exists).

For a greenfield migration the major work items are: rewriting the patcher to
method-level (step 5), mapping every request/response shape into OTel
`InputMessage`/`OutputMessage` parts (step 6), and migrating the unit-test
corpus while filtering openinference-framework plumbing tests out (step 7).

## Inputs

User specifies the source, e.g. `openinference-instrumentation-crewai`.

- **Source**: It should point to one of the folders in 
  `https://github.com/open-telemetry/donation-openinference/tree/main/python/instrumentation/`.
  Fetch a fresh shallow clone if you don't already have one locally:
  ```sh
  git clone --depth=1 https://github.com/open-telemetry/donation-openinference.git /tmp/openinference
  ```
  and use `/tmp/openinference/python/instrumentation/<source>/` as the
  source path in step 1.

User may also provide the **target package name**. If not provided: derive it from the source name:
- drop the leading `openinference-instrumentation-`. Remaining part should match the instrumented library name as it appears on PyPI. If it's not the case, flag it.
- The target package name should be `opentelemetry-instrumentation-genai-<lib>` where `<lib>` is the instrumented library name (e.g. `openai`, `anthropic`, `bedrock`). For example:
  - `openinference-instrumentation-openai` → `opentelemetry-instrumentation-genai-openai`
  - `openinference-instrumentation-anthropic` → `opentelemetry-instrumentation-genai-anthropic`
  Confirm the chosen name with the user.

## Before you start: is there already a package for this library?

Once the target name is settled, check whether the repo already ships it:

```sh
ls instrumentation/opentelemetry-instrumentation-genai-<lib> 2>/dev/null
```

- **Exists →** **augment mode**: don't scaffold a new package. OpenInference
  is now a *second* reference to mine for coverage the existing package
  lacks. Jump to [Augment mode](#augment-mode-the-package-already-exists).
- **Doesn't exist →** greenfield migration; continue below.

If a near-name sibling (a `-agents` / `-client` suffix) might instrument a
*different* surface of the same vendor, confirm the target name with the user
before deciding.

## Before you start: check for native OTel instrumentation

AI SDKs increasingly ship their **own** OpenTelemetry GenAI instrumentation.
When they do, migrating the OpenInference package is redundant. So
before writing any code, determine whether the instrumented library is
self-instrumenting.

```sh
# 1. Does the SDK depend on the OTel API / semconv?
pip show <lib> | grep -i requires        # or read its pyproject / METADATA
#    a dependency on opentelemetry-api or opentelemetry-semantic-conventions
#    is the tell.
# 2. Does its source actually emit GenAI spans?
python -c "import <lib>, os; print(os.path.dirname(<lib>.__file__))"
rg -l "opentelemetry|semconv\._incubating\.attributes\.gen_ai" <site-packages>/<lib>
```

A dependency on `opentelemetry-api` (or `-semantic-conventions`) **plus**
`gen_ai_attributes` usage in the SDK source means the library is
self-instrumented. Confirm empirically: set a **global** `TracerProvider`
(native hooks often activate only when a real, non-proxy provider is
configured), make one call, and inspect the emitted spans' instrumentation
scope.

**If the library is self-instrumented, do NOT migrate the OpenInference
package.** Pivot the work:

1. **Ignore the OpenInference instrumentation entirely** — the vendor owns
   the spans; there is nothing to re-implement, and no `src/` instrumentor /
   patcher to write.
2. **Write conformance tests against the native instrumentation.** Follow
   step 8 / the `write-conformance-tests` skill, but each scenario's `run()`
   configures providers and enables the **native** tracer (e.g. sets a global
   `TracerProvider`) instead of calling a `*Instrumentor` — then runs the
   emitted telemetry through weaver live-check.
3. **Identify gaps / inconsistencies** between the native output and the
   GenAI semconv: missing operations, wrong operation name, legacy/duplicate
   attributes, no metrics, no content-capture controls, no util-genai
   content modes / completion-hook / upload support, etc. Record each as an
   `expected_violation` or a documented skip, same as a normal migration.
4. **Write `MIGRATION_REPORT.md`** stating the library is self-instrumented,
   the conformance results, and the gap list — that report is the
   deliverable. **Stop and surface the finding to the user.** Do not build a
   competing package unless they explicitly decide to (e.g. to suppress
   native instrumentation and layer util-genai features on top).

Only when the library has **no** native OTel instrumentation do you continue
with the migration flow below.

## Reference material

- **OTel GenAI spans**: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai> — authoritative attribute names, spans, logs, and metrics definitions.
- **OpenInference → OTel attribute mapping** (Arize-maintained): <https://github.com/Arize-ai/openinference/blob/e9a8746daeb184c9aabc68ca29c05909ddcccf1e/spec/genai/README.md>. Use as a quick lookup for what an OpenInference attribute *roughly* corresponds to in OTel; when the mapping disagrees with the official semconv, **the official semconv wins**.
- **Message JSON schemas**:
  - input messages: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-input-messages.json>
  - output messages: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-output-messages.json>
  - system instructions: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-system-instructions.json>
  - tool definitions: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-tool-definitions.json>
  - retrieval documents: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/gen-ai-retrieval-documents.json>

- **Code for above models**: <https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai/non-normative/models.py>.

## Non-negotiable rules

The repo-wide rules in [AGENTS.md](../../../AGENTS.md) already apply
(telemetry through `opentelemetry-util-genai` public surface only, no
`type: ignore`, semconv enums over string literals, re-raise caught
exceptions). The rules below are the ones the migration is most likely to
violate:

1. **Zero OpenInference dependencies.** No `openinference-instrumentation`,
   no `openinference-semantic-conventions`, no `openinference-*` anywhere
   in the migrated package's `src/` or `tests/`. Verify with
   `rg openinference instrumentation/<target>` — output must be empty.
2. **Public util-genai surface only.** Beyond the AGENTS.md rule, the migrated package
   must not import any `opentelemetry.util.genai._*` module — the allowed
   modules are enumerated in step 4.
3. **Ignore all other OpenInference instrumentations during the migration.** The only
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

## Augment mode: the package already exists

The package already has a working, conformant implementation; OpenInference
is just another reference to mine for missing coverage. The job is a tight,
delta-only PR that closes specific gaps — **not** a rewrite.

All [Non-negotiable rules](#non-negotiable-rules) apply to every line you
add, plus two specific to this mode:

- **Don't rewrite or "improve" existing code.** Leave the existing patcher,
  wrappers, and tests alone unless OpenInference reveals a concrete bug — and
  then it's a *separate* PR. No opportunistic refactors.
- **Match the existing package's conventions.** New wrappers, helpers, and
  scenarios follow the patterns already there (helper names, wrapper
  structure, conftest/fixture wiring, scenario shape). Don't add a second way
  to do something the package already does.

### A. Inventory what's already there

Map the existing package before reading OpenInference (read, don't guess):
patched methods (`wrap_function_wrapper` / `unwrap` in `_instrument`),
request/response shapes its wrappers map, the unit-test matrix per method
(`rg -c '^\s*(async )?def test_' instrumentation/<target>/tests/`),
conformance scenarios under `tests/conformance/`, and cassettes.

### B. Diff against OpenInference

Run the OpenInference analysis as a greenfield migration would (the reading behind
steps 5–6): every method it patches, every shape it parses. Subtract
inventory A. The remainder is the work-list:

- Methods OpenInference patches that the package doesn't → new wrappers (step 5).
- Shape branches OpenInference handles that the wrappers drop → extend the
  mapping (step 6).
- Scenarios OpenInference covers that the package lacks → new tests /
  conformance (steps 7–8).

Coverage both already have → skip. Coverage the package has but OpenInference
lacks → leave it; not a regression.

### C. Add the delta

Apply steps **5–10 to the new parts only**:

- **Steps 1–3 (scaffold/rename/pyproject) skipped** — touch `pyproject.toml`
  only for a genuinely new entry point or dependency range, `README.rst` only
  for a new user-visible capability.
- **Step 4** applies to anything you copy in; nothing to excise from existing
  code.
- **Steps 5–9** extend the existing wrappers / test utils / conformance
  runner in place, not parallel ones.
- **Step 10** is done; revisit `tox.ini` / pyright only for a new test factor
  or requirements file.

### D. Report and review

Write `MIGRATION_REPORT.md` via the `review-migration` skill as usual — it
detects augment mode.

## Migration flow

> The numbered steps below are written for a **greenfield migration**. In
> [augment mode](#augment-mode-the-package-already-exists) skip steps 1–3,
> and scope steps 5–10 to the delta from the inventory/diff (sections A–C
> above).

### 1. Create the target package

Because the patcher, wrappers, and tests are all rewritten (steps 5–7), a
`cp -R` of the OpenInference tree mostly creates files you immediately
delete or overwrite. **Prefer scaffolding fresh** from the nearest existing
package (e.g. `opentelemetry-instrumentation-genai-anthropic`) and copy over from
OpenInference **only** what you actually reuse:

- `LICENSE` (Apache-2.0).
- Reusable cassettes (step 9), if any.

```sh
mkdir -p instrumentation/<target>/src/opentelemetry/instrumentation/genai/<lib>
mkdir -p instrumentation/<target>/tests/conformance instrumentation/<target>/tests/cassettes
cp <source-path>/LICENSE instrumentation/<target>/LICENSE
```

Do **not** carry over `examples/`, OpenInference's `README.md`, or its
`CHANGELOG.md` (per-package changelogs are towncrier-generated at release
time).

(If you do `cp -R` instead, clean it up afterwards:
`rm -rf .pytest_cache .tox .venv venv .vscode .DS_Store .claude .ruff_cache CHANGELOG.md`
and `find . -name __pycache__ -type d -exec rm -rf {} +`.)

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
  `util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py` — all
  workspace packages share one version. Verify:

  ```sh
  diff <(grep ^__version__ instrumentation/<target>/src/opentelemetry/instrumentation/genai/<lib>/version.py) \
       <(grep ^__version__ util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py)
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
**That pattern does not survive the migration.** util-genai's
`InferenceInvocation` model needs the request kwargs (`model`, `messages`,
`tools`, `stream`, …) at call time, which only the API methods see.

Replace every transport-level wrapper with method-level
`wrap_function_wrapper` calls — one per public API endpoint. Pass
**positional args only** to `wrap_function_wrapper` (newer wrapt versions reject keyword args):

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
| External media URL | `Uri(mime_type=…, modality=…, uri="…")` |
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

**Sanity check before committing step 7.** Count source vs migrated tests:

```sh
rg -c '^\s*(async )?def test_' <source-path>/tests/
rg -c '^\s*(async )?def test_' instrumentation/<target>/tests/
```

A migration that drops from 80 tests to 5 is a regression — go back to the
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
to **every** method patched in step 5. For the migration these are blockers for
the migration PR, not follow-up. The error variants must verify the
original exception is re-raised, `error.type` is recorded, and span status
is ERROR.

**`tests/requirements.{latest,oldest}.txt`** — OpenInference typically has its own
pin file; keep only the third-party version pins (`openai==`,
`anthropic==`, …). 
Add current `util/opentelemetry-util-genai` and `instrumentation/opentelemetry-instrumentation-genai-<lib>` to the latest; for the oldest, use the oldest version of `opentelemetry-util-genai` that works (there is none released yet, but check).

### 8. Conformance scenarios

Author conformance scenarios using the **`write-conformance-tests`** skill —
it's the generic procedure (scenario modules, the `test_conformance.py`
runner, declared gaps, lib-specific assertions, weaver policies) and applies
to any instrumentation. Migration-specific notes on top of that skill:

- Drop OpenInference's `examples/` tree — its end-to-end demos are replaced
  by conformance scenarios, not migrated.
- For an operation blocked by a util-genai/semconv gap, point the
  `expected_violations` / `xfail` `reason=` at the gap row in
  `MIGRATION_REPORT.md`.

### 9. Cassettes (or a transport proxy)

- Copy cassettes from OpenInference's `tests/cassettes/` (or wherever the OpenInference package
  parks them) into the migrated package's `tests/cassettes/`. Reuse names so existing
  unit tests keep loading them.
- Reuse existing cassettes for conformance scenarios when they are applicable.
- **AI-generated cassettes.** For a cassette OpenInference lacks and you
  can't record (no provider access), you may synthesize one from the
  provider's API reference via AI. Start it with a
  `# TODO: this is generated by AI, re-record` comment, mention it in the PR,
  and open a follow-up issue to re-record it against the real provider in CI.

**Transport proxy instead of cassettes.** If the OpenInference unit tests mock
HTTP (e.g. `respx`, `httpx.MockTransport`) rather than replay recorded
cassettes, you may do the same in the migrated package's **unit** tests — build the SDK
client with an `httpx.MockTransport` (or equivalent) returning canned
responses instead of `@pytest.mark.vcr`. When you go this route:

- Reuse the OpenInference cassettes' recorded **response bodies** (incl. the
  streaming SSE payloads) as the canned responses, so fidelity is preserved.
- Still register the shared VCR plugins in `conftest.py` (the shared
  `fixture_vcr` is autouse) and keep `vcr_config`.
- **Use the same mechanism in conformance scenarios.** Conformance does not
  require VCR — a scenario can build its client with the same transport mock
  and ignore the injected `vcr` (see the `write-conformance-tests` skill).
  Pick one mechanism (cassettes *or* transport mock) and use it consistently
  across the whole package.
- **Mention the choice in `MIGRATION_REPORT.md`** (the `review-migration` skill
  flags missing cassettes; note that the package mocks the transport by
  design so the absence is not a gap).

### 10. Workspace integration

Wire the new package into the workspace, `tox.ini`, and pyright per the
**Adding a package to the workspace** section of [AGENTS.md](../../../AGENTS.md)
— it applies to any new package, not just migrations. Migration-specific note on top:

- **Leave the package out of `[tool.pyright] include`.** A migration over untyped
  `wrapt` boundaries (`wrapped, instance, args, kwargs`) and vendor SDK members
  produces hundreds of strict-mode errors, so don't add it to `include` until
  typing lands — track that as a follow-up.

### 11. Local checks, review, and PR

Run the pre-PR checks from the **Commands** section of
[AGENTS.md](../../../AGENTS.md) — `tox -e precommit`, `tox -e typecheck`, and
the package's `-{oldest,latest}` (and `-conformance`) test envs.

Run the `review-migration` skill locally to generate `MIGRATION_REPORT.md`; iterate
until §4 (test coverage) is clean. The review skill compares the migrated package
against OpenInference (or any upstreams you name), so coverage gaps
surface in one report.

Finally ask human to create a PR with the `migration:openinference` label and
post the contents of `MIGRATION_REPORT.md` as the PR comment.

## See also

- [AGENTS.md](../../../AGENTS.md) — general repo rules that already apply to the migration.
- `util/opentelemetry-util-genai/AGENTS.md` — util-genai usage rules.
- `.github/skills/write-conformance-tests/SKILL.md` — generic conformance-scenario authoring (step 8).
- `.github/skills/review-migration/SKILL.md` — sister review skill (writes `MIGRATION_REPORT.md`).
