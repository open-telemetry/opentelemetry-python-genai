---
name: review-migration
description: Review a migrated or augmented instrumentation-genai package by comparing it against any known external upstream implementations of the same instrumentation (OpenInference, vendor-specific). Handles both greenfield migrations and augment-mode PRs that add coverage to a pre-existing package (checking the added parts, their consistency with existing code, and old-vs-new coexistence). Writes MIGRATION_REPORT.md in the migrated package root.
---

# Review a migrated instrumentation-genai package

Compare the migrated package in `instrumentation/<target>/` against
every known upstream implementation of the same instrumentation, and
write a `MIGRATION_REPORT.md` in the migrated package root listing only
what's missing or wrong.

## Inputs and upstreams

User supplies the **migrated package name**, e.g.
`opentelemetry-instrumentation-genai-openai`. Strip
`opentelemetry-instrumentation-genai-` to get `<lib>`; that drives the
upstream lookup.

**Upstreams come from the user, with a default fallback:**

- **If the user names one or more upstreams** (a repo, a directory path, a
  vendor's own SDK-side instrumentation, a Logfire / Pydantic AI package,
  …), compare against exactly those — one column per upstream. Use them
  verbatim; don't second-guess or substitute.
- **If the user names none**, default to OpenInference: search
  `https://github.com/open-telemetry/donation-openinference` under
  `python/instrumentation/` for a package matching `<lib>` (typically
  `openinference-instrumentation-<lib>`, but confirm by listing the
  directory — the name may differ). Fetch a shallow clone if you don't
  already have one locally:

  ```sh
  git clone --depth=1 https://github.com/open-telemetry/donation-openinference.git /tmp/openinference
  ls /tmp/openinference/python/instrumentation/ | rg <lib>
  ```

If there are no upstreams to compare against (no user-named upstream resolves and no
OpenInference match), this isn't a migration — bail with a one-line note.

## Greenfield vs augment mode

Detect the mode before reviewing — it changes what counts as a problem:

- **Greenfield migration** — the migration *created* the package; everything in
  `instrumentation/<target>/` is the migrated package.
- **Augment mode** — the package **already existed** and the migration only
  *adds* coverage mined from the upstream. The diff modifies a pre-existing
  package.

Tell them apart from the PR diff / git history / unstaged changes: if the
package's `src/` and tests predate the migration commits, it's augment mode.

In augment mode the deliverable shifts to: the **added** coverage's
completeness vs the upstream, its **consistency** with the pre-existing code,
and **old-vs-new coexistence** concerns. Do **not** flag pre-existing
coverage the upstream also has as missing. Per-section adjustments are inline
below.

## Rules

- **Deliverable is `instrumentation/<target>/MIGRATION_REPORT.md`.** Write
  there, not stdout. The file is gitignored — regenerate freely.
- **Report problems, not work.** No "✅ matches", no "all tests migrated",
  no recap of structure. The PR diff already shows what's there; the
  report is for what's missing or wrong.
- **Don't justify empty findings.** When a section/table has no problems,
  emit `_none_` (or skip the section, where the rule says so) and stop.
  Do not list which greps you ran, which env vars you searched for, or
  why you concluded clean. Clean means clean.
- **Tables only when there are ≥1 problem rows.** Empty section: `_none_`.
- **Read, don't guess.** Every claim from actual code or command output.
  Don't infer from package names or READMEs. Before listing something as
  missing, grep for it in the migrated package's `src/` and `tests/`.
- **Snapshot of the PR head**, not aspirational state.

## Report structure

First line:

```markdown
# Migration review: <target-package>

Mode: greenfield migration | augment existing package

Compared against:
- OpenInference: `openinference-instrumentation-<lib>` (add link)
- <other upstream>: `<path>`, `<link>` (omit the line if no other upstream
  is in scope)
```

State the mode on the second line. In augment mode, one sentence after it
naming what the PR adds (the gaps it closes) orients the reviewer.

Sections render in order. Each is bounded to its problems.

### 1. Instrumented API surface

Table of every API method any source patches, marking methods the
migrated package does **not** patch. **Be exhaustive.** Enumerate every
endpoint upstream emits a span for, even if attribute extraction
is generic. Do not collapse rows.

For each upstream, walk the actual instrumentation code, not docs:

- **Method-level patching** (the shape of this migration, and of any
  method-level upstream): read every `wrap_function_wrapper(...)` call in
  `_instrument()`. Each call = one row.
- **Transport-level patching** (typical of OpenInference, e.g.
  `openai.OpenAI.request` / `AsyncOpenAI.request`): enumerate every
  `cast_to` response type the accumulator/dispatch table handles **and**
  every other endpoint that flows through the wrapped method (assistants,
  threads, files, fine-tuning, images, audio, vector_stores, batches,
  uploads, moderations, …). Spans with only generic attribute extraction
  still count — list them, and note "generic span only" in the Notes.
- **Anything else** (vendor SDK hooks, monkey-patched class methods,
  decorator-based instrumentation): walk the actual entry points the
  upstream registers and list one row per emitted span site.

One column per upstream that exists, plus `This package`:

| API method | OpenInference | This package | Notes |
|---|---|---|---|
| `openai.resources.chat.completions.Completions.create` | ✅ | ✅ | |
| `openai.resources.responses.Responses.create` | ✅ | ❌ | |
| `openai.resources.beta.assistants.Assistants.create` | ✅ (generic span only) | ❌ | |

- ✅ = patched. ❌ = at least one upstream patches it, this migration doesn't.
  — = not patched (and that's expected — upstream doesn't patch it either).
- Sort `❌` rows to the top.
- For `❌` rows, the **Notes** cell must name the GenAI semconv operation
  the method maps to (`chat`, `embeddings`, `text_completion`,
  `invoke_agent`, `invoke_workflow`, `execute_tool`, `realtime`, …) — or
  "no semconv operation defined yet" if the spec hasn't caught up. 

Do not render a separate `**Gaps:**` bullet list below the table — the
table itself is the gap list.

**Augment mode.** Split `This package` into `Existed` and `Added this PR`:

| API method | OpenInference | Existed | Added this PR | Notes |
|---|---|---|---|---|
| `…chat.completions.Completions.create` | ✅ | ✅ | — | already covered |
| `…responses.Responses.create` | ✅ | ❌ | ✅ | added by this PR |
| `…batches.Batches.create` | ✅ | ❌ | ❌ | still missing — `chat` |

`❌` in both columns (upstream patches it, the package still doesn't) is the
gap; sort it to the top.

### 2. Gaps and open issues

Genuine **tooling/util gaps** — things the migration couldn't do because
`opentelemetry-util-genai` and/or the GenAI semconv doesn't yet support them
(missing util-genai factory, attribute not in the registry yet). 
Reference failing/skipped tests if any.

Do NOT list "OTel semconv doesn't cover X" as a gap when X is just an
inference API variant — semconv applies to all inference APIs.

| Gap | File / test | Upstream issue | Notes |
|---|---|---|---|

### 3. Significant behavioral changes

Material differences vs upstream — equivalences (different code shape,
same emitted telemetry) are not listed. Look for: patching strategy
(transport-level vs method-level); error recording (exception event vs
`error.type` + status); token counting (cache details folded vs
separate); message format (flat OpenInference attributes vs JSON `parts` array);
streaming wrapper shape; attributes set unconditionally vs gated on
`is_recording()` / `should_capture_content()`; anything emitted by
upstream that this migration deliberately drops without a one-line rationale.

| Aspect | Upstream | This package | Notes |
|---|---|---|---|

### 3b. Consistency and old-vs-new coexistence (augment mode only)

**Greenfield migration: skip.** Here the risk isn't "is it conformant" but "do
the additions fit the package they landed in." Compare the **added** code
against the **pre-existing** code (not the upstream); flag only real
problems:

- **Divergent patterns** — added wrappers/helpers/fixtures re-implementing
  something the package already has a convention for.
- **Duplicated scaffolding** — added tests not reusing the package's
  `tests/test_utils.py` helpers or conformance runner.
- **Telemetry contradictions** — an added method emitting an operation name,
  attribute shape, span-kind, or content-capture gating that disagrees with
  what pre-existing methods emit for the analogous case.
- **Coexistence hazards** — added code that alters pre-existing paths (shared
  module-level state, `_instrument()` ordering, a dependency range the old
  code wasn't tested against), or a `pyproject`/`tox` edit affecting existing
  tests.

| Concern | Pre-existing | Added | Notes |
|---|---|---|---|

`_none_` if the additions are consistent and isolated.

### 4. Test coverage

Three checklists. Render only **missing** cells; if every checklist is
clean, render `_Test coverage complete._` and skip the subsections.

Gaps in this section are **blockers for the migration PR**, not
follow-up work — address them before merge. Do not list §4 items as
follow-up issues in §5.

**Augment mode.** The matrix applies to methods **added this PR** (§1
`Added this PR` = ✅) — those block merge. A pre-existing method missing a
variant is a §5 follow-up, not a §4 blocker.

Also flag here: **🟡 missing-cassette** — any scenario or unit test that
references a cassette not committed under `tests/cassettes/`. And:
**unreferenced cassettes** — one-line count of `tests/cassettes/*.yaml`
files no test/scenario opens. Skip if 0.

For each unreferenced cassette, walk git history before naming a cause:

1. Check whether the same cassette is also unreferenced in the upstream
   it came from (for each unreferenced file, search the upstream's
   `tests/` for any reference). If yes, it's an **inherited upstream
   orphan** — say so plainly; the migration is not responsible.
2. If the cassette IS referenced upstream but not in the migrated package, the migration
   dropped a test. That is **not** a §4b cosmetic note — it's a §4a
   missing variant (or a missing scenario, depending on what the test
   covered). Add it to §4a's table with the upstream test name in Notes,
   and do not rationalize ("tests were renamed/removed during the migration"
   without naming the commit is speculation, not analysis).

Never write "appear to be" / "likely" / "safe to delete" without
evidence. Either you walked the history and know, or you didn't — in
which case say "history not checked" so the reviewer does it.

#### 4a. Unit-test matrix per wrapped method

For each method in §1 row where `This package` = ✅:

| Variant | Required when |
|---|---|
| **sync × happy** | always |
| **sync × error** | always — verify the original exception is re-raised unmodified, `error.type` is recorded, span status is ERROR |
| **async × happy** | the lib exposes an async counterpart |
| **async × error** | the lib exposes an async counterpart |
| **streaming × happy** | the method accepts `stream=True` or returns a stream wrapper |
| **streaming × error** | flag if streaming is supported but no error path is exercised at all or error is not recorded on telemetry |
| **async streaming × happy** | the method accepts `stream=True` or returns an async stream wrapper |
| **async streaming × error** | flag if streaming is supported but no error path is exercised at all or error is not recorded on telemetry |

Identify variants by reading the migrated package's `src/` (`is_streaming(kwargs)`,
async `def`, `Stream` / `AsyncStream` wrappers).

| Wrapped method | Missing variants | Notes |
|---|---|---|

#### 4b. Conformance scenarios

For each distinct GenAI semconv operation the migrated package emits (`chat`,
`embeddings`, `execute_tool`, `invoke_agent`, `invoke_workflow`,
`create_agent`, …) there should be at least one happy-path scenario file under
`tests/conformance/<op>.py` driven by `run_conformance(...)`. More
scenarios per operation are fine but never required.

| Operation | Scenario file | Status |
|---|---|---|

Mention if conformance scenario is skipped, there are expected_violations,
or `uv run tox -e py312-test-instrumentation-genai-<lib>` fails.

#### 4c. Docstring / README coverage

| Asset | Required content | Status |
|---|---|---|
| `README.rst` | install snippet, usage snippet importing from `opentelemetry.instrumentation.genai.<lib>`, pointer to `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` and the util-genai README, pointer to `tests/conformance/` (no `examples/`) | |
| `src/opentelemetry/instrumentation/genai/<lib>/__init__.py` module docstring | usage example using the new module path; configuration section listing the env vars users actually need | |

Render only rows that are missing/wrong/stale (e.g. README still imports
from `openinference.instrumentation.<lib>`, install command points
at the `openinference-instrumentation-<lib>` package name, links to an
OpenInference URL).

**Refactor misses.** One bullet *only if* unit tests re-implement generic
semconv shape checks inline instead of factoring them into the migrated package's
`tests/test_utils.py` helpers (e.g. `assert_all_attributes`,
`assert_completion_attributes`, `assert_messages_attribute`). There is no
shared `opentelemetry.test_util_genai.assertions` module — assert directly
on `span.attributes[GenAIAttributes.GEN_AI_…]` using the semconv constants.

### 5. Follow-up work

What goes in this PR vs. a follow-up:

- **In this PR**: §4 test coverage gaps. These block merge. In augment mode,
  scoped to the methods added this PR, plus any §3b consistency problem in the
  added code (it ships here).
- **Follow-up**: §1 ❌ rows (new instrumented methods), §2 util-genai /
  semconv gaps, §3 behavioral parity items, and — augment mode — pre-existing
  coverage gaps from §4 — each a **separate** PR, one logical change apiece.

Suggest an issue title per follow-up item, grouped by type (API surface /
util-genai gaps / behavioral parity); name the upstream that covers it and
the semconv operation where relevant. **Do not file the issues** — listing
them is the deliverable; filing AI-generated issues is against the
contributor policy, so the human author decides which to open. If there
are none, render `_No follow-up issues recommended._`

## See also

- `.github/skills/migrate-from-openinference/SKILL.md` — the migration skill; it
  runs this review at its final step to produce `MIGRATION_REPORT.md`.
- `.github/skills/write-conformance-tests/SKILL.md` — authoring the
  conformance scenarios this report checks in §4b.
- `.github/instructions/instrumentation.instructions.md` — the copilot
  PR-review rules for `instrumentation/**`; generic instrumentation
  violations are flagged there and not repeated in this report.