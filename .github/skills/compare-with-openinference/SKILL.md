---
name: compare-with-openinference
description: Compare an OTel GenAI instrumentation against upstream OpenInference to identify missing spans, attributes, and features. Maps gaps to semantic-conventions-genai schema, separating actionable fixes from semconv blockers.
---

# Compare Instrumentation with OpenInference

Compare an `opentelemetry-instrumentation-genai-<lib>` package against upstream `openinference-instrumentation-<lib>` to find missing spans and attributes. Outputs an actionable checklist.

## Inputs

Target package or library name, e.g.:
- `opentelemetry-instrumentation-genai-openai`
- `<lib>` (e.g. `openai`, `anthropic`, `langchain`)

## Upstream Resolution

### 1. OpenInference Repository
Upstream source: `https://github.com/open-telemetry/donation-openinference`.
Use `/tmp/openinference` or clone:

```sh
if [ ! -d "/tmp/openinference" ]; then
  git clone --depth=1 https://github.com/open-telemetry/donation-openinference.git /tmp/openinference
fi
ls /tmp/openinference/python/instrumentation/ | grep -i "$LIB"
```

Identify target package in `/tmp/openinference/python/instrumentation/openinference-instrumentation-<lib>/`.

### 2. Semantic Conventions Schema Repository
Locate `semantic-conventions-genai` registry using the pinned `SEMCONV_GENAI_REF` from `versions.env`:

```sh
SEMCONV_REF=$(grep SEMCONV_GENAI_REF versions.env | cut -d= -f2)
SEMCONV_DIR="${HOME}/.cache/otel-conformance/semconv/genai-${SEMCONV_REF}"

if [ ! -d "$SEMCONV_DIR" ]; then
  git clone https://github.com/open-telemetry/semantic-conventions-genai.git /tmp/semconv-genai
  git -C /tmp/semconv-genai checkout "$SEMCONV_REF"
  SEMCONV_DIR="/tmp/semconv-genai"
fi
```

Key schema definitions (under `$SEMCONV_DIR`):
- Spans: `model/gen-ai/spans.yaml` and `docs/gen-ai/` (`gen-ai-spans.md`, `gen-ai-agent-spans.md`, provider docs).
- Attributes: `model/gen-ai/registry.yaml` and `model/<provider>/`.

## Analysis

### 1. Spans (Entry Points and Emission Logic)

Audit all span emission sites in OpenInference:
- Method wrappers: wrapped methods in `_instrument()` or patchers.
- Transport hooks: request handlers, `cast_to` dispatch tables, endpoint dispatchers.
- Callbacks/events: all callback methods (`on_llm_start`, `on_chat_model_start`, `on_chain_start`, `on_tool_start`, `on_retriever_start`, etc.).

Compare with OTel package (`instrumentation/<target>/src/`):
- Patched entry points in `_instrument()`.
- Emission and suppression rules (e.g. root-only vs nested runs, child chain suppression).
- Execution hierarchies:
  - Nested workflows or sub-agents -> **Ready to Fix** (`invoke_workflow` or `invoke_agent`).
  - Generic execution nodes/runnables with no semconv operation -> **Blocked on Semantic Conventions**.
- Filter out non-gaps per [instrumentation/AGENTS.md](../../../instrumentation/AGENTS.md):
  - Model call delegation: if the framework delegates model calls to an underlying instrumented library (e.g. `openai`, `litellm`), do not emit `chat` or `embeddings` spans.
  - Plain chains / helper pipelines: plain call chains (RAG query engines, synthesizers) are not AI workflows; telemetry belongs to actual operations (`retrieval`, `chat`).

Classification against `semantic-conventions-genai`:
- Covered by defined semconv span (`chat`, `text_completion`, `embeddings`, `create_agent`, `invoke_agent`, `invoke_workflow`, `execute_tool`, `retrieval`, `fetch_response`) -> **Ready to Fix**.
- No semconv operation exists (e.g. cancel, delete, images, audio, batches, files, generic nodes) -> **Blocked on Semantic Conventions**.

### 2. Extracted Attributes and Features

Audit all attribute extraction in OpenInference:
- Calls to `span.set_attribute(...)` and `SpanAttributes.*`.
- Extracted fields: request params, response metadata, token counts/breakdowns, messages/contents (roles, text, tool calls, parts), tool definitions, streaming events, error handling.

Compare with OTel package and `opentelemetry-util-genai`:
- Map upstream concepts to semconv capability groups (e.g. prompt tracking -> `gen_ai.prompt.*`, token breakdowns -> `gen_ai.usage.*.tokens`).
- Verify span applicability in `model/gen-ai/spans.yaml`:
  - Check whether the emitted span type and included `ref_group`s define the attribute.
  - Match span kind: internal (`gen_ai.*.internal`) for in-process frameworks vs client (`gen_ai.*.client`) for remote/hosted SDKs.
  - Check attribute constraints in `registry.yaml`: remote resource attributes (e.g. `gen_ai.agent.id`) must not be applied to in-memory framework objects.
  - Standard OTel lifecycle: do not flag gaps if standard OTel mechanisms (span status, `error.type`, context propagation) handle the case.
  - Attribute placement: if semconv places the attribute on a parent span (e.g. `gen_ai.tool.definitions` on `invoke_agent`), do not report it missing from child spans.

Classification:
- Defined in semconv for the span type -> **Ready to Fix** (use exact attribute name).
- Not defined in semconv -> **Blocked on Semantic Conventions**.

### 3. Legacy and Deprecated APIs

Identify spans, parameters, or attributes tied to legacy features (e.g. text completions, legacy `functions`, `function_call`). Route to **Gaps for Legacy APIs**.

### 4. Check Open PRs and Issues

Check existing tracking before reporting:
- `gh pr list --search "<lib>" --state open`
- `gh issue list --search "<lib>" --state open`
- Link matching PRs/issues in the checklist items (e.g. `(in progress in #<pr>)`).

## Output Format

Output must be concise and actionable:

```markdown
## Ready to Fix

### Missing Spans
- [ ] Add `<method>` span (`<fully.qualified.method>`) - maps to GenAI `<operation>` (link open PR/issue if in progress)

### Missing Attributes & Features
- [ ] Capture `<feature/attribute>` on `<operation>` (OpenInference: `<openinference_attr>`, semconv: `<gen_ai.attr>`) (link open PR/issue if in progress)

## Blocked on Semantic Conventions

### Missing Spans
- [ ] Add `<method>` span (`<fully.qualified.method>`) (link open PR/issue if in progress)

### Missing Attributes & Features
- [ ] Capture `<feature/attribute>` on `<operation>` (OpenInference: `<openinference_attr>`) (link open PR/issue if in progress)

## Gaps for Legacy APIs
- [ ] Add legacy `<method>` span (`<fully.qualified.method>`) - maps to GenAI `<operation>`
- [ ] Support legacy `<feature/parameter>` on `<operation>` (OpenInference: `<openinference_attr>`)
```

Rules:
- Write `_None_` if a subcategory under `Ready to Fix` is empty.
- Omit `## Blocked on Semantic Conventions` and `## Gaps for Legacy APIs` entirely if clean.

## GitHub Issue Preparation

### 1. Umbrella Issue
- **Title**: `<package-name> gaps vs openinference instrumentation`
- **Body**: Generated markdown report (starts at `## Ready to Fix`).
- **Label**: `openinference-migration` (create if missing).
- Present draft title and body to the user first. Create via `gh issue create` only after explicit user confirmation.

### 2. Individual Tracking Issues (for Ready to Fix items)
If confirmed by the user:
1. Ensure label exists:
   ```sh
   gh label create openinference-migration --description "Tracking parity and migration from OpenInference" --color "1d76db"
   ```
2. Present drafts for each issue:
   - **Title**: `[<lib>] <action>` (e.g. `[openai] Add beta.assistants.create span`)
   - **Label**: `openinference-migration`
   - **Body format**:
     ```markdown
     > [!NOTE]
     > This issue was generated with AI assistance and requires investigation and confirmation prior to being worked on.

     Part of #<umbrella-issue-number>.

     <Short description of the work to be done>

     - <Detail or semconv mapping: maps to gen_ai.operation.name, attribute, etc.>
     ```
3. Create confirmed issues via `gh issue create`.
4. Update umbrella issue body with links to created sub-issues (`#<sub-issue-number>`) via `gh issue edit <umbrella-issue-number> --body-file <updated-body>`.

### Rules for Issue Creation
- Never create GitHub issues automatically without user confirmation.
- Always present drafts (titles and bodies) first.
- Do not mention affected files in sub-issues.
