# Validates GenAI span shape beyond what weaver's semconv-registry-driven
# checks already enforce. The registry validates per-attribute requirements
# (name, type, presence) for spans matching its definitions; this file adds
# cross-cutting span-level invariants the registry can't easily express.
#
# Two classes of rules, both keyed on `gen_ai.operation.name`:
#
#   1. Span name format → `violation`
#      (`{operation_name} {request_model}` for inference / embeddings,
#      `{operation_name} {agent_name}` for invoke_agent / create_agent,
#      `{operation_name} {tool_name}` for execute_tool).
#
#   2. Per-operation expected attributes → `violation`
#      Combines `Required` (always must be set) and the always-emit subset
#      of `Recommended` (e.g. response model/id, token usage on inference)
#      into one manifest per operation. Sourced from the rendered tables in
#      semantic-conventions/docs/gen-ai/gen-ai-spans.md and
#      gen-ai-agent-spans.md (the MD flattens the YAML inheritance chain
#      via `extends:`, so it's the right place to source from).
#
# The "set when known" Recommended subset (sampling parameters like
# `frequency_penalty`, `max_tokens`; provider-side caches; conditionally-
# emitted things like `gen_ai.response.time_to_first_chunk` for streaming)
# is deliberately NOT flagged here — those depend on user input or on the
# request shape and would produce noisy false positives. Cross-attribute
# conditional rules (e.g. "if streaming, response.time_to_first_chunk
# SHOULD be set") would also belong here.
#
# Required attributes are also flagged by weaver's registry-driven
# validation. Listing them here too is intentional: rego rules give us
# stable advice ids to grep for in reports and let us tighten the check
# regardless of how the registry classifies the gap.
#
# Attribute access: weaver hands rego a span sample where `attributes` is a
# **list** of `{name, value, type}` objects, not a dict — `_attr(name)`
# walks that list and returns the value (or `null` if absent).

package live_check_advice

import rego.v1

# ─── Operation classification ───────────────────────────────────────────────
#
# Mirrors the semconv `gen_ai.operation.name` enum
# (model/gen-ai/registry.yaml). When semconv adds a new operation, append it
# to the matching set below — or leave it out if the new operation has its
# own span definition with different conventions.

_inference_ops := {"chat", "generate_content", "text_completion"}

_embeddings_ops := {"embeddings"}

_tool_ops := {"execute_tool"}

_invoke_agent_ops := {"invoke_agent"}

_create_agent_ops := {"create_agent"}

# ─── Span name format (violation) ───────────────────────────────────────────

_span_name_keyed_attr["chat"]              := "gen_ai.request.model"
_span_name_keyed_attr["generate_content"]  := "gen_ai.request.model"
_span_name_keyed_attr["text_completion"]   := "gen_ai.request.model"
_span_name_keyed_attr["embeddings"]        := "gen_ai.request.model"
_span_name_keyed_attr["execute_tool"]      := "gen_ai.tool.name"
_span_name_keyed_attr["invoke_agent"]      := "gen_ai.agent.name"
_span_name_keyed_attr["create_agent"]      := "gen_ai.agent.name"

# Span name SHOULD be `{op}` (when the keyed attribute is absent) or
# `{op} {value}` (when present). Mirrors the "SHOULD append when known"
# guidance in semconv.
#
# Avoid `%v ` patterns in sprintf: weaver 0.22.1's OPA-based sprintf
# consumes a single space character immediately following any verb (`%v`,
# `%s`, `%d`) — interpreting it as Go's space-flag — so `%v %v` produces
# `<a><b>` instead of `<a> <b>`. We use `concat` for the literal-space
# joins below.
deny contains _span_finding(
	"genai_span_name_format",
	"violation",
	input.sample.span,
	{
		"operation":     op,
		"keyed_attr":    keyed_attr,
		"expected_form": concat("", [op, " or '", op, " <", keyed_attr, ">'"]),
	},
	concat("", [
		op, " span name should be '",
		op, "' or '",
		op, " <value of ", keyed_attr, ">', got '",
		input.sample.span.name, "'",
	]),
) if {
	input.sample.span
	op := _attr_value(input.sample.span, "gen_ai.operation.name")
	keyed_attr := _span_name_keyed_attr[op]
	not _valid_op_and_attr_span_name(input.sample.span, op, keyed_attr)
}

# ─── Per-operation expected attributes (violation) ──────────────────────────

_expected_for_op["chat"] := _inference_expected

_expected_for_op["generate_content"] := _inference_expected

_expected_for_op["text_completion"] := _inference_expected

_expected_for_op["embeddings"] := _embeddings_expected

_expected_for_op["execute_tool"] := _execute_tool_expected

_expected_for_op["invoke_agent"] := _invoke_agent_expected

_expected_for_op["create_agent"] := _create_agent_expected

_expected_for_op["retrieval"] := _retrieval_expected

# Inference (chat / generate_content / text_completion).
# Required: gen_ai.operation.name, gen_ai.provider.name.
# Always-emit Recommended: response model/id, finish reasons, token usage,
# server.address. Sampling parameters (frequency_penalty, max_tokens, …),
# cache counters, and `gen_ai.response.time_to_first_chunk` (streaming-only)
# are conditional and not flagged here.
_inference_expected := {
	"gen_ai.operation.name",
	"gen_ai.provider.name",
	"gen_ai.response.model",
	"gen_ai.response.id",
	"gen_ai.response.finish_reasons",
	"gen_ai.usage.input_tokens",
	"gen_ai.usage.output_tokens",
	# "server.address", sometimes not available
}

# Embeddings.
# Required: gen_ai.operation.name, gen_ai.provider.name.
# Always-emit Recommended: dimension.count, response.model, input tokens,
# server.address. (`gen_ai.request.encoding_formats` is conditional.)
_embeddings_expected := {
	"gen_ai.operation.name",
	"gen_ai.provider.name",
	"gen_ai.embeddings.dimension.count",
	"gen_ai.response.model",
	"gen_ai.usage.input_tokens",
	# "server.address", sometimes not available
}

# Tool execution.
# Required: gen_ai.operation.name, gen_ai.tool.name.
# Recommended-when-available: gen_ai.tool.call.id, gen_ai.tool.type. (Tool
# description is genuinely optional per provider — not flagged.)
_execute_tool_expected := {
	"gen_ai.operation.name",
	"gen_ai.tool.name",
	"gen_ai.tool.call.id",
	"gen_ai.tool.type",
}

# Invoke agent.
# Required: gen_ai.operation.name, gen_ai.provider.name.
_invoke_agent_expected := {
	"gen_ai.operation.name",
	"gen_ai.provider.name",
}

# Create agent. After creation completes the provider returns an agent.id;
# flag it as always-emit on create_agent.
_create_agent_expected := {
	"gen_ai.operation.name",
	"gen_ai.provider.name",
	"gen_ai.agent.id",
}

# Retrieval. Only gen_ai.operation.name is unconditionally required.
_retrieval_expected := {
	"gen_ai.operation.name",
	# "server.address", sometimes not available
}

# Per expected attribute, one violation if missing.
deny contains _span_finding(
	"genai_expected_attribute_missing",
	"violation",
	input.sample.span,
	{
		"operation":         op,
		"missing_attribute": attr_name,
	},
	sprintf(
		"Span '%v' (operation '%v') is missing expected attribute '%v'",
		[input.sample.span.name, op, attr_name],
	),
) if {
	input.sample.span
	op := _attr_value(input.sample.span, "gen_ai.operation.name")
	expected := _expected_for_op[op]
	some attr_name in expected
	not _has_attr(input.sample.span, attr_name)
}

# ─── Unknown gen_ai.operation.name (violation) ──────────────────────────────
#
# Weaver's built-in `undefined_enum_variant` advice is `information`-level;
# we raise unknown values on `gen_ai.operation.name` to a violation. Keep
# `_known_operation_names` in sync with model/gen-ai/registry.yaml.

_known_operation_names := {
	"chat",
	"generate_content",
	"text_completion",
	"embeddings",
	"retrieval",
	"create_agent",
	"invoke_agent",
	"execute_tool",
	"invoke_workflow",
	"plan",
}

deny contains _span_finding(
	"genai_operation_name_unknown",
	"violation",
	input.sample.span,
	{"operation": op},
	sprintf(
		"Span '%v' has gen_ai.operation.name='%v' which is not a documented enum value",
		[input.sample.span.name, op],
	),
) if {
	input.sample.span
	op := _attr_value(input.sample.span, "gen_ai.operation.name")
	not _known_operation_names[op]
}

# ─── Span status (violation) ────────────────────────────────────────────────
#
# Per the OpenTelemetry trace spec, instrumentation libraries MUST NOT set
# span status to OK — that value is reserved for application code that has
# explicitly verified the call succeeded. Instrumentations should leave
# status UNSET on success and set ERROR on failure.
# https://opentelemetry.io/docs/specs/otel/trace/api/#set-status

deny contains _span_finding(
	"genai_span_status_ok_set_by_instrumentation",
	"violation",
	input.sample.span,
	{"status_code": input.sample.span.status.code},
	sprintf(
		"Span '%v' has status.code='ok'; instrumentations must leave status UNSET on success (OK is reserved for application code).",
		[input.sample.span.name],
	),
) if {
	input.sample.span
	input.sample.span.status.code == "ok"
}

# ─── Helpers ────────────────────────────────────────────────────────────────

# Span attributes arrive as `[{"name": ..., "value": ..., "type": ...}]`.

# True when the span has an attribute named `name`.
_has_attr(span, name) if {
	some attr in span.attributes
	attr.name == name
}

# Returns the value of the named attribute. Undefined (rule body fails) when
# the attribute isn't present — callers must guard with `_has_attr` first if
# they need to distinguish "absent" from "set to a falsy value".
_attr_value(span, name) := value if {
	some attr in span.attributes
	attr.name == name
	value := attr.value
}

# A valid span name is either exactly `{op}` (when the keyed attribute is
# absent) or `{op} {value}` (when present).
_valid_op_and_attr_span_name(span, op, attr_key) if {
	span.name == op
	not _has_attr(span, attr_key)
}

_valid_op_and_attr_span_name(span, op, attr_key) if {
	value := _attr_value(span, attr_key)
	# concat (not sprintf): see the note above the deny rule. sprintf("%v %v", ...)
	# silently produces "<a><b>" with no space, so every span with a `{op} {value}`
	# name would be reported as a violation.
	span.name == concat(" ", [op, value])
}

# PolicyFinding format per
# https://github.com/open-telemetry/weaver/blob/main/crates/weaver_live_check/README.md#policyfinding
_span_finding(id, level, span, context, message) := {
	"id":          id,
	"level":       level,
	"signal_type": "span",
	"signal_name": span.name,
	"context":     context,
	"message":     message,
}
