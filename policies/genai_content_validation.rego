# Validates the JSON payload of GenAI content attributes against the
# semconv JSON schemas.  Schema constants (_schema_*) are defined in
# _schemas.rego, which is generated at test-run time from the semconv
# repository (docs/gen-ai/*.json) and placed alongside this file.
# Weaver only loads .rego files from --advice-policies, so schemas are
# inlined as Rego constants rather than loaded as OPA data documents.

package live_check_advice

import rego.v1

_genai_content_schemas := {
	"gen_ai.input.messages":      _schema_input_messages,
	"gen_ai.output.messages":     _schema_output_messages,
	"gen_ai.system_instructions": _schema_system_instructions,
	"gen_ai.tool.definitions":    _schema_tool_definitions,
	"gen_ai.retrieval.documents": _schema_retrieval_documents,
}

deny contains result if {
	input.sample.attribute
	attr_name := input.sample.attribute.name
	attr_value := input.sample.attribute.value
	is_string(attr_value)

	schema := _genai_content_schemas[attr_name]
	# Skip when the schema constant isn't present in the pinned semconv
	# version yet (the script emits `null` stubs for forward-looking
	# attributes like `gen_ai.tool.definitions` until upstream catches up).
	schema != null

	parsed := json.unmarshal(attr_value)

	[matched, errors] := json.match_schema(parsed, schema)
	not matched

	# PolicyFinding format per
	# https://github.com/open-telemetry/weaver/blob/main/crates/weaver_live_check/README.md#policyfinding
	# (id / level / context / message; signal_* omitted because the sample
	# is attribute-level and weaver doesn't surface the parent span here).
	result := {
		"id":    "genai_content_schema",
		"level": "violation",
		"context": {
			"attribute": attr_name,
			"errors":    errors,
		},
		"message": sprintf(
			"Attribute '%v' value does not conform to the GenAI schema: %v",
			[attr_name, errors],
		),
	}
}
