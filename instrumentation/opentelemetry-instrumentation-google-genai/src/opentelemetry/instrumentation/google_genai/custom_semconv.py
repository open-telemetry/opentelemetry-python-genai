# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


# Semantic Convention still being defined in:
# https://github.com/open-telemetry/semantic-conventions/pull/2125
GCP_GENAI_OPERATION_CONFIG = "gcp.gen_ai.operation.config"

# Prompt size of an interaction. gen_ai.usage.input_tokens instead sums the
# input billed across automatic continuation hops, each of which re-prefills it.
GCP_GENAI_USAGE_INITIAL_INPUT_TOKENS = "gcp.gen_ai.usage.initial_input_tokens"
