# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from enum import Enum
from typing import Final

GEN_AI_AGENT_DESCRIPTION: Final[str] = "gen_ai.agent.description"
"""Free-form description of the GenAI agent provided by the application."""

GEN_AI_AGENT_ID: Final[str] = "gen_ai.agent.id"
"""The unique and stable identifier of the GenAI hosted agent resource.
For hosted agents, this SHOULD be the provider-assigned stable identifier of the agent resource such as [AWS Bedrock agent ARN](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_Agent.html) or [GCP Agent Registry identifier](https://docs.cloud.google.com/agent-registry/concepts#agent-identifier).
It's NOT RECOMMENDED to record in-memory agent instance ids on this attribute due to their transient nature."""

GEN_AI_AGENT_NAME: Final[str] = "gen_ai.agent.name"
"""Human-readable name of the GenAI agent provided by the application."""

GEN_AI_AGENT_VERSION: Final[str] = "gen_ai.agent.version"
"""The version of the GenAI agent."""

GEN_AI_CONVERSATION_COMPACTED: Final[str] = "gen_ai.conversation.compacted"
"""Indicates whether the effective conversation context used for this operation is a compacted view of a prior conversation.
This attribute is a positive indicator of context compaction. Instrumentations
SHOULD set it to `true` only when they can reliably determine that context
compaction was applied. Instrumentations SHOULD NOT set it to `false`; they
SHOULD leave it unset otherwise."""

GEN_AI_CONVERSATION_ID: Final[str] = "gen_ai.conversation.id"
"""The unique identifier for a conversation (session, thread), used to store and correlate messages within this conversation.
Instrumentations SHOULD populate conversation id when they have an identifier
for the conversation readily available for a given operation, for example:

- when the client framework being instrumented manages conversation history
(see [LlamaIndex chat store](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/),
[LangChain `session_id`](https://reference.langchain.com/python/langchain-core/runnables/history/RunnableWithMessageHistory),
and [Google ADK sessions](https://adk.dev/sessions/session))
- when instrumenting GenAI client libraries that maintain a conversation on the backend
(see [AWS Bedrock agent sessions](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-session-state.html),
[OpenAI Assistant threads](https://platform.openai.com/docs/api-reference/threads))

When no identifier for the conversation is available, instrumentations SHOULD NOT
populate conversation id. For example, a new UUID, a trace identifier, or a hash
of request content SHOULD NOT be used as a fallback value.

Application developers that manage conversation history MAY add conversation id to GenAI and other
spans or logs using custom span or log record processors or hooks provided by instrumentation
libraries."""

GEN_AI_DATA_SOURCE_ID: Final[str] = "gen_ai.data_source.id"
"""The data source identifier.
Data sources are used by AI agents and RAG applications to store grounding data. A data source may be an external database, object store, document collection, website, or any other storage system used by the GenAI agent or application. The `gen_ai.data_source.id` SHOULD match the identifier used by the GenAI system rather than a name specific to the external storage, such as a database or object store. Semantic conventions referencing `gen_ai.data_source.id` MAY also leverage additional attributes, such as `db.*`, to further identify and describe the data source."""

GEN_AI_EMBEDDINGS_DIMENSION_COUNT: Final[str] = (
    "gen_ai.embeddings.dimension.count"
)
"""The number of dimensions the resulting output embeddings should have."""

GEN_AI_EVALUATION_EXPLANATION: Final[str] = "gen_ai.evaluation.explanation"
"""A free-form explanation for the assigned score provided by the evaluator."""

GEN_AI_EVALUATION_NAME: Final[str] = "gen_ai.evaluation.name"
"""The name of the evaluation metric used for the GenAI response."""

GEN_AI_EVALUATION_SCORE_LABEL: Final[str] = "gen_ai.evaluation.score.label"
"""Human readable label for evaluation.
This attribute provides a human-readable interpretation of the evaluation score produced by an evaluator. For example, a score value of 1 could mean "relevant" in one evaluation system and "not relevant" in another, depending on the scoring range and evaluator. The label SHOULD have low cardinality. Possible values depend on the evaluation metric and evaluator used; implementations SHOULD document the possible values."""

GEN_AI_EVALUATION_SCORE_VALUE: Final[str] = "gen_ai.evaluation.score.value"
"""The evaluation score returned by the evaluator."""

GEN_AI_INPUT_MESSAGES: Final[str] = "gen_ai.input.messages"
"""The chat history provided to the model as an input.
Messages MUST be provided in the order they were sent to the model.
Instrumentations MAY provide a way for users to filter or truncate
input messages.

> [!Warning]
> This attribute is likely to contain sensitive information including user/PII data.

See [Recording content on attributes](/docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes)
section for more details."""

GEN_AI_MEMORY_QUERY_TEXT: Final[str] = "gen_ai.memory.query.text"
"""The search query used to retrieve memories.
Instrumentations SHOULD NOT capture this attribute by default. Capture SHOULD be gated
by an explicit user opt-in, for example `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.

> [!Warning]
> This attribute may contain sensitive information."""

GEN_AI_MEMORY_RECORD_COUNT: Final[str] = "gen_ai.memory.record.count"
"""The number of memory records relevant to the operation.
For `search_memory` operations, this is the number of memory records returned by the operation. For `create_memory` operations, this is the number of memory records the operation attempted to create. For `update_memory` operations, this is the number of memory records the operation attempted to modify. For `upsert_memory` operations, this is the number of memory records the operation attempted to create or update. For `delete_memory` operations, this is the number of memory records the operation attempted to delete."""

GEN_AI_MEMORY_RECORD_ID: Final[str] = "gen_ai.memory.record.id"
"""The unique identifier of the memory record."""

GEN_AI_MEMORY_RECORDS: Final[str] = "gen_ai.memory.records"
"""The memory records stored or retrieved in a memory operation.
Instrumentations SHOULD NOT capture this attribute by default. Capture SHOULD be gated
by an explicit user opt-in, for example `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.

> [!Warning]
> This attribute may contain sensitive information including user/PII data."""

GEN_AI_MEMORY_STORE_ID: Final[str] = "gen_ai.memory.store.id"
"""The unique identifier of the memory store.
Semantic conventions for individual components SHOULD document what `gen_ai.memory.store.id` maps to within the implementation."""

GEN_AI_OPERATION_NAME: Final[str] = "gen_ai.operation.name"
"""The name of the operation being performed.
If one of the predefined values applies, but specific system uses a different name it's RECOMMENDED to document it in the semantic conventions for specific GenAI system and use system-specific name in the instrumentation. If a different name is not documented, instrumentation libraries SHOULD use applicable predefined value."""

GEN_AI_OUTPUT_MESSAGES: Final[str] = "gen_ai.output.messages"
"""Messages returned by the model where each message represents a specific model response (choice, candidate).
Each message represents a single output choice/candidate generated by
the model. Each message corresponds to exactly one generation
(choice/candidate) and vice versa - one choice cannot be split across
multiple messages or one message cannot contain parts from multiple choices.

Instrumentations MAY provide a way for users to filter or truncate
output messages. `gen_ai.response.finish_reasons` remains aligned with the
generations returned by the provider, not with a filtered or truncated
`gen_ai.output.messages` value.

> [!Warning]
> This attribute is likely to contain sensitive information including user/PII data.

See [Recording content on attributes](/docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes)
section for more details."""

GEN_AI_OUTPUT_TYPE: Final[str] = "gen_ai.output.type"
"""Represents the content type requested by the client.
This attribute SHOULD be used when the client requests output of a specific type. The model may return zero or more outputs of this type.
This attribute specifies the output modality and not the actual output format. For example, if an image is requested, the actual output could be a URL pointing to an image file.
Additional output format details may be recorded in the future in the `gen_ai.output.{type}.*` attributes."""

GEN_AI_PROMPT_NAME: Final[str] = "gen_ai.prompt.name"
"""The name of the prompt that uniquely identifies it."""

GEN_AI_PROMPT_VARIABLE: Final[str] = "gen_ai.prompt.variable"
"""The variables supplied to the prompt template, the `<key>` being the variable name, the value being the variable value.
Prompt templates are parameterized with variables that are filled in
at runtime. This attribute records the variable values passed to the
template. The attribute name defines the variable name, and the
attribute value is the variable value serialized as a string.

Examples:

- A variable `user_name` with value `Alice` SHOULD be recorded as
  the `gen_ai.prompt.variable.user_name` attribute with value `"Alice"`.
- A variable `language` with value `French` SHOULD be recorded as
  the `gen_ai.prompt.variable.language` attribute with value `"French"`.

> [!Warning]
> This attribute may contain sensitive information."""

GEN_AI_PROMPT_VERSION: Final[str] = "gen_ai.prompt.version"
"""The version of the prompt template used.
The version string can follow any versioning scheme chosen by the
application (e.g., SemVer, date-based, or platform-specific tags).
When a prompt management system is in use, this SHOULD match the
version identifier used by that system."""

GEN_AI_PROVIDER_NAME: Final[str] = "gen_ai.provider.name"
"""The Generative AI provider as identified by the client or server instrumentation.
Semantic conventions for individual GenAI operations SHOULD clarify which
kinds of providers (e.g. inference, embeddings, retrieval, memory, hosted
agent providers) apply when it is not clear from context.

The attribute SHOULD be set based on the instrumentation's best knowledge
and may differ from the actual upstream provider. For example, a client SDK
may be configured against a proxy or hosting platform that transparently
relays requests to a different provider.

The `gen_ai.provider.name` attribute acts as a discriminator that
identifies the GenAI telemetry format flavor specific to that provider
within GenAI semantic conventions.
It SHOULD be set consistently with provider-specific attributes and signals.
For example, GenAI spans, metrics, and events related to AWS Bedrock
should have the `gen_ai.provider.name` set to `aws.bedrock` and include
applicable `aws.bedrock.*` attributes and are not expected to include
`openai.*` attributes."""

GEN_AI_REQUEST_CHOICE_COUNT: Final[str] = "gen_ai.request.choice.count"
"""The target number of candidate completions to return."""

GEN_AI_REQUEST_ENCODING_FORMATS: Final[str] = "gen_ai.request.encoding_formats"
"""The encoding formats requested in an embeddings operation, if specified.
In some GenAI systems the encoding formats are called embedding types. Also, some GenAI systems only accept a single format per request."""

GEN_AI_REQUEST_FREQUENCY_PENALTY: Final[str] = (
    "gen_ai.request.frequency_penalty"
)
"""The frequency penalty setting for the GenAI request."""

GEN_AI_REQUEST_MAX_TOKENS: Final[str] = "gen_ai.request.max_tokens"
"""The maximum number of tokens the model generates for a request."""

GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
"""The name of the GenAI model a request is being made to."""

GEN_AI_REQUEST_PRESENCE_PENALTY: Final[str] = "gen_ai.request.presence_penalty"
"""The presence penalty setting for the GenAI request."""

GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: Final[str] = (
    "gen_ai.request.previous_response.id"
)
"""The unique identifier of a previous response or interaction used to provide context for the current operation.
Instrumentations SHOULD populate this attribute when the request references a previous response or interaction identifier to continue a conversation or pass prior context.
For example, `previous_response_id` in [OpenAI Responses API](https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response)
or `previous_interaction_id` in [Google GenAI Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)."""

GEN_AI_REQUEST_REASONING_LEVEL: Final[str] = "gen_ai.request.reasoning.level"
"""The reasoning or thinking effort level requested for a GenAI model.
The value SHOULD be the exact string value sent to the provider.
Semantic conventions for individual providers SHOULD document which input parameter maps to this attribute."""

GEN_AI_REQUEST_SEED: Final[str] = "gen_ai.request.seed"
"""Requests with same seed value more likely to return same result."""

GEN_AI_REQUEST_STOP_SEQUENCES: Final[str] = "gen_ai.request.stop_sequences"
"""List of sequences that the model will use to stop generating further tokens."""

GEN_AI_REQUEST_STREAM: Final[str] = "gen_ai.request.stream"
"""Indicates whether the GenAI request was made in streaming mode."""

GEN_AI_REQUEST_STREAM_CURSOR: Final[str] = "gen_ai.request.stream_cursor"
"""The cursor identifying the last streamed event already received, used to resume a streamed response from that position.
Instrumentations SHOULD populate this attribute when a request resumes a streamed response from a prior position, for example when fetching a stored response with streaming enabled.
For example, `starting_after` in the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/background#streaming-a-background-response)
or `last_event_id` in the [Google GenAI Interactions API](https://ai.google.dev/api/interactions-api)."""

GEN_AI_REQUEST_TEMPERATURE: Final[str] = "gen_ai.request.temperature"
"""The temperature setting for the GenAI request."""

GEN_AI_REQUEST_TOP_K: Final[str] = "gen_ai.request.top_k"
"""The top-K sampling setting for the GenAI request: restricts token generation at each step to the K most likely next tokens.
This is a decoding/sampling parameter (e.g., Anthropic `top_k`, Cohere `k`, Google `topK`), not an output-shaping parameter. In particular, OpenAI's `top_logprobs` controls how many per-token log-probabilities are returned in the response and does not change generation; it MUST NOT be reported as `gen_ai.request.top_k`."""

GEN_AI_REQUEST_TOP_P: Final[str] = "gen_ai.request.top_p"
"""The top_p sampling setting for the GenAI request."""

GEN_AI_RESPONSE_FINISH_REASONS: Final[str] = "gen_ai.response.finish_reasons"
"""Array of reasons the model stopped generating tokens, corresponding to each generation received.
Values correspond to generations in the same order as the returned
choices/candidates.

Each position SHOULD contain the finish reason for the corresponding
choice/candidate. If a finish reason was expected but not received, for
example because generation failed, was cancelled, or a stream ended before
the final event, instrumentations SHOULD report `error` for that position
instead of omitting it.

`error` indicates that the generation ended abnormally, whether reported
by the provider or inferred by the instrumentation."""

GEN_AI_RESPONSE_ID: Final[str] = "gen_ai.response.id"
"""The unique identifier for the completion."""

GEN_AI_RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
"""The name of the model that generated the response."""

GEN_AI_RESPONSE_STATUS: Final[str] = "gen_ai.response.status"
"""The lifecycle status of a generated response, as reported by the provider when the response is fetched or polled.
This attribute captures the lifecycle state of a (possibly background or long-running) generation, such as whether it is queued, still running, or has reached a terminal state. It is distinct from `gen_ai.response.finish_reasons`, which describes why the model stopped once it began producing output.
The value SHOULD be the provider's response status mapped onto the closest member of this enum or a provider-specific value when none of these apply. Semantic conventions for individual GenAI providers SHOULD document how their status values map to this attribute."""

GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: Final[str] = (
    "gen_ai.response.time_to_first_chunk"
)
"""Time to first chunk in a streaming response, measured from request issuance, in seconds. The value is measured from when the client issues the generation request to when the first chunk is received in the response stream."""

GEN_AI_RETRIEVAL_DOCUMENTS: Final[str] = "gen_ai.retrieval.documents"
"""The documents retrieved.
Each document object SHOULD contain the following properties when available:
`id` (string): A unique identifier for the document, `score` (double): The relevance score of the document"""

GEN_AI_RETRIEVAL_QUERY_TEXT: Final[str] = "gen_ai.retrieval.query.text"
"""The query text used for retrieval.
> [!Warning]
> This attribute may contain sensitive information."""

GEN_AI_RETRIEVAL_TOP_K: Final[str] = "gen_ai.retrieval.top_k"
"""The maximum number of documents the retriever was asked to return for the query (also known as `k`, `limit`, or `max_num_results`)."""

GEN_AI_SYSTEM_INSTRUCTIONS: Final[str] = "gen_ai.system_instructions"
"""The system message or instructions provided to the GenAI model separately from the chat history.
This attribute SHOULD be used when the corresponding provider or API
allows to provide system instructions or messages separately from the
chat history.

Instructions that are part of the chat history SHOULD be recorded in
`gen_ai.input.messages` attribute instead.

Instrumentations MAY provide a way for users to filter or truncate
system instructions.

> [!Warning]
> This attribute may contain sensitive information.

See [Recording content on attributes](/docs/gen-ai/gen-ai-spans.md#recording-content-on-attributes)
section for more details."""

GEN_AI_TOKEN_TYPE: Final[str] = "gen_ai.token.type"
"""The type of token being counted."""

GEN_AI_TOOL_CALL_ARGUMENTS: Final[str] = "gen_ai.tool.call.arguments"
"""Parameters passed to the tool call.
> [!WARNING]
> This attribute may contain sensitive information.

It's expected to be an object - in case a serialized string is available
to the instrumentation, the instrumentation SHOULD do the best effort to
deserialize it to an object."""

GEN_AI_TOOL_CALL_ID: Final[str] = "gen_ai.tool.call.id"
"""The tool call identifier."""

GEN_AI_TOOL_CALL_RESULT: Final[str] = "gen_ai.tool.call.result"
"""The result returned by the tool call (if any and if execution was successful).
> [!WARNING]
> This attribute may contain sensitive information.

It's expected to be an object - in case a serialized string is available
to the instrumentation, the instrumentation SHOULD do the best effort to
deserialize it to an object."""

GEN_AI_TOOL_DEFINITIONS: Final[str] = "gen_ai.tool.definitions"
"""The list of tool definitions available to the GenAI agent or model.
> [!WARNING]
> This attribute may contain sensitive information.

Since this attribute could be large, it's NOT RECOMMENDED to populate
non-required properties by default. Instrumentations MAY provide a way
to enable populating optional properties."""

GEN_AI_TOOL_DESCRIPTION: Final[str] = "gen_ai.tool.description"
"""The tool description.
> [!WARNING]
> This attribute may contain sensitive information."""

GEN_AI_TOOL_NAME: Final[str] = "gen_ai.tool.name"
"""Name of the tool utilized by the agent."""

GEN_AI_TOOL_TYPE: Final[str] = "gen_ai.tool.type"
"""Type of the tool utilized by the agent
Extension: A tool executed on the agent-side to directly call external APIs, bridging the gap between the agent and real-world systems.
  Agent-side operations involve actions that are performed by the agent on the server or within the agent's controlled environment.
Function: A tool executed on the client-side, where the agent generates parameters for a predefined function, and the client executes the logic.
  Client-side operations are actions taken on the user's end or within the client application.
Datastore: A tool used by the agent to access and query structured or unstructured external data for retrieval-augmented tasks or knowledge updates."""

GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS: Final[str] = (
    "gen_ai.usage.audio.cache_read.input_tokens"
)
"""The number of audio input tokens served from a provider-managed cache.
The value SHOULD be included in `gen_ai.usage.cache_read.input_tokens` and in `gen_ai.usage.audio.input_tokens`."""

GEN_AI_USAGE_AUDIO_INPUT_TOKENS: Final[str] = "gen_ai.usage.audio.input_tokens"
"""The number of audio input tokens.
The value SHOULD be included in `gen_ai.usage.input_tokens`."""

GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: Final[str] = (
    "gen_ai.usage.audio.output_tokens"
)
"""The number of audio output tokens.
The value SHOULD be included in `gen_ai.usage.output_tokens`."""

GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: Final[str] = (
    "gen_ai.usage.cache_read.input_tokens"
)
"""The number of input tokens served from a provider-managed cache.
The value SHOULD be included in `gen_ai.usage.input_tokens`."""

GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: Final[str] = (
    "gen_ai.usage.cache_write.input_tokens"
)
"""The number of input tokens written to a provider-managed cache.
The value SHOULD be included in `gen_ai.usage.input_tokens`."""

GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS: Final[str] = (
    "gen_ai.usage.image.cache_read.input_tokens"
)
"""The number of image input tokens served from a provider-managed cache.
The value SHOULD be included in `gen_ai.usage.cache_read.input_tokens` and in `gen_ai.usage.image.input_tokens`."""

GEN_AI_USAGE_IMAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.image.input_tokens"
"""The number of image input tokens.
The value SHOULD be included in `gen_ai.usage.input_tokens`."""

GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS: Final[str] = (
    "gen_ai.usage.image.output_tokens"
)
"""The number of image output tokens.
The value SHOULD be included in `gen_ai.usage.output_tokens`."""

GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
"""The number of tokens used in the GenAI input (prompt).
This value SHOULD include all types of input tokens, including cached tokens.
Instrumentations SHOULD make a best effort to populate this value, using a total
provided by the provider when available or, depending on the provider API,
by summing different token types parsed from the provider output.

When the provider reports both billed token counts and model-consumed
token counts (for example, Cohere exposes both `usage.billed_units` and
`usage.tokens`), instrumentations SHOULD report the billed count so the
value matches the units the customer is charged for.

Detailed usage attributes are subsets of total and aggregate counts. For example,
if a request has 100 text tokens (40 cached) and 200 image tokens:
- `gen_ai.usage.input_tokens`: 300
- `gen_ai.usage.cache_read.input_tokens`: 40
- `gen_ai.usage.text.input_tokens`: 100
- `gen_ai.usage.text.cache_read.input_tokens`: 40
- `gen_ai.usage.image.input_tokens`: 200"""

GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
"""The number of tokens used in the GenAI response (completion).
When the provider reports both billed token counts and model-consumed
token counts (for example, Cohere exposes both `usage.billed_units` and
`usage.tokens`), instrumentations SHOULD report the billed count so the
value matches the units the customer is charged for."""

GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: Final[str] = (
    "gen_ai.usage.reasoning.output_tokens"
)
"""The number of output tokens used for reasoning (e.g. chain-of-thought, extended thinking).
The value SHOULD be included in `gen_ai.usage.output_tokens`."""

GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS: Final[str] = (
    "gen_ai.usage.text.cache_read.input_tokens"
)
"""The number of text input tokens served from a provider-managed cache.
The value SHOULD be included in `gen_ai.usage.cache_read.input_tokens` and in `gen_ai.usage.text.input_tokens`."""

GEN_AI_USAGE_TEXT_INPUT_TOKENS: Final[str] = "gen_ai.usage.text.input_tokens"
"""The number of text input tokens.
The value SHOULD be included in `gen_ai.usage.input_tokens`."""

GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.text.output_tokens"
"""The number of text output tokens.
The value SHOULD be included in `gen_ai.usage.output_tokens`."""

GEN_AI_WORKFLOW_NAME: Final[str] = "gen_ai.workflow.name"
"""Human-readable name of the GenAI workflow provided by the application.
The workflow name is usually a static, application-unique identifier defined
in a framework-specific way.

For example, it can be the name of the first chain in LangChain,
the name of the crew in CrewAI, or the entry point agent in ADK or
OpenAI Agents when no explicit workflow name is provided.

This attribute MUST have low cardinality. It is NOT RECOMMENDED to use
instrumentation-time constants or names of types representing the workflow,
such as "StateGraph". When no meaningful, low-cardinality workflow name is
available for a given framework, this attribute MUST NOT be captured by default.

Semantic conventions for individual Generative AI frameworks SHOULD document
what `gen_ai.workflow.name` means in the context of that framework."""


class GenAIOperationName(str, Enum):
    """Known values for `gen_ai.operation.name`."""

    CHAT = "chat"
    """Chat completion operation such as [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat)"""
    GENERATE_CONTENT = "generate_content"
    """Multimodal content generation operation such as [Gemini Generate Content](https://ai.google.dev/api/generate-content)"""
    TEXT_COMPLETION = "text_completion"
    """Text completions operation such as [OpenAI Completions API (Legacy)](https://platform.openai.com/docs/api-reference/completions)"""
    EMBEDDINGS = "embeddings"
    """Embeddings operation such as [OpenAI Create embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create)"""
    RETRIEVAL = "retrieval"
    """Retrieval operation such as [OpenAI Search Vector Store API](https://platform.openai.com/docs/api-reference/vector-stores/search)"""
    FETCH_RESPONSE = "fetch_response"
    """Fetch a previously generated model response by its identifier, without performing inference, such as [OpenAI Get a model response](https://platform.openai.com/docs/api-reference/responses/get)
    Instrumentations SHOULD NOT report token usage (as attributes or metrics) for this operation."""
    CREATE_AGENT = "create_agent"
    """Create GenAI agent"""
    INVOKE_AGENT = "invoke_agent"
    """Invoke GenAI agent"""
    EXECUTE_TOOL = "execute_tool"
    """Execute a tool"""
    INVOKE_WORKFLOW = "invoke_workflow"
    """Invoke GenAI workflow"""
    PLAN = "plan"
    """Agent planning or task decomposition phase"""
    SEARCH_MEMORY = "search_memory"
    """Search/query memories from a memory store"""
    CREATE_MEMORY = "create_memory"
    """Create new memory records"""
    UPDATE_MEMORY = "update_memory"
    """Update existing memory records"""
    UPSERT_MEMORY = "upsert_memory"
    """Create or update memory records without the caller choosing which"""
    DELETE_MEMORY = "delete_memory"
    """Delete memory records"""
    CREATE_MEMORY_STORE = "create_memory_store"
    """Create or initialize a memory store"""
    DELETE_MEMORY_STORE = "delete_memory_store"
    """Delete or deprovision a memory store"""


class GenAIOutputType(str, Enum):
    """Known values for `gen_ai.output.type`."""

    TEXT = "text"
    """Plain text"""
    JSON = "json"
    """JSON object with known or unknown schema"""
    IMAGE = "image"
    """Image"""
    SPEECH = "speech"
    """Speech"""


class GenAIProviderName(str, Enum):
    """Known values for `gen_ai.provider.name`."""

    OPENAI = "openai"
    """[OpenAI](https://openai.com/)"""
    GCP_GEN_AI = "gcp.gen_ai"
    """Any Google generative AI endpoint
    May be used when specific backend is unknown."""
    GCP_VERTEX_AI = "gcp.vertex_ai"
    """[Vertex AI](https://cloud.google.com/vertex-ai)
    Used when accessing the 'aiplatform.googleapis.com' endpoint."""
    GCP_GEMINI = "gcp.gemini"
    """[Gemini](https://cloud.google.com/products/gemini)
    Used when accessing the 'generativelanguage.googleapis.com' endpoint. Also known as the AI Studio API."""
    ANTHROPIC = "anthropic"
    """[Anthropic](https://www.anthropic.com/)"""
    COHERE = "cohere"
    """[Cohere](https://cohere.com/)"""
    AZURE_AI_INFERENCE = "azure.ai.inference"
    """Azure AI Inference"""
    AZURE_AI_OPENAI = "azure.ai.openai"
    """[Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)"""
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    """[IBM Watsonx AI](https://www.ibm.com/products/watsonx-ai)"""
    AWS_BEDROCK = "aws.bedrock"
    """[AWS Bedrock](https://aws.amazon.com/bedrock)"""
    PERPLEXITY = "perplexity"
    """[Perplexity](https://www.perplexity.ai/)"""
    X_AI = "x_ai"
    """[xAI](https://x.ai/)"""
    DEEPSEEK = "deepseek"
    """[DeepSeek](https://www.deepseek.com/)"""
    GROQ = "groq"
    """[Groq](https://groq.com/)"""
    MISTRAL_AI = "mistral_ai"
    """[Mistral AI](https://mistral.ai/)"""
    MOONSHOT_AI = "moonshot_ai"
    """[Moonshot AI](https://www.moonshot.ai/)"""


class GenAIResponseStatus(str, Enum):
    """Known values for `gen_ai.response.status`."""

    QUEUED = "queued"
    """The response has been accepted by the provider but generation has not started yet."""
    IN_PROGRESS = "in_progress"
    """The response is still being generated, for example a background or streamed response that has not finished."""
    COMPLETED = "completed"
    """The response finished generating successfully."""
    INCOMPLETE = "incomplete"
    """The response stopped before generation completed, for example because a token limit or content filter was reached."""
    FAILED = "failed"
    """The response generation failed with an error."""
    CANCELLED = "cancelled"
    """The response generation was cancelled before it completed."""


class GenAITokenType(str, Enum):
    """Known values for `gen_ai.token.type`."""

    INPUT = "input"
    """Input tokens (prompt, input, etc.)"""
    OUTPUT = "output"
    """Output tokens (completion, response, etc.)"""


GenAiOperationName = GenAIOperationName
GenAiOutputType = GenAIOutputType
GenAiProviderName = GenAIProviderName
GenAiResponseStatus = GenAIResponseStatus
GenAiTokenType = GenAITokenType

__all__ = [
    "GEN_AI_AGENT_DESCRIPTION",
    "GEN_AI_AGENT_ID",
    "GEN_AI_AGENT_NAME",
    "GEN_AI_AGENT_VERSION",
    "GEN_AI_CONVERSATION_COMPACTED",
    "GEN_AI_CONVERSATION_ID",
    "GEN_AI_DATA_SOURCE_ID",
    "GEN_AI_EMBEDDINGS_DIMENSION_COUNT",
    "GEN_AI_EVALUATION_EXPLANATION",
    "GEN_AI_EVALUATION_NAME",
    "GEN_AI_EVALUATION_SCORE_LABEL",
    "GEN_AI_EVALUATION_SCORE_VALUE",
    "GEN_AI_INPUT_MESSAGES",
    "GEN_AI_MEMORY_QUERY_TEXT",
    "GEN_AI_MEMORY_RECORDS",
    "GEN_AI_MEMORY_RECORD_COUNT",
    "GEN_AI_MEMORY_RECORD_ID",
    "GEN_AI_MEMORY_STORE_ID",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_OUTPUT_MESSAGES",
    "GEN_AI_OUTPUT_TYPE",
    "GEN_AI_PROMPT_NAME",
    "GEN_AI_PROMPT_VARIABLE",
    "GEN_AI_PROMPT_VERSION",
    "GEN_AI_PROVIDER_NAME",
    "GEN_AI_REQUEST_CHOICE_COUNT",
    "GEN_AI_REQUEST_ENCODING_FORMATS",
    "GEN_AI_REQUEST_FREQUENCY_PENALTY",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_PRESENCE_PENALTY",
    "GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID",
    "GEN_AI_REQUEST_REASONING_LEVEL",
    "GEN_AI_REQUEST_SEED",
    "GEN_AI_REQUEST_STOP_SEQUENCES",
    "GEN_AI_REQUEST_STREAM",
    "GEN_AI_REQUEST_STREAM_CURSOR",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_RESPONSE_STATUS",
    "GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK",
    "GEN_AI_RETRIEVAL_DOCUMENTS",
    "GEN_AI_RETRIEVAL_QUERY_TEXT",
    "GEN_AI_RETRIEVAL_TOP_K",
    "GEN_AI_SYSTEM_INSTRUCTIONS",
    "GEN_AI_TOKEN_TYPE",
    "GEN_AI_TOOL_CALL_ARGUMENTS",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_CALL_RESULT",
    "GEN_AI_TOOL_DEFINITIONS",
    "GEN_AI_TOOL_DESCRIPTION",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_TOOL_TYPE",
    "GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_AUDIO_INPUT_TOKENS",
    "GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS",
    "GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_IMAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_REASONING_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS",
    "GEN_AI_USAGE_TEXT_INPUT_TOKENS",
    "GEN_AI_USAGE_TEXT_OUTPUT_TOKENS",
    "GEN_AI_WORKFLOW_NAME",
    "GenAIOperationName",
    "GenAIOutputType",
    "GenAIProviderName",
    "GenAIResponseStatus",
    "GenAITokenType",
    "GenAiOperationName",
    "GenAiOutputType",
    "GenAiProviderName",
    "GenAiResponseStatus",
    "GenAiTokenType",
]
