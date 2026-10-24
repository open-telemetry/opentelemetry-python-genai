# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Public re-export of all GenAI invocation types.

Users can import everything from this single module:

    from opentelemetry.util.genai.invocation import (
        Error,
        GenAIInvocation,
        FetchResponseInvocation,
        InferenceInvocation,
        EmbeddingInvocation,
        RetrievalInvocation,
        ToolInvocation,
        WorkflowInvocation,
    )
"""

from opentelemetry.util.genai._agent_invocation import (
    AgentInvocation,
    LocalAgentInvocation,
    RemoteAgentInvocation,
)
from opentelemetry.util.genai._discovery_invocation import (
    DiscoveryInvocation,
    MCPListToolsInvocation,
)
from opentelemetry.util.genai._embedding_invocation import (
    CLIENT_EMBEDDING_CONTEXT_KEY,
    EmbeddingData,
    EmbeddingInvocation,
)
from opentelemetry.util.genai._fetch_response_invocation import (
    CLIENT_FETCH_RESPONSE_CONTEXT_KEY,
    FetchResponseData,
    FetchResponseInvocation,
)
from opentelemetry.util.genai._inference_invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    InferenceData,
    InferenceInvocation,
)
from opentelemetry.util.genai._invocation import (
    ContextToken,
    Error,
    GenAIInvocation,
)
from opentelemetry.util.genai._retrieval_invocation import (
    CLIENT_RETRIEVAL_CONTEXT_KEY,
    RetrievalData,
    RetrievalInvocation,
)
from opentelemetry.util.genai._tool_invocation import (
    ToolInvocation,
)
from opentelemetry.util.genai._workflow_invocation import (
    WorkflowInvocation,
)

__all__ = [
    "CLIENT_EMBEDDING_CONTEXT_KEY",
    "CLIENT_FETCH_RESPONSE_CONTEXT_KEY",
    "CLIENT_INFERENCE_CONTEXT_KEY",
    "CLIENT_RETRIEVAL_CONTEXT_KEY",
    "AgentInvocation",
    "ContextToken",
    "DiscoveryInvocation",
    "EmbeddingData",
    "EmbeddingInvocation",
    "Error",
    "FetchResponseData",
    "FetchResponseInvocation",
    "GenAIInvocation",
    "InferenceData",
    "InferenceInvocation",
    "LocalAgentInvocation",
    "MCPListToolsInvocation",
    "RemoteAgentInvocation",
    "RetrievalData",
    "RetrievalInvocation",
    "ToolInvocation",
    "WorkflowInvocation",
]
