# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain Files API image reference (ChatAnthropic)."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

_ANTHROPIC_FILE_ID = "file_011CNhaGCM5eyZmDsFmQJVQe"

ChatAnthropic(
    model="claude-sonnet-4-5", temperature=0.1, max_tokens=1024
).invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "file",
                        "file_id": _ANTHROPIC_FILE_ID,
                    },
                },
            ]
        ),
    ]
)
