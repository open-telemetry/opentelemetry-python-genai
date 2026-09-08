# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain Responses API input_image (ChatOpenAI)."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

_REAL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3X"
    "QQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAg"
    "ICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="
)

ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=100,
    use_responses_api=True,
    include_response_headers=True,
).invoke(
    [
        HumanMessage(
            content=[
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
                },
            ]
        ),
    ]
)
