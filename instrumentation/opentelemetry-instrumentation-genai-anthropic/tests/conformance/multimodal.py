# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic messages carrying non-text content."""

from anthropic import Anthropic

Anthropic().messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these images."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "QUJD",
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/image.png",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "QUJD",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/document.pdf",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Document text",
                    },
                },
                {
                    "type": "document",
                    "title": "Reference",
                    "context": "Use the nested content.",
                    "citations": {"enabled": True},
                    "source": {
                        "type": "content",
                        "content": [
                            {"type": "text", "text": "Nested text"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": "https://example.com/nested.png",
                                },
                            },
                        ],
                    },
                },
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": "file-image"},
                },
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "file-document"},
                },
            ],
        }
    ],
)
