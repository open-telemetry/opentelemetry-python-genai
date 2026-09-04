# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import unittest

from google.genai import types as genai_types

from opentelemetry.instrumentation.google_genai.message import (
    to_system_instructions,
)
from opentelemetry.util.genai.types import GenericPart, TextPart


class TestGoogleGenAiMessage(unittest.TestCase):
    def test_to_system_instructions_text(self):
        content = genai_types.Content(
            parts=[genai_types.Part.from_text(text="Be concise")]
        )
        instructions = to_system_instructions(content=content)
        self.assertEqual(instructions, [TextPart(content="Be concise")])

    def test_to_system_instructions_non_text_parts(self):
        content = genai_types.Content(
            parts=[
                genai_types.Part.from_text(text="Be concise"),
                genai_types.Part.from_bytes(
                    data=b"image_bytes", mime_type="image/png"
                ),
                genai_types.Part.from_uri(
                    file_uri="gs://bucket/doc.pdf",
                    mime_type="application/pdf",
                ),
            ]
        )
        instructions = to_system_instructions(content=content)
        self.assertEqual(
            instructions,
            [
                TextPart(content="Be concise"),
                GenericPart(type="blob"),
                GenericPart(type="uri"),
            ],
        )

    def test_to_system_instructions_empty_or_unknown_part(self):
        content = genai_types.Content(parts=[genai_types.Part()])
        instructions = to_system_instructions(content=content)
        self.assertEqual(instructions, [])
