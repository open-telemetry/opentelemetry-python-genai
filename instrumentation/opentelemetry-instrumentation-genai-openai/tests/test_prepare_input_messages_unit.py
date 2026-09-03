# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for input message preparation, including multimodal content."""

import base64
from dataclasses import asdict
from datetime import datetime, timezone
from types import MappingProxyType

from opentelemetry.instrumentation.genai.openai.utils import (
    _prepare_input_messages,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    GenericPart,
    TextPart,
    UriPart,
)
from opentelemetry.util.genai.utils import gen_ai_json_dumps

IMAGE_URL = "https://example.com/cat.png"
# 1x1 transparent PNG.
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = f"data:image/png;base64,{PNG_BASE64}"
PDF_BASE64 = "JVBERi0="  # b"%PDF-"


def _user_parts(content):
    messages = [{"role": "user", "content": content}]
    return _prepare_input_messages(messages)[0].parts


def test_string_content_maps_to_text_part():
    assert _user_parts("hello") == [TextPart(content="hello")]


def test_image_url_maps_to_uri_part():
    parts = _user_parts(
        [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": IMAGE_URL, "detail": "low"},
            },
        ]
    )

    assert parts == [
        TextPart(content="What is in this image?"),
        UriPart(mime_type=None, modality="image", uri=IMAGE_URL),
    ]


def test_image_data_url_maps_to_blob_part():
    parts = _user_parts(
        [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]
    )

    assert parts == [
        BlobPart(
            mime_type="image/png",
            modality="image",
            content=base64.b64decode(PNG_BASE64),
        )
    ]


def test_input_audio_part_is_not_captured():
    # Inline audio can be megabytes per part; it is deliberately left out of
    # telemetry while the surrounding parts are still captured.
    parts = _user_parts(
        [
            {"type": "text", "text": "transcribe this"},
            {
                "type": "input_audio",
                "input_audio": {"data": "UklGRg==", "format": "wav"},
            },
        ]
    )

    assert parts == [TextPart(content="transcribe this")]


def test_file_id_maps_to_file_part():
    parts = _user_parts(
        [
            {
                "type": "file",
                "file": {"file_id": "file-abc", "filename": "a.pdf"},
            }
        ]
    )

    assert parts == [
        FilePart(
            mime_type="application/pdf",
            modality="document",
            file_id="file-abc",
        )
    ]


def test_inline_file_data_is_not_captured():
    # Inline documents can be megabytes per part; they are deliberately left
    # out of telemetry while the surrounding parts are still captured.
    parts = _user_parts(
        [
            {"type": "text", "text": "summarize this"},
            {
                "type": "file",
                "file": {"filename": "doc.pdf", "file_data": PDF_BASE64},
            },
        ]
    )

    assert parts == [TextPart(content="summarize this")]


def test_unknown_part_type_is_preserved_as_generic_part():
    item = {"type": "custom_widget", "custom_widget": {"id": 7}}

    assert _user_parts([item]) == [
        GenericPart(type="custom_widget", value=item)
    ]


def test_generic_part_value_is_json_safe():
    item = {
        "type": "custom",
        "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "tags": {"a"},
        "nested": {"n": (1, 2)},
    }

    parts = _user_parts([item])

    assert parts == [
        GenericPart(
            type="custom",
            value={
                "type": "custom",
                "when": "2026-01-01 00:00:00+00:00",
                "tags": "{'a'}",
                "nested": {"n": [1, 2]},
            },
        )
    ]
    gen_ai_json_dumps(asdict(parts[0]))


def test_generic_part_value_does_not_alias_input():
    item = {"type": "custom", "nested": {"n": 1}}

    parts = _user_parts([item])
    item["nested"]["n"] = 2

    assert parts[0].value == {"type": "custom", "nested": {"n": 1}}


def test_untyped_part_is_dropped():
    assert _user_parts([{"unexpected": "shape"}]) == []


def test_non_string_text_part_is_dropped():
    assert _user_parts([{"type": "text", "text": {"a": 1}}]) == []


def test_raising_part_is_skipped_without_breaking_the_message():
    class ExplodingPart:
        @property
        def type(self):
            raise RuntimeError("boom")

    parts = _user_parts(
        [ExplodingPart(), {"type": "text", "text": "still captured"}]
    )

    assert parts == [TextPart(content="still captured")]


def test_multimodal_assistant_history_content():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is a cat."}],
        }
    ]

    result = _prepare_input_messages(messages)

    assert result[0].parts == [TextPart(content="It is a cat.")]


def test_none_content_produces_no_parts():
    messages = [{"role": "assistant", "content": None}]

    result = _prepare_input_messages(messages)

    assert result[0].parts == []


def test_string_list_content_maps_each_item_to_text_part():
    assert _user_parts(["a", "b"]) == [
        TextPart(content="a"),
        TextPart(content="b"),
    ]


def test_tuple_content_is_treated_as_content_part_array():
    assert _user_parts(({"type": "text", "text": "hello"},)) == [
        TextPart(content="hello")
    ]


def test_mapping_content_is_treated_as_single_part():
    parts = _user_parts({"type": "text", "text": "hello"})

    assert parts == [TextPart(content="hello")]


def test_non_dict_mapping_content_is_treated_as_single_part():
    content = MappingProxyType({"type": "text", "text": "hello"})

    assert _user_parts(content) == [TextPart(content="hello")]


def test_generator_content_is_left_unconsumed_and_not_captured():
    # The OpenAI SDK accepts any iterable of content parts and materializes
    # it itself; capturing it here would drain the caller's input before the
    # request is sent.
    def content_parts():
        yield {"type": "text", "text": "hello"}
        yield {"type": "image_url", "image_url": {"url": IMAGE_URL}}

    content = content_parts()

    assert _user_parts(content) == []
    assert len(list(content)) == 2


def test_multimodal_message_is_json_serializable():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ],
        }
    ]

    result = _prepare_input_messages(messages)

    serialized = gen_ai_json_dumps([asdict(message) for message in result])
    assert IMAGE_URL in serialized
    assert '"type": "uri"' in serialized or '"type":"uri"' in serialized
    assert '"type": "blob"' in serialized or '"type":"blob"' in serialized
