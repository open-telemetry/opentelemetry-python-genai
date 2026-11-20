# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal media input capture in Agno instrumentation."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from agno.agent import Agent
from agno.media import Audio, File, Image, Video
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.team import Team
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace.status import StatusCode


def _run_agent(
    agent: Agent,
    *,
    is_async: bool = False,
    stream: bool = False,
    sent_image_urls: list[str] | None = None,
    **run_kwargs: Any,
) -> None:
    def _record(*args: Any, **kwargs: Any) -> None:
        if sent_image_urls is None:
            return
        messages = kwargs.get("messages") or (args[0] if args else ())
        for msg in messages:
            if getattr(msg, "images", None):
                sent_image_urls.extend(
                    img.url for img in msg.images if img.url
                )

    def _sync_resp(*args: Any, **kwargs: Any) -> ModelResponse:
        _record(*args, **kwargs)
        return ModelResponse(content="ok")

    async def _async_resp(*args: Any, **kwargs: Any) -> ModelResponse:
        _record(*args, **kwargs)
        return ModelResponse(content="ok")

    def _sync_stream(*args: Any, **kwargs: Any):
        _record(*args, **kwargs)
        yield ModelResponse(content="ok")

    async def _async_stream(*args: Any, **kwargs: Any):
        _record(*args, **kwargs)
        yield ModelResponse(content="ok")

    if is_async and stream:

        async def _consume() -> None:
            with patch(
                "agno.models.base.Model.aresponse_stream",
                side_effect=_async_stream,
            ):
                async for _ in agent.arun(stream=True, **run_kwargs):
                    pass

        asyncio.run(_consume())
    elif is_async:
        with patch(
            "agno.models.base.Model.aresponse", side_effect=_async_resp
        ):
            asyncio.run(agent.arun(**run_kwargs))
    elif stream:
        with patch(
            "agno.models.base.Model.response_stream", side_effect=_sync_stream
        ):
            list(agent.run(stream=True, **run_kwargs))
    else:
        with patch("agno.models.base.Model.response", side_effect=_sync_resp):
            agent.run(**run_kwargs)


def _input_parts(span_exporter: Any) -> tuple[dict[str, Any], ...]:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    messages = spans[0].attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert messages is not None and len(messages) == 1
    assert messages[0]["role"] == "user"
    return messages[0]["parts"]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_agent_captures_url_media_parts(
    instrument_agno_content_capture,
    span_exporter,
    is_async: bool,
    stream: bool,
) -> None:
    """Agent.run/arun (sync, async, streaming) capture URL media across all four modalities."""
    agent = Agent(name="media-agent", model=MockModel(id="mock-model"))
    _run_agent(
        agent,
        is_async=is_async,
        stream=stream,
        input="Describe these",
        images=[Image(url="https://example.com/cat.png")],
        audio=[
            Audio(url="https://example.com/clip.mp3", mime_type="audio/mpeg")
        ],
        videos=[Video(url="https://example.com/clip.mp4")],
        files=[
            File(
                url="https://example.com/doc.pdf", mime_type="application/pdf"
            )
        ],
    )

    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Describe these"},
        {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/cat.png",
        },
        {
            "type": "uri",
            "modality": "audio",
            "mime_type": "audio/mpeg",
            "uri": "https://example.com/clip.mp3",
        },
        {
            "type": "uri",
            "modality": "video",
            "uri": "https://example.com/clip.mp4",
        },
        {
            "type": "uri",
            "modality": "document",
            "mime_type": "application/pdf",
            "uri": "https://example.com/doc.pdf",
        },
    )


def test_agent_captures_blob_data_url_and_filepath_media(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Raw bytes, data URLs, format/filename MIME types, and filepaths map cleanly."""
    agent = Agent(name="blob-agent", model=MockModel(id="mock-model"))
    jpeg_b64 = base64.b64encode(b"\xff\xd8\xff").decode()
    pdf_b64 = base64.b64encode(b"%PDF-1.7").decode()

    _run_agent(
        agent,
        input="Inspect",
        images=[
            Image(content=b"\x89PNG", mime_type="image/png"),
            Image(url=f"data:image/jpeg;base64,{jpeg_b64}"),
            Image(url="data:image/png;base64,!!!not-base64!!!"),
            Image(filepath=Path("/nonexistent/path/cat.png")),
        ],
        audio=[
            Audio(content=b"ID3", format="mp3"),
            Audio(content=b"RIFF", format="wav"),
        ],
        videos=[Video(content=b"\x00\x00\x00\x18", format="video/mp4")],
        files=[
            File(content=b"%PDF-1.4", filename="report.pdf"),
            File(content="a,b\n1,2", mime_type="text/csv"),
            File(url=f"data:application/pdf;base64,{pdf_b64}"),
        ],
    )

    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Inspect"},
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": b"\x89PNG",
        },
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/jpeg",
            "content": b"\xff\xd8\xff",
        },
        {
            "type": "uri",
            "modality": "image",
            "uri": Path("/nonexistent/path/cat.png").absolute().as_uri(),
        },
        {
            "type": "blob",
            "modality": "audio",
            "mime_type": "audio/mpeg",
            "content": b"ID3",
        },
        {
            "type": "blob",
            "modality": "audio",
            "mime_type": "audio/wav",
            "content": b"RIFF",
        },
        {
            "type": "blob",
            "modality": "video",
            "mime_type": "video/mp4",
            "content": b"\x00\x00\x00\x18",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": b"%PDF-1.4",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "text/csv",
            "content": b"a,b\n1,2",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": b"%PDF-1.7",
        },
    )


@pytest.mark.parametrize(
    ("run_kwargs", "expected_parts"),
    [
        (
            {
                "input": None,
                "images": [Image(url="https://example.com/only.png")],
            },
            (
                {
                    "type": "uri",
                    "modality": "image",
                    "uri": "https://example.com/only.png",
                },
            ),
        ),
        (
            {
                "input": "",
                "images": [Image(url="https://example.com/only.png")],
            },
            (
                {
                    "type": "uri",
                    "modality": "image",
                    "uri": "https://example.com/only.png",
                },
            ),
        ),
        (
            {
                "input": Message(
                    role="user",
                    content="Look",
                    images=[Image(url="https://example.com/msg.png")],
                )
            },
            (
                {"type": "text", "content": "Look"},
                {
                    "type": "uri",
                    "modality": "image",
                    "uri": "https://example.com/msg.png",
                },
            ),
        ),
        (
            {
                "input": Message(
                    role="user",
                    content=None,
                    images=[Image(url="https://example.com/msg.png")],
                )
            },
            (
                {
                    "type": "uri",
                    "modality": "image",
                    "uri": "https://example.com/msg.png",
                },
            ),
        ),
    ],
)
def test_agent_media_without_text_and_message_inputs(
    instrument_agno_content_capture,
    span_exporter,
    run_kwargs: dict[str, Any],
    expected_parts: tuple[dict[str, Any], ...],
) -> None:
    """Media-only inputs and Message inputs with attached media omit empty text parts."""
    agent = Agent(name="msg-media-agent", model=MockModel(id="mock-model"))
    _run_agent(agent, **run_kwargs)
    assert _input_parts(span_exporter) == expected_parts


@pytest.mark.parametrize("is_async", [False, True])
def test_agent_media_omitted_when_content_capture_disabled(
    instrument_agno,
    span_exporter,
    is_async: bool,
) -> None:
    """Media input reaches Agno untouched and is not recorded when content capture is disabled."""
    agent = Agent(name="no-capture-agent", model=MockModel(id="mock-model"))
    sent_urls: list[str] = []
    _run_agent(
        agent,
        is_async=is_async,
        sent_image_urls=sent_urls,
        input="Describe",
        images=[Image(url="https://example.com/cat.png")],
        files=[File(content=b"%PDF", mime_type="application/pdf")],
    )

    assert sent_urls == ["https://example.com/cat.png"]
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in spans[0].attributes


def test_agent_agno3_file_id_external_and_media_reference(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Agno 3.x File(id=...), MediaReference, and File(external=...) map cleanly."""
    MediaReference = pytest.importorskip("agno.media.reference").MediaReference
    agent = Agent(name="agno3-media-agent", model=MockModel(id="mock-model"))

    _run_agent(
        agent,
        input="Agno 3 media",
        images=[
            Image(
                media_reference=MediaReference(
                    media_id="m1",
                    storage_key="k1",
                    storage_backend="s3",
                    url="https://s3.example.com/cat.png",
                    mime_type="image/png",
                )
            ),
            Image(
                media_reference=MediaReference(
                    media_id="m2", storage_key="k2", storage_backend="s3"
                )
            ),
        ],
        files=[
            File(id="file-abc123", mime_type="application/pdf"),
            File(external=object()),
            File(
                media_reference=MediaReference(
                    media_id="m3", storage_key="k3", storage_backend="s3"
                )
            ),
        ],
    )

    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Agno 3 media"},
        {
            "type": "uri",
            "modality": "image",
            "mime_type": "image/png",
            "uri": "https://s3.example.com/cat.png",
        },
        {
            "type": "file",
            "modality": "document",
            "mime_type": "application/pdf",
            "file_id": "file-abc123",
        },
    )


def test_agent_run_error_preserves_media_input(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Failed Agent.run still records media inputs and sets error.type."""
    agent = Agent(name="err-media-agent", model=MockModel(id="mock-model"))

    with (
        patch.object(
            Agent, "initialize_agent", side_effect=RuntimeError("model boom")
        ),
        pytest.raises(RuntimeError, match="model boom"),
    ):
        agent.run(
            input="Fail with media",
            images=[Image(url="https://example.com/err.png")],
            files=[File(url="https://example.com/err.pdf")],
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert (
        spans[0].attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"
    )
    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Fail with media"},
        {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/err.png",
        },
        {
            "type": "uri",
            "modality": "document",
            "uri": "https://example.com/err.pdf",
        },
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_team_run_captures_media_parts(
    instrument_agno_content_capture,
    span_exporter,
    is_async: bool,
) -> None:
    """Team.run and Team.arun capture attached media parts."""
    member = Agent(name="member", model=MockModel(id="mock-model"))
    team = Team(
        name="media-team", members=[member], model=MockModel(id="mock-model")
    )
    kwargs: dict[str, Any] = {
        "input": "Team describe",
        "images": [Image(url="https://example.com/team.png")],
        "files": [File(url="https://example.com/team.pdf")],
    }
    mock_output = ModelResponse(content="team ok")
    if is_async:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            asyncio.run(team.arun(**kwargs))
    else:
        with patch(
            "agno.models.base.Model.response", return_value=mock_output
        ):
            team.run(**kwargs)

    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Team describe"},
        {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/team.png",
        },
        {
            "type": "uri",
            "modality": "document",
            "uri": "https://example.com/team.pdf",
        },
    )


@pytest.mark.parametrize("mode", ["sync", "async", "background"])
def test_workflow_captures_media_parts(
    instrument_agno_content_capture,
    span_exporter,
    mode: str,
) -> None:
    """Workflow.run, Workflow.arun, and background Workflow.arun capture media parts."""
    pytest.importorskip("fastapi")
    from agno.workflow.workflow import Workflow

    async def step(step_input: Any) -> str:
        return "done"

    workflow = Workflow(
        name="media-workflow",
        steps=[step] if mode == "background" else [],
        telemetry=False,
    )
    kwargs: dict[str, Any] = {
        "input": "Workflow describe",
        "images": [Image(url="https://example.com/wf.png")],
        "files": [File(content=b"hello", mime_type="text/plain")],
    }

    if mode == "sync":
        workflow.run(**kwargs)
    elif mode == "async":
        asyncio.run(workflow.arun(**kwargs))
    else:

        async def _run_bg() -> None:
            await workflow.arun(background=True, **kwargs)
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            if pending:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=5.0,
                )

        asyncio.run(_run_bg())

    assert _input_parts(span_exporter) == (
        {"type": "text", "content": "Workflow describe"},
        {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/wf.png",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "text/plain",
            "content": b"hello",
        },
    )
