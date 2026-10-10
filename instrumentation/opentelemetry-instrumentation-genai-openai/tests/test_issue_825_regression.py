"""Regression test for issue #825: instrumented wrapper materializes generators."""
import os
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")

from unittest.mock import MagicMock, AsyncMock, patch

# Import the instrumented wrappers (production code unchanged)
from opentelemetry.instrumentation.genai.openai.patch import (
    chat_completions_create_v_new,
    async_chat_completions_create_v_new,
)

def test_sync_wrapper_materializes_generator_messages():
    mock_create = MagicMock(return_value={"choices": []})
    handler = MagicMock()
    handler.should_capture_content.return_value = False
    wrapper = chat_completions_create_v_new(handler)
    generator = (msg for msg in [{"role":"user","content":"hi"}])
    wrapper(mock_create, None, [], {"messages": generator})
    call_kwargs = mock_create.call_args.kwargs
    assert isinstance(call_kwargs["messages"], list)
    assert call_kwargs["messages"] == [{"role":"user","content":"hi"}]

def test_sync_wrapper_materializes_generator_tools():
    mock_create = MagicMock(return_value={"choices": []})
    handler = MagicMock()
    handler.should_capture_content.return_value = False
    wrapper = chat_completions_create_v_new(handler)
    generator = (t for t in [{"type":"function","function":{"name":"f"}}])
    wrapper(mock_create, None, [], {"messages": [{"role":"user","content":"hi"}], "tools": generator})
    call_kwargs = mock_create.call_args.kwargs
    assert isinstance(call_kwargs["tools"], list)
    assert call_kwargs["tools"] == [{"type":"function","function":{"name":"f"}}]

def test_async_wrapper_materializes_generator_messages():
    import asyncio
    mock_create = AsyncMock(return_value={"choices": []})
    handler = MagicMock()
    handler.should_capture_content.return_value = False
    wrapper = async_chat_completions_create_v_new(handler)
    async def run():
        generator = (msg for msg in [{"role":"user","content":"hi"}])
        await wrapper(mock_create, None, [], {"messages": generator})
    asyncio.run(run())
    call_kwargs = mock_create.call_args.kwargs
    assert isinstance(call_kwargs["messages"], list)
    assert call_kwargs["messages"] == [{"role":"user","content":"hi"}]

def test_async_wrapper_materializes_generator_tools():
    import asyncio
    mock_create = AsyncMock(return_value={"choices": []})
    handler = MagicMock()
    handler.should_capture_content.return_value = False
    wrapper = async_chat_completions_create_v_new(handler)
    async def run():
        generator = (t for t in [{"type":"function","function":{"name":"f"}}])
        await wrapper(mock_create, None, [], {"messages": [{"role":"user","content":"hi"}], "tools": generator})
    asyncio.run(run())
    call_kwargs = mock_create.call_args.kwargs
    assert isinstance(call_kwargs["tools"], list)
    assert call_kwargs["tools"] == [{"type":"function","function":{"name":"f"}}]
