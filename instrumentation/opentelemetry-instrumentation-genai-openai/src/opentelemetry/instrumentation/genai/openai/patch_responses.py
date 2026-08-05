# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Union, cast

from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import FetchResponseInvocation

from ._raw_response import wrap_stream_result
from .response_extractors import (
    apply_request_attributes,
    extract_params,
    get_fetch_response_creation_kwargs,
    get_inference_creation_kwargs,
    get_response_error,
    is_streamed_raw_response,
    set_fetch_response_attributes,
    set_invocation_response_attributes,
)
from .response_wrappers import (
    AsyncFetchResponseStreamWrapper,
    AsyncResponseStreamManagerWrapper,
    AsyncResponseStreamWrapper,
    FetchResponseStreamWrapper,
    ResponseStreamManagerWrapper,
    ResponseStreamWrapper,
    responses_stream_context,
)
from .utils import is_streaming, value_is_set

if TYPE_CHECKING:
    from openai import AsyncStream as OpenAIAsyncStream
    from openai import Stream as OpenAIStream
    from openai.lib.streaming.responses._responses import (  # pylint: disable=no-name-in-module
        AsyncResponseStream,
        AsyncResponseStreamManager,
        ResponseStream,
        ResponseStreamManager,
    )
    from openai.resources.responses.responses import AsyncResponses, Responses
    from openai.types.responses import (  # pylint: disable=no-name-in-module
        ParsedResponse,
        Response,
    )

ResponseResult = Union["ParsedResponse[Any]", "Response"]
ResponseStreamResult = Union["OpenAIStream[Any]", "ResponseStream[Any]"]
AsyncResponseStreamResult = Union[
    "OpenAIAsyncStream[Any]", "AsyncResponseStream[Any]"
]


def responses_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    ResponseResult | ResponseStreamResult | ResponseStreamWrapper[Any],
]:
    """Wrap ``Responses.create`` to trace Responses API calls.

    Traces :meth:`openai.resources.responses.responses.Responses.create`.
    OpenAI SDK source:
    https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py#L914
    """

    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., ResponseResult | ResponseStreamResult],
        instance: Responses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ResponseResult | ResponseStreamResult | ResponseStreamWrapper[Any]:
        stream_context = responses_stream_context.get()
        if stream_context is not None:
            # Called by the Responses.stream() manager: it owns telemetry, so
            # return the SDK stream unwrapped without a second invocation.
            return wrapped(*args, **kwargs)

        params = extract_params(**kwargs)
        invocation = handler.inference(
            **get_inference_creation_kwargs(params, instance)
        )
        apply_request_attributes(invocation, params, capture_content)

        try:
            result = wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return wrap_stream_result(
                    ResponseStreamWrapper,
                    result,
                    invocation,
                    capture_content,
                )

            set_invocation_response_attributes(
                invocation, result, capture_content, kwargs
            )
            error = get_response_error(result, kwargs)
            if error is not None:
                invocation.fail(error)
            else:
                invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., "ResponseResult" | "ResponseStreamResult" | ResponseStreamWrapper[Any]]',
        traced_method,
    )


def async_responses_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Awaitable[
        ResponseResult
        | AsyncResponseStreamResult
        | AsyncResponseStreamWrapper[Any]
    ],
]:
    """Wrap ``AsyncResponses.create`` to trace async Responses API calls.

    Traces :meth:`openai.resources.responses.responses.AsyncResponses.create`.
    OpenAI SDK source:
    https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py#L2661
    """

    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[
            ...,
            Awaitable[ResponseResult | AsyncResponseStreamResult],
        ],
        instance: AsyncResponses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> (
        ResponseResult
        | AsyncResponseStreamResult
        | AsyncResponseStreamWrapper[Any]
    ):
        stream_context = responses_stream_context.get()
        if stream_context is not None:
            # Called by the Responses.stream() manager: it owns telemetry, so
            # return the SDK stream unwrapped without a second invocation.
            return await wrapped(*args, **kwargs)

        params = extract_params(**kwargs)
        invocation = handler.inference(
            **get_inference_creation_kwargs(params, instance)
        )
        apply_request_attributes(invocation, params, capture_content)

        try:
            result = await wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return wrap_stream_result(
                    AsyncResponseStreamWrapper,
                    result,
                    invocation,
                    capture_content,
                )

            set_invocation_response_attributes(
                invocation, result, capture_content, kwargs
            )
            error = get_response_error(result, kwargs)
            if error is not None:
                invocation.fail(error)
            else:
                invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., Awaitable["ResponseResult" | "AsyncResponseStreamResult" | AsyncResponseStreamWrapper[Any]]]',
        traced_method,
    )


def _get_retrieve_response_id(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    """Return the ``response_id`` a ``responses.retrieve`` call was made with."""
    response_id = args[0] if args else kwargs.get("response_id")
    return response_id if isinstance(response_id, str) else None


def _get_stream_cursor(kwargs: dict[str, Any]) -> str | None:
    """Return the ``starting_after`` cursor a streamed fetch resumes from."""
    starting_after = kwargs.get("starting_after")
    if not value_is_set(starting_after):
        return None
    return str(starting_after)


def _start_fetch_response_invocation(
    handler: TelemetryHandler,
    instance: Responses | AsyncResponses,
    response_id: str,
    kwargs: dict[str, Any],
) -> FetchResponseInvocation:
    invocation = handler.fetch_response(
        **get_fetch_response_creation_kwargs(response_id, instance),
        request_stream=(
            True
            if is_streaming(kwargs) or is_streamed_raw_response(kwargs)
            else None
        ),
    )
    invocation.stream_cursor = _get_stream_cursor(kwargs)
    invocation.attributes[OpenAIAttributes.OPENAI_API_TYPE] = (
        OpenAIAttributes.OpenaiApiTypeValues.RESPONSES.value
    )
    return invocation


def responses_retrieve(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Union[
        ResponseResult,
        ResponseStreamResult,
        FetchResponseStreamWrapper[Any],
    ],
]:
    """Wrap ``Responses.retrieve`` to trace fetching a stored response by id.

    Traces :meth:`openai.resources.responses.responses.Responses.retrieve`.
    Fetching a stored response performs no inference and consumes no tokens, so
    it is reported as a ``fetch_response`` operation rather than as an
    inference call.
    """

    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Union[ResponseResult, ResponseStreamResult]],
        instance: Responses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        ResponseResult,
        ResponseStreamResult,
        FetchResponseStreamWrapper[Any],
    ]:
        response_id = _get_retrieve_response_id(args, kwargs)
        if response_id is None:
            # gen_ai.response.id is required on a fetch_response span and the
            # SDK rejects the call without it, so leave it untraced.
            return wrapped(*args, **kwargs)

        invocation = _start_fetch_response_invocation(
            handler, instance, response_id, kwargs
        )

        try:
            result = wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return wrap_stream_result(
                    FetchResponseStreamWrapper,
                    result,
                    invocation,
                    capture_content,
                )

            set_fetch_response_attributes(
                invocation,
                result,
                capture_content,
                kwargs,
            )
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., Union["ResponseResult", "ResponseStreamResult", FetchResponseStreamWrapper[Any]]]',
        traced_method,
    )


def async_responses_retrieve(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Awaitable[
        Union[
            ResponseResult,
            AsyncResponseStreamResult,
            AsyncFetchResponseStreamWrapper[Any],
        ]
    ],
]:
    """Wrap ``AsyncResponses.retrieve`` to trace fetching a stored response by id.

    Traces :meth:`openai.resources.responses.responses.AsyncResponses.retrieve`.
    Fetching a stored response performs no inference and consumes no tokens, so
    it is reported as a ``fetch_response`` operation rather than as an
    inference call.
    """

    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[
            ...,
            Awaitable[Union[ResponseResult, AsyncResponseStreamResult]],
        ],
        instance: AsyncResponses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        ResponseResult,
        AsyncResponseStreamResult,
        AsyncFetchResponseStreamWrapper[Any],
    ]:
        response_id = _get_retrieve_response_id(args, kwargs)
        if response_id is None:
            # gen_ai.response.id is required on a fetch_response span and the
            # SDK rejects the call without it, so leave it untraced.
            return await wrapped(*args, **kwargs)

        invocation = _start_fetch_response_invocation(
            handler, instance, response_id, kwargs
        )

        try:
            result = await wrapped(*args, **kwargs)
            if is_streaming(kwargs):
                return wrap_stream_result(
                    AsyncFetchResponseStreamWrapper,
                    result,
                    invocation,
                    capture_content,
                )

            set_fetch_response_attributes(
                invocation, result, capture_content, kwargs
            )
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., Awaitable[Union["ResponseResult", "AsyncResponseStreamResult", AsyncFetchResponseStreamWrapper[Any]]]]',
        traced_method,
    )


def responses_stream(
    handler: TelemetryHandler,
) -> Callable[..., ResponseStreamManagerWrapper[Any]]:
    """Wrap ``Responses.stream`` to trace sync stream manager calls.

    Traces :meth:`openai.resources.responses.responses.Responses.stream`.
    OpenAI SDK source:
    https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py#L1062
    """

    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., ResponseStreamManager[Any]],
        instance: Responses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> ResponseStreamManagerWrapper[Any]:
        def invocation_factory() -> Any:
            params = extract_params(**kwargs)
            invocation = handler.inference(
                **get_inference_creation_kwargs(params, instance)
            )
            apply_request_attributes(invocation, params, capture_content)
            return invocation

        return ResponseStreamManagerWrapper(
            wrapped(*args, **kwargs),
            invocation_factory,
            capture_content,
        )

    return cast(
        "Callable[..., ResponseStreamManagerWrapper[Any]]", traced_method
    )


def async_responses_stream(
    handler: TelemetryHandler,
) -> Callable[..., AsyncResponseStreamManagerWrapper[Any]]:
    """Wrap ``AsyncResponses.stream`` to trace async stream manager calls.

    Traces :meth:`openai.resources.responses.responses.AsyncResponses.stream`.
    OpenAI SDK source:
    https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py#L2809
    """

    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., AsyncResponseStreamManager[Any]],
        instance: AsyncResponses,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncResponseStreamManagerWrapper[Any]:
        def invocation_factory() -> Any:
            params = extract_params(**kwargs)
            invocation = handler.inference(
                **get_inference_creation_kwargs(params, instance)
            )
            apply_request_attributes(invocation, params, capture_content)
            return invocation

        return AsyncResponseStreamManagerWrapper(
            wrapped(*args, **kwargs),
            invocation_factory,
            capture_content,
        )

    return cast(
        "Callable[..., AsyncResponseStreamManagerWrapper[Any]]", traced_method
    )
