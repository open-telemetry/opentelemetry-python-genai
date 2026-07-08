# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union, cast

from opentelemetry.util.genai.handler import TelemetryHandler

from .response_extractors import (
    apply_request_attributes,
    extract_params,
    get_inference_creation_kwargs,
    set_invocation_response_attributes,
)
from .response_wrappers import (
    AsyncResponseStreamManagerWrapper,
    AsyncResponseStreamWrapper,
    ResponseStreamManagerWrapper,
    ResponseStreamWrapper,
    responses_stream_context,
)

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

try:
    from openai import AsyncStream as _OpenAIAsyncStream
    from openai import Stream as _OpenAIStream
    from openai.lib.streaming.responses._responses import (  # pylint: disable=no-name-in-module
        AsyncResponseStream as _AsyncResponseStream,
    )
    from openai.lib.streaming.responses._responses import (  # pylint: disable=no-name-in-module
        ResponseStream as _ResponseStream,
    )
except ImportError:
    _AsyncResponseStream = None
    _OpenAIAsyncStream = None
    _OpenAIStream = None
    _ResponseStream = None

ResponseResult = Union["ParsedResponse[Any]", "Response"]
ResponseStreamResult = Union["OpenAIStream[Any]", "ResponseStream[Any]"]
AsyncResponseStreamResult = Union[
    "OpenAIAsyncStream[Any]", "AsyncResponseStream[Any]"
]


def responses_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Union[
        ResponseResult,
        ResponseStreamResult,
        ResponseStreamWrapper[Any],
    ],
]:
    """Wrap ``Responses.create`` to trace Responses API calls.

    Traces :meth:`openai.resources.responses.responses.Responses.create`.
    OpenAI SDK source:
    https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py#L914
    """

    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Union[ResponseResult, ResponseStreamResult]],
        instance: "Responses",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        ResponseResult,
        ResponseStreamResult,
        ResponseStreamWrapper[Any],
    ]:
        stream_context = responses_stream_context.get()
        if stream_context is not None:
            result = wrapped(*args, **kwargs)
            return _get_response_stream_result(result)

        params = extract_params(**kwargs)
        invocation = handler.inference(
            **get_inference_creation_kwargs(params, instance)
        )
        apply_request_attributes(invocation, params, capture_content)

        try:
            result = wrapped(*args, **kwargs)
            parsed_result = _get_response_stream_result(result)

            if (
                _ResponseStream is not None
                and isinstance(parsed_result, _ResponseStream)
            ) or (
                _OpenAIStream is not None
                and isinstance(parsed_result, _OpenAIStream)
            ):
                return ResponseStreamWrapper(
                    cast("ResponseStreamResult", parsed_result),
                    invocation,
                    capture_content,
                )

            set_invocation_response_attributes(
                invocation,
                cast("ResponseResult", parsed_result),
                capture_content,
            )
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., Union["ResponseResult", "ResponseStreamResult", ResponseStreamWrapper[Any]]]',
        traced_method,
    )


def async_responses_create(
    handler: TelemetryHandler,
) -> Callable[
    ...,
    Awaitable[
        Union[
            ResponseResult,
            AsyncResponseStreamResult,
            AsyncResponseStreamWrapper[Any],
        ]
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
            Awaitable[Union[ResponseResult, AsyncResponseStreamResult]],
        ],
        instance: "AsyncResponses",
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Union[
        ResponseResult,
        AsyncResponseStreamResult,
        AsyncResponseStreamWrapper[Any],
    ]:
        stream_context = responses_stream_context.get()
        if stream_context is not None:
            result = await wrapped(*args, **kwargs)
            return _get_response_stream_result(result)

        params = extract_params(**kwargs)
        invocation = handler.inference(
            **get_inference_creation_kwargs(params, instance)
        )
        apply_request_attributes(invocation, params, capture_content)

        try:
            result = await wrapped(*args, **kwargs)
            parsed_result = _get_response_stream_result(result)

            if (
                _AsyncResponseStream is not None
                and isinstance(parsed_result, _AsyncResponseStream)
            ) or (
                _OpenAIAsyncStream is not None
                and isinstance(parsed_result, _OpenAIAsyncStream)
            ):
                return AsyncResponseStreamWrapper(
                    cast("AsyncResponseStreamResult", parsed_result),
                    invocation,
                    capture_content,
                )

            set_invocation_response_attributes(
                invocation,
                cast("ResponseResult", parsed_result),
                capture_content,
            )
            invocation.stop()
            return result
        except Exception as error:
            invocation.fail(error)
            raise

    return cast(
        'Callable[..., Awaitable[Union["ResponseResult", "AsyncResponseStreamResult", AsyncResponseStreamWrapper[Any]]]]',
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
        wrapped: Callable[..., "ResponseStreamManager[Any]"],
        instance: "Responses",
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
        wrapped: Callable[..., "AsyncResponseStreamManager[Any]"],
        instance: "AsyncResponses",
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


def _get_response_stream_result(result):
    if hasattr(result, "parse"):
        return result.parse()
    return result
