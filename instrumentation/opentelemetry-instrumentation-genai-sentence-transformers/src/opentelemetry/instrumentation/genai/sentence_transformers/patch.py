# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers that turn ``SentenceTransformer.encode`` calls into embedding spans."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from opentelemetry.util.genai.handler import TelemetryHandler

# The GenAI registry has no value for in-process Hugging Face runtimes, so the
# product name is used, matching the smolagents instrumentation's value for
# ``TransformersModel``. ``gen_ai.provider.name`` is also a metric attribute,
# so it stays a constant.
_PROVIDER = "huggingface"

# On 6.x, ``save`` builds the model card's usage snippet by calling ``encode``,
# which is not an embedding request made by the caller.
_in_save: ContextVar[bool] = ContextVar("_in_save", default=False)


def _model_name(instance: Any) -> str | None:
    """Best-effort model identifier for a ``SentenceTransformer`` instance.

    ``base_model`` holds the Hub model a model was loaded from or fine-tuned
    on, and ``model_id`` the Hub id it was saved or pushed under. The card's
    ``model_name`` is not used: it is a display name that ``save`` sets to
    ``SentenceTransformer`` when there is no base model.
    """
    card = getattr(instance, "model_card_data", None)
    for attr in ("base_model", "model_id"):
        name = getattr(card, attr, None)
        if isinstance(name, str) and name:
            return name
    return None


def _dimension_count(instance: Any) -> int | None:
    # ``get_sentence_embedding_dimension`` was renamed to
    # ``get_embedding_dimension`` in recent releases; prefer the new name and
    # fall back to the old one so this works (and stays quiet) on both.
    for name in (
        "get_embedding_dimension",
        "get_sentence_embedding_dimension",
    ):
        getter = getattr(instance, name, None)
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:  # pylint: disable=broad-except
            return None
        return value if isinstance(value, int) else None
    return None


def wrap_encode(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrap ``SentenceTransformer.encode`` in an ``embeddings`` invocation."""
    if _in_save.get():
        return wrapped(*args, **kwargs)

    invocation = handler.embedding(
        _PROVIDER,
        request_model=_model_name(instance),
    )
    try:
        result = wrapped(*args, **kwargs)
    except BaseException as exc:
        invocation.fail(exc)
        raise

    invocation.dimension_count = _dimension_count(instance)
    invocation.stop()
    return result


def wrap_save(
    wrapped: Any,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Wrap ``SentenceTransformer.save`` so its internal ``encode`` calls are not recorded.

    ``save_pretrained`` and ``push_to_hub`` delegate to ``save``.
    """
    token = _in_save.set(True)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _in_save.reset(token)
