# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from enum import Enum
from typing import Final

OPENAI_API_TYPE: Final[str] = "openai.api.type"
"""The type of OpenAI API being used."""

OPENAI_REQUEST_SERVICE_TIER: Final[str] = "openai.request.service_tier"
"""The service tier requested. May be a specific tier, default, or auto."""

OPENAI_RESPONSE_SERVICE_TIER: Final[str] = "openai.response.service_tier"
"""The service tier used for the response."""

OPENAI_RESPONSE_SYSTEM_FINGERPRINT: Final[str] = (
    "openai.response.system_fingerprint"
)
"""A fingerprint to track any eventual change in the Generative AI environment."""


class OpenAIApiType(str, Enum):
    """Known values for `openai.api.type`."""

    CHAT_COMPLETIONS = "chat_completions"
    """The OpenAI [Chat Completions API](https://developers.openai.com/api/reference/chat-completions/overview)."""
    RESPONSES = "responses"
    """The OpenAI [Responses API](https://developers.openai.com/api/reference/responses/overview)."""


class OpenAIRequestServiceTier(str, Enum):
    """Known values for `openai.request.service_tier`."""

    AUTO = "auto"
    """The system will utilize scale tier credits until they are exhausted."""
    DEFAULT = "default"
    """The system will utilize the default scale tier."""


OpenaiApiType = OpenAIApiType
OpenaiRequestServiceTier = OpenAIRequestServiceTier

__all__ = [
    "OPENAI_API_TYPE",
    "OPENAI_REQUEST_SERVICE_TIER",
    "OPENAI_RESPONSE_SERVICE_TIER",
    "OPENAI_RESPONSE_SYSTEM_FINGERPRINT",
    "OpenAIApiType",
    "OpenAIRequestServiceTier",
    "OpenaiApiType",
    "OpenaiRequestServiceTier",
]
