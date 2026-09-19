# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from typing import Final

AWS_BEDROCK_GUARDRAIL_ID: Final[str] = "aws.bedrock.guardrail.id"
"""The unique identifier of the AWS Bedrock Guardrail. A [guardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) helps safeguard and prevent unwanted behavior from model responses or user messages."""

AWS_BEDROCK_KNOWLEDGE_BASE_ID: Final[str] = "aws.bedrock.knowledge_base.id"
"""The unique identifier of the AWS Bedrock Knowledge base. A [knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) is a bank of information that can be queried by models to generate more relevant responses and augment prompts."""


__all__ = [
    "AWS_BEDROCK_GUARDRAIL_ID",
    "AWS_BEDROCK_KNOWLEDGE_BASE_ID",
]
