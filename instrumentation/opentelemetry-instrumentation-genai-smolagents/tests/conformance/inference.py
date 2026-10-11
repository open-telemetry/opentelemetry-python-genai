# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: smolagents chat via an in-process model stub."""

from _helpers import MESSAGES, transformers_model

transformers_model().generate(messages=MESSAGES)
