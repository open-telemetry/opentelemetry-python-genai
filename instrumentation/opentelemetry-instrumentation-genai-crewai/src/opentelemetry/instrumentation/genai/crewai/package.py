# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# 1.11.0 is the first release that copies contextvars into the thread pools
# used for timed agent execution and parallel native tool calls; on earlier
# releases those tool spans would not be parented to the agent span.
_instruments = ("crewai >= 1.11.0, < 2",)
