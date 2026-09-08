# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from _helpers import MESSAGES, transformers_model

model = transformers_model(stream_chunks=["In ", "Paris"])
list(model.generate_stream(messages=MESSAGES))
