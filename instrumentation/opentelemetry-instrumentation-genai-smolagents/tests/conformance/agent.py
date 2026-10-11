# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pathlib

from PIL import Image
from smolagents import OpenAIModel, ToolCallingAgent

image_path = pathlib.Path(__file__).parent.parent / "fixtures" / "img.png"
model = OpenAIModel(
    model_id="gpt-4o",
    api_key="test_openai_api_key",
    api_base=os.environ["OPENAI_BASE_URL"],
)
agent = ToolCallingAgent(tools=[], model=model, max_steps=3)
agent.run(
    "Describe what you see in this image briefly.",
    images=[Image.open(image_path)],
)
