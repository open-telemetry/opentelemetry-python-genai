# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from _helpers import vllm_model

_IMAGE_URL = (
    "https://fastly.picsum.photos/id/237/200/300.jpg"
    "?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
)

model = vllm_model()
model._is_vlm = True
model.flatten_messages_as_text = False
model.generate(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What breed is this dog?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _IMAGE_URL},
                },
            ],
        }
    ]
)
