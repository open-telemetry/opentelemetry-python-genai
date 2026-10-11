# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

_REAL_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAARklEQVR42u3XQQ0AIAwAsSnZG4lInJxJwMRICGlyAvq9yF1PFUBAQEBAQBdAXWskICAgICAgICAgICAgIOcKBAQEBPQd6ACUHHNEU5qggAAAAABJRU5ErkJggg=="

OpenAI().responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
                },
            ],
        }
    ],
)
