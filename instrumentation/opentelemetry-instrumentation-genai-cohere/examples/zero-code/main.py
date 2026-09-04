# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
"""Placeholder example for Cohere instrumentation via opentelemetry-instrument.

The CohereInstrumentor shipped in this release is scaffold-only; it does
not yet wrap any client methods. Chat completions wrapping lands in a
follow-up PR. This file shows the planned client surface so the example
directory mirrors other instrumentation packages.
"""

import cohere


def main() -> None:
    co = cohere.ClientV2()
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "hello world!"}],
    )
    print(response)


if __name__ == "__main__":
    main()
