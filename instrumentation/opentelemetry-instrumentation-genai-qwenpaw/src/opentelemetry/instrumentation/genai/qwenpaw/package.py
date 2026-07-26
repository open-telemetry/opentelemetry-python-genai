# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Instrumented-distribution metadata for the QwenPaw instrumentation.

QwenPaw was originally published as ``copaw``; the last release under that
name is ``copaw 1.0.2``. Both distributions expose the same
``<package>.app.runner.runner.AgentRunner`` surface, so this package ships
one instrumentor plugin per distribution (``QwenPawInstrumentor`` and
``CoPawInstrumentor``).
"""

from __future__ import annotations

_instruments_qwenpaw = "qwenpaw >= 1.1.0, < 2.0.0"
_instruments_copaw = "copaw >= 0.1.0, <= 1.0.2"

# Read by scripts/generate_instrumentation_readme.py: any (not all) of these
# distributions is expected to be installed.
_instruments: tuple[str, ...] = ()
_instruments_any: tuple[str, ...] = (_instruments_qwenpaw, _instruments_copaw)

_supports_metrics = True
