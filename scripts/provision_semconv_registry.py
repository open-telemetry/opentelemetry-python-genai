# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the GenAI semconv registry pinned in ``versions.env`` at ``.semconv/``.

The conformance runner carries a registry pin of its own, which lags this
repo's ``SEMCONV_GENAI_REF`` — the ref util-genai's types are written against.
Each package's ``conformance.yaml`` overrides it under ``weaver.registry`` so
conformance checks the same conventions the code targets.

That override has to be a path, so the registry is staged at a fixed location
instead of one named after the ref: bumping the pin then touches only
``versions.env``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from opentelemetry.conformance import provision, require_pin

REPO = "open-telemetry/semantic-conventions-genai"
PIN = "SEMCONV_GENAI_REF"

ROOT = Path(__file__).resolve().parent.parent
# Referenced from every conformance.yaml as `../../../../.semconv/model`.
STAGED = ROOT / ".semconv"
STAMP = STAGED / ".ref"


def main() -> int:
    ref = require_pin(ROOT / "versions.env", PIN)
    if STAMP.is_file() and STAMP.read_text(encoding="utf-8").strip() == ref:
        print(f"{STAGED} already at {ref}")
        return 0

    registry = provision(REPO, ref, label=REPO.rpartition("/")[2]) / "model"
    if not registry.is_dir():
        raise SystemExit(f"{registry} is missing — {REPO}@{ref} has no model/")

    # Copied rather than linked: a symlink needs privileges on Windows, and
    # weaver is handed this path directly.
    if STAGED.exists():
        shutil.rmtree(STAGED)
    shutil.copytree(registry, STAGED / "model")
    STAMP.write_text(f"{ref}\n", encoding="utf-8")
    print(f"staged {REPO}@{ref} at {STAGED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
