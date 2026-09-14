#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["ruff==0.16.1"]
# ///

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "util/opentelemetry-test-util-genai/src"))
from opentelemetry.test_util_genai._setup_weaver import ensure_weaver

TEMPLATES = Path(__file__).with_name("templates")
OUTPUT = (
    ROOT / "util/opentelemetry-util-genai/src/opentelemetry/util/genai/semconv"
)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in (
        (ROOT / "versions.env").read_text(encoding="utf-8").splitlines()
    ):
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            versions[name] = value
    return versions


def main() -> None:
    versions = _versions()
    weaver = ensure_weaver(versions["WEAVER_VERSION"])
    registry = (
        "https://github.com/open-telemetry/semantic-conventions-genai.git@"
        f"{versions['SEMCONV_GENAI_REF']}[model]"
    )
    subprocess.run(
        [
            str(weaver),
            "registry",
            "generate",
            "--registry",
            registry,
            "--templates",
            str(TEMPLATES),
            "--v2",
            "python-genai",
            str(OUTPUT),
        ],
        cwd=ROOT,
        check=True,
    )
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RuntimeError("ruff is required; run this script with `uv run`")
    generated_files = [
        path
        for path in OUTPUT.glob("**/*.py")
        if "Code generated" in path.read_text(encoding="utf-8")[:256]
    ]
    subprocess.run(
        [ruff, "check", "--fix", *map(str, generated_files)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [ruff, "format", *map(str, generated_files)], cwd=ROOT, check=True
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
