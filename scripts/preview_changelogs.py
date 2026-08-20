# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Script to preview changelogs with towncrier and ensure CHANGELOG.md exists."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    packages = sorted(
        p for p in (ROOT / "instrumentation").iterdir() if p.is_dir()
    )
    util_pkg = ROOT / "util" / "opentelemetry-util-genai"
    if util_pkg.is_dir():
        packages.append(util_pkg)

    error = False
    for pkg in packages:
        pyproject = pkg / "pyproject.toml"
        if not pyproject.is_file():
            continue

        pyproject_content = pyproject.read_text(encoding="utf-8")
        if "tool.towncrier" not in pyproject_content:
            continue

        changelog = pkg / "CHANGELOG.md"
        if not changelog.is_file():
            print(
                f"FAILED: {pkg.name} configures towncrier but is missing CHANGELOG.md (copy from an existing package)",
                file=sys.stderr,
            )
            error = True
            continue

        print(f"=== {pkg.name} ===")
        res = subprocess.run(
            ["towncrier", "build", "--draft", "--version", "Unreleased"],
            cwd=str(pkg),
            check=False,
        )
        if res.returncode != 0:
            error = True

    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()
