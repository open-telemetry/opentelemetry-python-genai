# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a `tox -l` listing on stdin into GitHub Actions matrix JSON.

Emits two lines suitable for $GITHUB_OUTPUT:

  test_matrix={"include": [...]}
  lint_matrix={"include": [...]}

Each test entry has `name`, `tox_env`, `python_version`, and `needs_weaver`
when the tox env ends in `-conformance`. Each lint entry has `name` and
`tox_env`. Misc tox envs (everything else) are emitted by hand in misc.yml
because each has bespoke per-job logic.

Usage in CI:

  uvx --with tox-uv tox -l | python scripts/tox_matrix.py >> "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import json
import re
import sys

_PY_FACTOR_VERSION: dict[str, str] = {"pypy3": "pypy-3.10"}
_PY_RE = re.compile(r"^py(\d)(\d+)$")


def py_factor_to_version(factor: str) -> str:
    if factor in _PY_FACTOR_VERSION:
        return _PY_FACTOR_VERSION[factor]
    m = _PY_RE.fullmatch(factor)
    if not m:
        raise ValueError(f"Unrecognized python factor: {factor!r}")
    return f"{m.group(1)}.{m.group(2)}"


def main() -> None:
    envs = [line.strip() for line in sys.stdin if line.strip()]
    test_include: list[dict[str, object]] = []
    lint_include: list[dict[str, object]] = []
    for env in envs:
        if "-test-" in env:
            py_factor, rest = env.split("-test-", 1)
            py_version = py_factor_to_version(py_factor)
            entry: dict[str, object] = {
                "name": f"{rest} {py_version} Ubuntu",
                "tox_env": env,
                "python_version": py_version,
            }
            if rest.endswith("-conformance"):
                entry["needs_weaver"] = True
            test_include.append(entry)
        elif env.startswith("lint-"):
            lint_include.append(
                {
                    "name": env[len("lint-") :],
                    "tox_env": env,
                }
            )
    test_include.sort(key=lambda e: e["tox_env"])
    lint_include.sort(key=lambda e: e["tox_env"])
    print(
        f"test_matrix={json.dumps({'include': test_include}, sort_keys=True)}"
    )
    print(
        f"lint_matrix={json.dumps({'include': lint_include}, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
