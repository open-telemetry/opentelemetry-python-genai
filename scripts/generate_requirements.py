# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a tox oldest/latest environment into a pinned requirements file."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ENV_PATTERN = re.compile(r"^py(\d)(\d+)-test-.+-(oldest|latest)$")
_ANTHROPIC_ENV = re.compile(
    r"^py\d+-test-instrumentation-genai-anthropic-(oldest|latest)$"
)


def environment_options(environment: str) -> tuple[str, str]:
    """Return the Python version and oldest/latest resolution."""
    match = _ENV_PATTERN.fullmatch(environment)
    if match is None:
        raise ValueError(
            "Expected a py311-test-...-oldest or py315-test-...-latest "
            "environment."
        )
    python_version = f"{match[1]}.{match[2]}"
    resolution = "lowest-direct" if match[3] == "oldest" else "highest"
    return python_version, resolution


def portable(value: str, repo_root: Path) -> str:
    """Make tox-expanded paths relative to the repository root."""
    if os.name == "nt":
        value = value.replace("\\", "/")
    root = repo_root.as_posix()
    for prefix in (root.replace(" ", "\\ "), root):
        value = value.replace(prefix + "/", "./")
    return value


def generate_requirements(
    repo_root: Path, environment: str, output: Path
) -> None:
    """Resolve the environment's complete tox dependency list with uv."""
    python_version, resolution = environment_options(environment)
    repo_root = repo_root.resolve()
    output = output.resolve()
    command = [
        sys.executable,
        "-m",
        "tox",
        "config",
        "-e",
        environment,
        "-k",
        "deps",
        "set_env",
        "--format",
        "json",
        "--colored",
        "no",
    ]
    pilot = _ANTHROPIC_ENV.fullmatch(environment)
    if pilot:
        # Read source dependencies so regeneration never resolves from a lock.
        command.extend(
            [
                "-x",
                f"testenv:{environment}.deps={{[testenv]anthropic_{pilot[1]}_lock_inputs}}",
            ]
        )
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    configuration = json.loads(result.stdout)["env"][environment]
    dependencies = configuration["deps"]
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        raise ValueError("tox did not return a dependency list.")
    if not dependencies:
        raise ValueError(f"No dependencies found for {environment}.")

    dependencies = [
        portable(dependency, repo_root) for dependency in dependencies
    ]
    requirements = "\n".join(dependencies) + "\n"
    process_env = os.environ.copy()
    process_env.update(configuration.get("set_env", {}))
    process_env.pop("UV_CONFIG_FILE", None)

    with tempfile.TemporaryDirectory() as temporary:
        compiled = Path(temporary) / "requirements.txt"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "-",
                "--no-config",
                "--no-sources",
                "--no-header",
                "--no-annotate",
                "--python-version",
                python_version,
                "--resolution",
                resolution,
                *(["--universal"] if pilot else []),
                "--output-file",
                str(compiled),
            ],
            cwd=repo_root,
            check=True,
            input=requirements,
            env=process_env,
            text=True,
            stdout=subprocess.DEVNULL,
        )
        content = compiled.read_text(encoding="utf-8")
        content = portable(content, repo_root)
        header = (
            "# Copyright The OpenTelemetry Authors\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            f"# Generated for {environment}; Python {python_version}, "
            f"resolution {resolution}.\n"
        )
        output.write_text(header + content, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", help="A tox oldest/latest test env")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination requirements file",
    )
    args = parser.parse_args()
    try:
        generate_requirements(
            Path(__file__).resolve().parent.parent,
            args.environment,
            args.output,
        )
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
