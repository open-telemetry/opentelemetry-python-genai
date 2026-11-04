# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a tox oldest/latest environment into a pinned requirements file."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement

_ENV_PATTERN = re.compile(r"^py(\d)(\d+)-test-.+-(oldest|latest)$")
_ANTHROPIC_ENV = re.compile(
    r"^py\d+-test-instrumentation-genai-anthropic-(oldest|latest)$"
)
_ANTHROPIC = Path(
    "instrumentation/opentelemetry-instrumentation-genai-anthropic"
)
_RESOLUTION_ENV = (
    "UV_OVERRIDE*",
    "UV_CONSTRAINT*",
    "UV_EXCLUDE*",
    "UV_PRERELEASE*",
    "UV_FORK_STRATEGY",
    "UV_RESOLUTION",
    "UV_PYTHON_PLATFORM",
)
_LOCK_PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s;*]+)(?P<marker>\s*;.*)?$"
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


def included_requirements(
    root: Path, dependencies: list[str]
) -> dict[Path, bool]:
    """Find local requirements and constraints included by the tox inputs."""
    found: dict[Path, bool] = {}
    pending = [(root, line, True) for line in dependencies]
    while pending:
        base, line, required = pending.pop()
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        option = tokens[0]
        if option in ("-r", "--requirement", "-c", "--constraint"):
            if len(tokens) != 2:
                raise ValueError(f"Invalid requirements include: {line}")
            target = tokens[1]
        elif option.startswith(("--requirement=", "--constraint=")):
            target = option.split("=", 1)[1]
        elif option.startswith(("-r", "-c")) and len(option) > 2:
            target = option[2:]
        else:
            continue
        required = required and not option.startswith(("-c", "--constraint"))
        if "$" in target or "://" in target:
            raise ValueError(f"Unsupported requirements include: {target}")
        path = (base / target).resolve()
        try:
            relative = path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"Requirements include is outside the repo: {target}"
            ) from error
        if relative not in found or (required and not found[relative]):
            found[relative] = required
            pending.extend(
                (path.parent, item, required)
                for item in path.read_text(encoding="utf-8").splitlines()
            )
    return found


def lock_shape(content: str, *, pin_versions: bool = False) -> str:
    """Fingerprint packages and markers, optionally including their versions."""
    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            entries.append(line)
            continue
        match = _LOCK_PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"Lock contains an unpinned requirement: {line}")
        entries.append(
            line
            if pin_versions
            else match["name"].lower() + (match["marker"] or "")
        )
    return hashlib.sha256(
        "\n".join(sorted(entries)).encode("utf-8")
    ).hexdigest()


def validate_direct_requirements(
    root: Path, dependencies: list[str], content: str
) -> None:
    """Reject pinned versions incompatible with the declared inputs."""
    pins: dict[str, list[str]] = {}
    for line in content.splitlines():
        match = _LOCK_PIN.fullmatch(line.strip())
        if match:
            name = re.sub(r"[-_.]+", "-", match["name"]).lower()
            pins.setdefault(name, []).append(match["version"])
    inputs = [(line, True) for line in dependencies]
    for path, required in included_requirements(root, dependencies).items():
        inputs.extend(
            (line, required)
            for line in (root / path).read_text(encoding="utf-8").splitlines()
        )
    for line, required in inputs:
        line = line.partition(" #")[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        if requirement.marker:
            continue
        name = re.sub(r"[-_.]+", "-", requirement.name).lower()
        if required and name not in pins:
            raise ValueError(f"Lock is missing direct requirement {name}.")
        if name in pins and any(
            not requirement.specifier.contains(version, prereleases=True)
            for version in pins[name]
        ):
            raise ValueError(f"Locked {name} does not satisfy {requirement}.")


def input_fingerprint(
    repo_root: Path,
    environment: str,
    dependencies: list[str],
    settings: dict[str, str] | None = None,
) -> str:
    """Fingerprint the Anthropic lock inputs without resolving dependencies."""
    paths = [
        _ANTHROPIC / "pyproject.toml",
        _ANTHROPIC
        / "src/opentelemetry/instrumentation/genai/anthropic/version.py",
        Path("util/opentelemetry-test-util-genai/pyproject.toml"),
    ]
    if environment.endswith("-latest"):
        paths.extend(
            [
                _ANTHROPIC / "tests/requirements.latest.txt",
                Path("util/opentelemetry-util-genai/pyproject.toml"),
                Path(
                    "util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py"
                ),
            ]
        )
    for path in sorted(included_requirements(repo_root, dependencies)):
        if path not in paths:
            paths.append(path)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [environment, dependencies, settings or {}], sort_keys=True
        ).encode("utf-8")
    )
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update((repo_root / path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def tox_configuration(
    repo_root: Path, environment: str, *, source: bool
) -> dict:
    """Read the selected tox dependencies, optionally before locking."""
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
    if source and pilot:
        command.extend(
            [
                "-x",
                f"testenv:{environment}.deps={{[testenv]anthropic_{pilot[1]}_lock_inputs}}",
            ]
        )
    result = subprocess.run(
        command, cwd=repo_root, check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)["env"][environment]


def portable(value: str, repo_root: Path) -> str:
    """Make tox-expanded paths independent of the checkout location."""
    if os.name == "nt":
        value = value.replace("\\", "/")
    root = repo_root.as_posix()
    for prefix in (root.replace(" ", "\\ "), root):
        value = value.replace(prefix + "/", "./")
    return value


def generate_requirements(
    repo_root: Path, environment: str, output: Path, *, check: bool = False
) -> None:
    """Resolve the environment's complete tox dependency list with uv."""
    python_version, resolution = environment_options(environment)
    repo_root = repo_root.resolve()
    output = output.resolve()
    pilot = _ANTHROPIC_ENV.fullmatch(environment)
    configuration = tox_configuration(repo_root, environment, source=True)
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
    settings = {
        name: portable(value, repo_root)
        for name, value in configuration.get("set_env", {}).items()
        if name.startswith("UV_") and name != "UV_CONFIG_FILE"
    }
    if pilot and (unsupported := settings.keys() - {"UV_RESOLUTION"}):
        raise ValueError(
            f"Unsupported Anthropic lock settings: {', '.join(sorted(unsupported))}"
        )
    fingerprint = (
        input_fingerprint(repo_root, environment, dependencies, settings)
        if pilot
        else None
    )
    if check:
        if fingerprint is None:
            raise ValueError(
                "Input checking is only supported for Anthropic locks."
            )
        installed = tox_configuration(repo_root, environment, source=False)[
            "deps"
        ]
        if [portable(dependency, repo_root) for dependency in installed] != [
            f"-r {portable(str(output), repo_root)}"
        ]:
            raise ValueError(f"{environment} must install only {output}.")
        if not output.is_file():
            raise ValueError(
                f"{output} is missing or out of date; regenerate it."
            )
        content = output.read_text(encoding="utf-8")
        if (
            f"# Inputs SHA256: {fingerprint}\n" not in content
            or f"# Packages SHA256: {lock_shape(content, pin_versions=environment.endswith('-oldest'))}\n"
            not in content
        ):
            raise ValueError(f"{output} is out of date; regenerate it.")
        validate_direct_requirements(repo_root, dependencies, content)
        return
    process_env = os.environ.copy()
    if pilot:
        for name in list(process_env):
            if any(
                fnmatch.fnmatchcase(name.upper(), pattern)
                for pattern in _RESOLUTION_ENV
            ):
                process_env.pop(name)
    process_env.update(configuration.get("set_env", {}))
    process_env.pop("UV_CONFIG_FILE", None)

    with tempfile.TemporaryDirectory() as temporary:
        compiled = Path(temporary) / "requirements.txt"
        compile_command = [
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
            "--output-file",
            str(compiled),
        ]
        if pilot:
            compile_command.append("--universal")
        subprocess.run(
            compile_command,
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
        if fingerprint:
            header += f"# Inputs SHA256: {fingerprint}\n"
            header += f"# Packages SHA256: {lock_shape(content, pin_versions=environment.endswith('-oldest'))}\n"
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that lock inputs are unchanged",
    )
    args = parser.parse_args()
    try:
        generate_requirements(
            Path(__file__).resolve().parent.parent,
            args.environment,
            args.output,
            check=args.check,
        )
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
