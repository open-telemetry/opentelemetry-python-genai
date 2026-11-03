# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for exporting tox dependency environments without a package index."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "generate_requirements.py"
SPEC = importlib.util.spec_from_file_location("generate_requirements", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)

OLDEST = "py311-test-instrumentation-genai-anthropic-oldest"
LATEST = "py314-test-instrumentation-genai-qwenpaw-latest"


def write_pilot_inputs(root: Path) -> None:
    for path in (
        generator._ANTHROPIC / "pyproject.toml",
        generator._ANTHROPIC
        / "src/opentelemetry/instrumentation/genai/anthropic/version.py",
        Path("util/opentelemetry-test-util-genai/pyproject.toml"),
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("test input\n", encoding="utf-8")


class GenerateRequirementsTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("tox"), "tox is not installed"
    )
    def test_anthropic_tox_environments_use_available_inputs(self) -> None:
        root = SCRIPT.parents[1]
        locked = (
            "py311-test-instrumentation-genai-anthropic-oldest",
            "py311-test-instrumentation-genai-anthropic-latest",
            "py315-test-instrumentation-genai-anthropic-latest",
        )
        other = (
            "py312-test-instrumentation-genai-anthropic-oldest",
            "py314-test-instrumentation-genai-anthropic-latest",
        )
        coverage = "py311-test-instrumentation-genai-anthropic-latest-coverage"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tox",
                "config",
                "-e",
                ",".join((*locked, *other, coverage)),
                "-k",
                "deps",
                "--format",
                "json",
                "--colored",
                "no",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        configurations = json.loads(result.stdout)["env"]
        lock_dir = "./instrumentation/opentelemetry-instrumentation-genai-anthropic/tests/locks"
        for environment in locked:
            dependencies = [
                generator.portable(dependency, root)
                for dependency in configurations[environment]["deps"]
            ]
            self.assertEqual(
                dependencies, [f"-r {lock_dir}/{environment}.txt"]
            )
        for environment in other:
            dependencies = [
                generator.portable(dependency, root)
                for dependency in configurations[environment]["deps"]
            ]
            self.assertTrue(dependencies)
            self.assertFalse(
                any("/tests/locks/" in dep for dep in dependencies)
            )
        self.assertTrue(
            any(
                dependency.endswith("anthropic[instruments]")
                for dependency in configurations[other[0]]["deps"]
            )
        )
        self.assertTrue(
            any(
                dependency.endswith("/tests/requirements.latest.txt")
                for dependency in configurations[other[1]]["deps"]
            )
        )
        coverage_dependencies = [
            generator.portable(dependency, root)
            for dependency in configurations[coverage]["deps"]
        ]
        self.assertIn(f"-r {lock_dir}/{locked[1]}.txt", coverage_dependencies)

    def test_environment_resolution_and_python_version(self) -> None:
        self.assertEqual(
            generator.environment_options(OLDEST), ("3.11", "lowest-direct")
        )
        self.assertEqual(
            generator.environment_options(LATEST), ("3.14", "highest")
        )
        self.assertEqual(
            generator.environment_options(
                "py313-test-instrumentation-genai-qwenpaw-latest"
            ),
            ("3.13", "highest"),
        )

    def test_invalid_environment_does_not_run_tools(self) -> None:
        with patch.object(generator.subprocess, "run") as run:
            for environment in (
                "typecheck",
                "py314-test-instrumentation-genai-anthropic-conformance",
                "py311-test-instrumentation-genai-anthropic-oldest,precommit",
            ):
                with self.subTest(environment=environment):
                    with self.assertRaises(ValueError):
                        generator.generate_requirements(
                            Path.cwd(), environment, Path("unused.txt")
                        )
            run.assert_not_called()

    def test_export_preserves_dependencies_and_portable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            write_pilot_inputs(root)
            output = root / "locked.txt"
            dependencies = [
                "pytest>=8",
                "tomli; python_version < '3.11'",
                f"-e {root.as_posix()}/instrumentation/example[instruments]",
                f"-r {root.as_posix()}/tests/requirements.oldest.txt",
            ]

            def run(command: list[str], **kwargs: object) -> object:
                if command[0] != "uv":
                    self.assertIn(
                        f"testenv:{OLDEST}.deps={{[testenv]anthropic_oldest_lock_inputs}}",
                        command,
                    )
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        json.dumps(
                            {
                                "env": {
                                    OLDEST: {
                                        "deps": dependencies,
                                        "set_env": {
                                            "UV_RESOLUTION": "lowest-direct",
                                            "UV_CONFIG_FILE": "tox-uv.toml",
                                        },
                                    }
                                }
                            }
                        ),
                    )
                self.assertEqual(kwargs["cwd"], root)
                self.assertEqual(
                    kwargs["env"]["UV_RESOLUTION"], "lowest-direct"
                )
                self.assertNotIn("UV_CONFIG_FILE", kwargs["env"])
                self.assertEqual(
                    kwargs["input"],
                    "pytest>=8\n"
                    "tomli; python_version < '3.11'\n"
                    "-e ./instrumentation/example[instruments]\n"
                    "-r ./tests/requirements.oldest.txt\n",
                )
                self.assertEqual(
                    command[command.index("--python-version") + 1], "3.11"
                )
                self.assertEqual(
                    command[command.index("--resolution") + 1], "lowest-direct"
                )
                self.assertIn("--universal", command)
                Path(command[command.index("--output-file") + 1]).write_text(
                    f"-e {root.as_posix()}/instrumentation/example\n"
                    "pytest==8.0.0\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with patch.object(generator.subprocess, "run", side_effect=run):
                generator.generate_requirements(root, OLDEST, output)
            content = output.read_text(encoding="utf-8")
            self.assertNotIn(str(root), content)
            self.assertIn("-e ./instrumentation/example\n", content)
            self.assertIn("pytest==8.0.0\n", content)
            self.assertIn(OLDEST, content)

    def test_check_rejects_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            write_pilot_inputs(root)
            output = root / "locked.txt"

            def config(command: list[str], **kwargs: object) -> object:
                dependencies = (
                    ["pytest>=8"] if "-x" in command else [f"-r {output}"]
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps({"env": {OLDEST: {"deps": dependencies}}}),
                )

            fingerprint = generator.input_fingerprint(
                root, OLDEST, ["pytest>=8"]
            )
            output.write_text(
                f"# Inputs SHA256: {fingerprint}\n", encoding="utf-8"
            )
            with patch.object(generator.subprocess, "run", side_effect=config):
                generator.generate_requirements(
                    root, OLDEST, output, check=True
                )
                (root / generator._ANTHROPIC / "pyproject.toml").write_text(
                    "changed\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "out of date"):
                    generator.generate_requirements(
                        root, OLDEST, output, check=True
                    )

    def test_check_rejects_tox_bypassing_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            write_pilot_inputs(root)
            output = root / "locked.txt"
            fingerprint = generator.input_fingerprint(
                root, OLDEST, ["pytest>=8"]
            )
            output.write_text(
                f"# Inputs SHA256: {fingerprint}\n", encoding="utf-8"
            )
            config = subprocess.CompletedProcess(
                [], 0, json.dumps({"env": {OLDEST: {"deps": ["pytest>=8"]}}})
            )
            with patch.object(
                generator.subprocess, "run", return_value=config
            ):
                with self.assertRaisesRegex(ValueError, "must install only"):
                    generator.generate_requirements(
                        root, OLDEST, output, check=True
                    )

    def test_nonpilot_preserves_tox_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "locked.txt"

            def run(command: list[str], **kwargs: object) -> object:
                if command[0] != "uv":
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        json.dumps(
                            {
                                "env": {
                                    LATEST: {
                                        "deps": ["pytest>=8"],
                                        "set_env": {
                                            "UV_OVERRIDE": "overrides.txt"
                                        },
                                    }
                                }
                            }
                        ),
                    )
                self.assertEqual(kwargs["env"]["UV_OVERRIDE"], "overrides.txt")
                self.assertNotIn("--universal", command)
                Path(command[command.index("--output-file") + 1]).write_text(
                    "pytest==8.0.0\n", encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0)

            with patch.object(generator.subprocess, "run", side_effect=run):
                generator.generate_requirements(root, LATEST, output)

    def test_failed_resolution_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "locked.txt"
            output.write_text("previous lock\n", encoding="utf-8")
            config = subprocess.CompletedProcess(
                [],
                0,
                json.dumps({"env": {LATEST: {"deps": ["pytest>=8"]}}}),
            )
            with patch.object(
                generator.subprocess,
                "run",
                side_effect=[config, subprocess.CalledProcessError(1, "uv")],
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    generator.generate_requirements(root, LATEST, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"), "previous lock\n"
            )

    def test_empty_dependencies_preserve_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "locked.txt"
            output.write_text("previous lock\n", encoding="utf-8")
            config = subprocess.CompletedProcess(
                [], 0, json.dumps({"env": {LATEST: {"deps": []}}})
            )
            with patch.object(
                generator.subprocess, "run", return_value=config
            ) as run:
                with self.assertRaises(ValueError):
                    generator.generate_requirements(root, LATEST, output)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(
                output.read_text(encoding="utf-8"), "previous lock\n"
            )


if __name__ == "__main__":
    unittest.main()
