# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for exporting tox dependency environments without a package index."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "generate_requirements.py"
SPEC = importlib.util.spec_from_file_location("generate_requirements", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)

OLDEST = "py310-test-instrumentation-genai-anthropic-oldest"
LATEST = "py314-test-instrumentation-genai-anthropic-latest"


class GenerateRequirementsTest(unittest.TestCase):
    def test_environment_resolution_and_python_version(self) -> None:
        self.assertEqual(
            generator.environment_options(OLDEST), ("3.10", "lowest-direct")
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
                "py310-test-instrumentation-genai-anthropic-oldest,precommit",
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
            output = root / "locked.txt"
            dependencies = [
                "pytest>=8",
                "tomli; python_version < '3.11'",
                f"-e {root.as_posix()}/instrumentation/example[instruments]",
                f"-r {root.as_posix()}/tests/requirements.oldest.txt",
            ]

            def run(command: list[str], **kwargs: object) -> object:
                if command[0] != "uv":
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        json.dumps(
                            {
                                "env": {
                                    OLDEST: {
                                        "deps": dependencies,
                                        "set_env": {
                                            "UV_OVERRIDE": "overrides.txt",
                                            "UV_CONFIG_FILE": "tox-uv.toml",
                                        },
                                    }
                                }
                            }
                        ),
                    )
                self.assertEqual(kwargs["cwd"], root)
                self.assertEqual(kwargs["env"]["UV_OVERRIDE"], "overrides.txt")
                self.assertNotIn("UV_CONFIG_FILE", kwargs["env"])
                self.assertEqual(
                    kwargs["input"],
                    "pytest>=8\n"
                    "tomli; python_version < '3.11'\n"
                    "-e ./instrumentation/example[instruments]\n"
                    "-r ./tests/requirements.oldest.txt\n",
                )
                self.assertEqual(
                    command[command.index("--python-version") + 1], "3.10"
                )
                self.assertEqual(
                    command[command.index("--resolution") + 1], "lowest-direct"
                )
                Path(command[-1]).write_text(
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
