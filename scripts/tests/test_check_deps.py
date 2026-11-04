# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace dependency validation for explicit oldest inputs."""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_deps.py"
SPEC = importlib.util.spec_from_file_location("check_deps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_deps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_deps)


class WorkspaceInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "instrumentation/example"
        (self.package / "tests").mkdir(parents=True)
        self.inputs = self.package / "tests/requirements.oldest.in"
        self.workspace = {
            "example": (self.package, "1.0b0.dev"),
            "opentelemetry-util-genai": (
                self.root / "util/opentelemetry-util-genai",
                "1.4b0.dev",
            ),
        }

    def configure(self, floor, local):
        (self.package / "pyproject.toml").write_text(
            '[project]\nname = "example"\n'
            f'dependencies = ["opentelemetry-util-genai >= {floor}, <2"]\n',
            encoding="utf-8",
        )
        self.inputs.write_text(
            "-e util/opentelemetry-util-genai\n"
            if local
            else "# No runtime overrides.\n",
            encoding="utf-8",
        )

    def run_checks(self):
        errors = io.StringIO()
        with (
            patch.object(
                check_deps,
                "__file__",
                str(self.root / "scripts/check_deps.py"),
            ),
            patch.object(
                check_deps,
                "get_workspace_packages",
                return_value=self.workspace,
            ),
            patch.object(
                check_deps, "check_instruments_match", return_value=[]
            ),
            redirect_stderr(errors),
            redirect_stdout(io.StringIO()),
        ):
            return check_deps.main(), errors.getvalue()

    def test_unreleased_floor_accepts_editable_in_new_inputs(self):
        self.configure("1.4b0.dev", local=True)
        self.assertEqual(self.run_checks(), (0, ""))

    def test_unreleased_floor_rejects_missing_editable(self):
        self.configure("1.4b0.dev", local=False)
        result, errors = self.run_checks()
        self.assertEqual(result, 1)
        self.assertIn(
            "tests/requirements.oldest.in is missing a local/editable install",
            errors,
        )

    def test_released_floor_rejects_editable(self):
        self.configure("1.3b0", local=True)
        result, errors = self.run_checks()
        self.assertEqual(result, 1)
        self.assertIn(
            "tests/requirements.oldest.in installs it locally", errors
        )

    def test_released_floor_accepts_pypi_inputs(self):
        self.configure("1.3b0", local=False)
        self.assertEqual(self.run_checks(), (0, ""))

    def test_legacy_oldest_file_still_works(self):
        self.configure("1.4b0.dev", local=True)
        legacy = self.inputs.with_suffix(".txt")
        self.inputs.rename(legacy)
        self.assertEqual(self.run_checks(), (0, ""))

    def test_release_cleanup_updates_new_inputs(self):
        self.configure("1.3b0", local=True)
        with patch.object(
            check_deps, "get_workspace_packages", return_value=self.workspace
        ):
            modified = check_deps.strip_released_local_installs(self.root)
        self.assertEqual(modified, [self.inputs])
        self.assertNotIn("-e", self.inputs.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
