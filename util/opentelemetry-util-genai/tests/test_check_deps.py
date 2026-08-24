# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to sys.path to import check_deps
repo_root = Path(__file__).resolve().parent.parent.parent.parent
scripts_dir = str(repo_root / "scripts")
sys.path.insert(0, scripts_dir)
try:
    from check_deps import (
        check_instruments_match,
        check_latest_requirements,
        check_workspace_dependencies,
    )
finally:
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)


class TestCheckWorkspaceDependencies(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)

        # Setup workspace packages
        self.util_dir = self.repo_root / "util" / "opentelemetry-util-genai"
        self.util_dir.mkdir(parents=True, exist_ok=True)
        (self.util_dir / "pyproject.toml").write_text(
            '[project]\nname = "opentelemetry-util-genai"\n',
            encoding="utf-8",
        )

        self.workspace_packages = {
            "opentelemetry-util-genai": (self.util_dir, "1.2b0.dev"),
        }

        self.pkg_dir = (
            self.repo_root
            / "instrumentation"
            / "opentelemetry-instrumentation-test"
        )
        self.pkg_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir = self.pkg_dir / "tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.oldest_path = self.tests_dir / "requirements.oldest.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_released_floor_without_editable_passes(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0, < 2"],
            }
        }
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            None,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(errors, [])

    def test_released_floor_with_editable_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0, < 2"],
            }
        }
        self.oldest_path.write_text(
            "-e ../../util/opentelemetry-util-genai\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("released version on PyPI", errors[0])
        self.assertIn("Remove the local/editable install", errors[0])

    def test_released_floor_with_non_editable_local_path_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0, < 2"],
            }
        }
        self.oldest_path.write_text(
            "../../util/opentelemetry-util-genai\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("released version on PyPI", errors[0])
        self.assertIn("Remove the local/editable install", errors[0])

    def test_matching_dev_floor_with_editable_passes(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.2b0.dev, < 2"],
            }
        }
        self.oldest_path.write_text(
            "-e ../../util/opentelemetry-util-genai\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(errors, [])

    def test_matching_dev_floor_with_non_editable_local_path_passes(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.2b0.dev, < 2"],
            }
        }
        self.oldest_path.write_text(
            "../../util/opentelemetry-util-genai\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(errors, [])

    def test_matching_dev_floor_missing_editable_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.2b0.dev, < 2"],
            }
        }
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            None,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("unreleased '1.2b0.dev'", errors[0])
        self.assertIn("missing a local/editable install", errors[0])

    def test_previous_dev_floor_without_editable_passes(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0.dev, < 2"],
            }
        }
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            None,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(errors, [])

    def test_exceeding_dev_floor_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.3b0.dev, < 2"],
            }
        }
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            None,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds current workspace version", errors[0])

    def test_extraneous_editable_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0, < 2"],
            }
        }
        # Another workspace package not in declared deps
        other_dir = self.repo_root / "util" / "other-pkg"
        other_dir.mkdir(parents=True, exist_ok=True)
        (other_dir / "pyproject.toml").write_text(
            '[project]\nname = "other-pkg"\n',
            encoding="utf-8",
        )
        self.oldest_path.write_text(
            "-e ../../util/other-pkg\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("is not permitted", errors[0])

    def test_extraneous_non_editable_local_path_fails(self):
        pyproject = {
            "project": {
                "name": "opentelemetry-instrumentation-test",
                "dependencies": ["opentelemetry-util-genai >= 1.1b0, < 2"],
            }
        }
        other_dir = self.repo_root / "util" / "other-pkg"
        other_dir.mkdir(parents=True, exist_ok=True)
        (other_dir / "pyproject.toml").write_text(
            '[project]\nname = "other-pkg"\n',
            encoding="utf-8",
        )
        self.oldest_path.write_text(
            "../../util/other-pkg\n",
            encoding="utf-8",
        )
        errors = check_workspace_dependencies(
            self.pkg_dir,
            pyproject,
            self.oldest_path,
            self.repo_root,
            self.workspace_packages,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("is not permitted", errors[0])


class TestCheckInstrumentsMatch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pkg_dir = Path(self.temp_dir.name)
        self.src_dir = (
            self.pkg_dir
            / "src"
            / "opentelemetry"
            / "instrumentation"
            / "genai"
            / "test_pkg"
        )
        self.src_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matching_with_bounds_passes(self):
        (self.src_dir / "package.py").write_text(
            '_instruments = ("some-pkg >= 1.0.0, < 2",)\n',
            encoding="utf-8",
        )
        pyproject = {
            "project": {
                "optional-dependencies": {
                    "instruments": ["some-pkg >= 1.0.0, < 2"]
                }
            }
        }
        errors = check_instruments_match(self.pkg_dir, pyproject)
        self.assertEqual(errors, [])

    def test_missing_upper_bound_fails(self):
        (self.src_dir / "package.py").write_text(
            '_instruments = ("some-pkg >= 1.0.0",)\n',
            encoding="utf-8",
        )
        pyproject = {
            "project": {
                "optional-dependencies": {"instruments": ["some-pkg >= 1.0.0"]}
            }
        }
        errors = check_instruments_match(self.pkg_dir, pyproject)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing an upper bound", errors[0])

    def test_missing_lower_bound_fails(self):
        (self.src_dir / "package.py").write_text(
            '_instruments = ("some-pkg < 2",)\n',
            encoding="utf-8",
        )
        pyproject = {
            "project": {
                "optional-dependencies": {"instruments": ["some-pkg < 2"]}
            }
        }
        errors = check_instruments_match(self.pkg_dir, pyproject)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing a lower bound", errors[0])

    def test_mismatch_fails(self):
        (self.src_dir / "package.py").write_text(
            '_instruments = ("some-pkg >= 1.0.0, < 2",)\n',
            encoding="utf-8",
        )
        pyproject = {
            "project": {
                "optional-dependencies": {
                    "instruments": ["other-pkg >= 1.0.0, < 2"]
                }
            }
        }
        errors = check_instruments_match(self.pkg_dir, pyproject)
        self.assertEqual(len(errors), 1)
        self.assertIn("does not match", errors[0])


class TestCheckLatestRequirements(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pkg_dir = Path(self.temp_dir.name)
        self.tests_dir = self.pkg_dir / "tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.tests_dir / "requirements.latest.txt"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_latest_pinned_passes(self):
        self.latest_path.write_text("some-pkg ~= 1.0\n", encoding="utf-8")
        pyproject = {
            "project": {
                "optional-dependencies": {
                    "instruments": ["some-pkg >= 1.0.0, < 2"]
                }
            }
        }
        errors = check_latest_requirements(self.pkg_dir, pyproject)
        self.assertEqual(errors, [])

    def test_latest_unbounded_fails(self):
        self.latest_path.write_text("some-pkg >= 1.0\n", encoding="utf-8")
        pyproject = {
            "project": {
                "optional-dependencies": {
                    "instruments": ["some-pkg >= 1.0.0, < 2"]
                }
            }
        }
        errors = check_latest_requirements(self.pkg_dir, pyproject)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing an upper bound or pin", errors[0])
