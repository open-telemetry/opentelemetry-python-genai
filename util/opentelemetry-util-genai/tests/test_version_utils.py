# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to sys.path to import version_utils
repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

from version_utils import bump_major, bump_minor, bump_patch, parse_major_minor


class TestVersionUtils(unittest.TestCase):
    def test_parse_major_minor(self):
        self.assertEqual(parse_major_minor("1.1b0.dev"), (1, 1))
        self.assertEqual(parse_major_minor("1.2.3.dev"), (1, 2))
        self.assertEqual(parse_major_minor("10.5"), (10, 5))
        self.assertEqual(parse_major_minor("2.0b0"), (2, 0))

        with self.assertRaises(ValueError):
            parse_major_minor("invalid")
        with self.assertRaises(ValueError):
            parse_major_minor("1")

    def test_bump_minor(self):
        self.assertEqual(bump_minor("1.0.5"), "1.1.0.dev")
        self.assertEqual(bump_minor("1.0.5.dev"), "1.1.0.dev")
        self.assertEqual(bump_minor("1.0b3.dev"), "1.1b0.dev")
        self.assertEqual(bump_minor("1.0b3"), "1.1b0.dev")

        with self.assertRaises(ValueError):
            bump_minor("invalid")

    def test_bump_major(self):
        self.assertEqual(bump_major("1.5b2.dev"), "2.0b0.dev")
        self.assertEqual(bump_major("1.5b2"), "2.0b0.dev")
        self.assertEqual(bump_major("1.5.3"), "2.0.0.dev")
        self.assertEqual(bump_major("1.5.3.dev"), "2.0.0.dev")

        with self.assertRaises(ValueError):
            bump_major("invalid")

    def test_bump_patch(self):
        self.assertEqual(bump_patch("1.2.3"), "1.2.4")
        self.assertEqual(bump_patch("1.2.3.dev"), "1.2.4")
        self.assertEqual(bump_patch("1.1b0"), "1.1b1")
        self.assertEqual(bump_patch("1.1b0.dev"), "1.1b1")
        self.assertEqual(bump_patch("1.1.0"), "1.1.1")

        with self.assertRaises(ValueError):
            bump_patch("invalid")


class TestVersionUtilsCheck(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.ini_path = self.repo_root / "eachdist.ini"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_eachdist_ini(self, version: str):
        content = f"""[prerelease]
version={version}
"""
        self.ini_path.write_text(content, encoding="utf-8")

    def create_mock_package(
        self,
        name: str,
        relative_version_path: str | None,
        version_content: str | None = None,
    ):
        pkg_dir = self.repo_root / "instrumentation" / name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        pyproject_content = f'[project]\nname = "{name}"\n'
        if relative_version_path:
            pyproject_content += (
                f'[tool.hatch.version]\npath = "{relative_version_path}"\n'
            )

        (pkg_dir / "pyproject.toml").write_text(
            pyproject_content, encoding="utf-8"
        )

        if relative_version_path and version_content is not None:
            version_file = pkg_dir / relative_version_path
            version_file.parent.mkdir(parents=True, exist_ok=True)
            version_file.write_text(version_content, encoding="utf-8")

    def test_run_check_happy_path(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package(
            "pkg-a", "src/pkg_a/version.py", '__version__ = "1.1b0.dev"'
        )
        self.create_mock_package(
            "pkg-b", "src/pkg_b/version.py", '__version__ = "1.1.5b1"'
        )
        self.create_mock_package("pkg-c", None)

        from version_utils import run_check

        self.assertEqual(run_check(self.repo_root, self.ini_path), 0)

    def test_run_check_mismatched_minor(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package(
            "pkg-a", "src/pkg_a/version.py", '__version__ = "1.2.0"'
        )

        from version_utils import run_check

        self.assertEqual(run_check(self.repo_root, self.ini_path), 1)

    def test_run_check_missing_version_file(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package("pkg-a", "src/pkg_a/version.py", None)

        from version_utils import run_check

        self.assertEqual(run_check(self.repo_root, self.ini_path), 1)


class TestVersionUtilsBump(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.ini_path = self.repo_root / "eachdist.ini"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_eachdist_ini(self, version: str):
        content = f"""[prerelease]
version={version}
"""
        self.ini_path.write_text(content, encoding="utf-8")

    def create_mock_package(
        self,
        name: str,
        relative_version_path: str | None,
        version_content: str | None = None,
    ):
        pkg_dir = self.repo_root / "instrumentation" / name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        pyproject_content = f'[project]\nname = "{name}"\n'
        if relative_version_path:
            pyproject_content += (
                f'[tool.hatch.version]\npath = "{relative_version_path}"\n'
            )

        (pkg_dir / "pyproject.toml").write_text(
            pyproject_content, encoding="utf-8"
        )

        if relative_version_path and version_content is not None:
            version_file = pkg_dir / relative_version_path
            version_file.parent.mkdir(parents=True, exist_ok=True)
            version_file.write_text(version_content, encoding="utf-8")

    def get_version_file_content(
        self, name: str, relative_version_path: str
    ) -> str:
        version_file = (
            self.repo_root / "instrumentation" / name / relative_version_path
        )
        return version_file.read_text(encoding="utf-8")

    def test_bump_single_package(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package(
            "pkg-a", "src/pkg_a/version.py", '__version__ = "1.1b0.dev"'
        )
        self.create_mock_package(
            "pkg-b", "src/pkg_b/version.py", '__version__ = "1.1b0.dev"'
        )

        from version_utils import get_repo_version
        from version_utils import main as version_utils_main

        # Bump only pkg-a
        ret = version_utils_main(
            argv=["bump", "--package", "pkg-a", "--patch"],
            repo_root=self.repo_root,
        )
        self.assertEqual(ret, 0)

        # pkg-a should be bumped to patch version 1.1b1
        pkg_a_content = self.get_version_file_content(
            "pkg-a", "src/pkg_a/version.py"
        )
        self.assertIn('__version__ = "1.1b1"', pkg_a_content)

        # pkg-b should remain unchanged
        pkg_b_content = self.get_version_file_content(
            "pkg-b", "src/pkg_b/version.py"
        )
        self.assertIn('__version__ = "1.1b0.dev"', pkg_b_content)

        # eachdist.ini version should NOT be updated
        self.assertEqual(get_repo_version(self.ini_path), "1.1b0.dev")

    def test_bump_repo_wide(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package(
            "pkg-a", "src/pkg_a/version.py", '__version__ = "1.1b0.dev"'
        )
        self.create_mock_package(
            "pkg-b", "src/pkg_b/version.py", '__version__ = "1.1b0.dev"'
        )
        self.create_mock_package(
            "pkg-c", None
        )  # Ignored non-dynamically versioned package

        from version_utils import get_repo_version
        from version_utils import main as version_utils_main

        # Bump repo-wide to next minor dev version
        ret = version_utils_main(
            argv=["bump", "--dev"], repo_root=self.repo_root
        )
        self.assertEqual(ret, 0)

        # eachdist.ini should be updated to next dev version
        self.assertEqual(get_repo_version(self.ini_path), "1.2b0.dev")

        # pkg-a & pkg-b should be updated to 1.2b0.dev
        pkg_a_content = self.get_version_file_content(
            "pkg-a", "src/pkg_a/version.py"
        )
        self.assertIn('__version__ = "1.2b0.dev"', pkg_a_content)

        pkg_b_content = self.get_version_file_content(
            "pkg-b", "src/pkg_b/version.py"
        )
        self.assertIn('__version__ = "1.2b0.dev"', pkg_b_content)

    def test_bump_missing_version_file(self):
        self.write_eachdist_ini("1.1b0.dev")
        self.create_mock_package("pkg-a", "src/pkg_a/version.py", None)

        from version_utils import main as version_utils_main

        ret = version_utils_main(
            argv=["bump", "--package", "pkg-a", "--patch"],
            repo_root=self.repo_root,
        )
        self.assertEqual(ret, 1)
