#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Guard dependency invariants across packages.

1. Oldest dependency invariant:
   The oldest tox envs install the lowest versions of each package's *declared* deps straight from
   pyproject.toml via ``UV_RESOLUTION=lowest-direct`` (see AGENTS.md). pyproject.toml is therefore the
   single source of truth for those floors and running the env is what validates them. This script
   covers the two gaps a passing oldest env can't see:
   - Re-introduced drift: a pyproject-declared dep hand-pinned again in tests/requirements.oldest.txt.
     Such a pin silently overrides the derived floor and can drift from the declared bound; remove it.
   - Missing coverage: a package with an oldest tox factor (a tests/requirements.latest.txt) but no
     tests/requirements.oldest.txt at all, so its declared floors are never exercised.

2. Package runtime dependency specifier invariant:
   Instrumentation packages declare their target library dependencies in [project.optional-dependencies]
   `instruments` in pyproject.toml, and expose them at runtime via `_instruments` in `package.py`.
   These two must match to ensure runtime instrumentor compatibility checks and package metadata stay in sync.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def declared_dep_names(pyproject: dict) -> set[str]:
    """Canonical names of every dep declared in [project.dependencies] and optional-dependencies."""
    project = pyproject.get("project", {})
    names: set[str] = set()
    for dep in project.get("dependencies", []):
        names.add(canonicalize_name(Requirement(dep).name))
    for deps in project.get("optional-dependencies", {}).values():
        for dep in deps:
            names.add(canonicalize_name(Requirement(dep).name))
    return names


def pinned_names(oldest_req_path: Path) -> set[str]:
    """Canonical names of every pinned requirement in an oldest requirements file.

    Skips option lines (-e/-r/-c/--flag) and anything that isn't a parseable requirement.
    """
    names: set[str] = set()
    for raw in oldest_req_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            names.add(canonicalize_name(Requirement(line).name))
        except Exception:
            continue
    return names


def extract_instruments_from_package_py(
    package_py_path: Path,
) -> tuple[tuple[str, ...], str | None]:
    """Parse package.py with AST and extract _instruments value."""
    try:
        tree = ast.parse(
            package_py_path.read_text(encoding="utf-8"),
            filename=str(package_py_path),
        )
    except Exception as exc:
        return (), f"failed to parse {package_py_path}: {exc}"

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "_instruments"
                ):
                    try:
                        val = ast.literal_eval(stmt.value)
                    except Exception as exc:
                        return (
                            (),
                            f"failed to evaluate _instruments in {package_py_path}: {exc}",
                        )
                    if not isinstance(val, (list, tuple)):
                        return (
                            (),
                            f"_instruments in {package_py_path} must be a tuple or list",
                        )
                    return tuple(val), None
    return (), f"_instruments definition not found in {package_py_path}"


def check_instruments_match(pkg_dir: Path, pyproject: dict) -> list[str]:
    """Verify that package.py _instruments matches pyproject.toml [project.optional-dependencies] instruments."""
    errors: list[str] = []
    pyproject_instruments_raw = (
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get("instruments", None)
    )

    package_py_files = sorted((pkg_dir / "src").glob("**/package.py"))

    if pyproject_instruments_raw is None and not package_py_files:
        return errors

    if pyproject_instruments_raw is not None and not package_py_files:
        errors.append(
            f"{pkg_dir.name}: pyproject.toml declares instruments extra, but no package.py found under src/"
        )
        return errors

    if pyproject_instruments_raw is None and package_py_files:
        errors.append(
            f"{pkg_dir.name}: package.py found under src/, but pyproject.toml is missing [project.optional-dependencies] instruments"
        )
        return errors

    if len(package_py_files) > 1:
        errors.append(
            f"{pkg_dir.name}: found multiple package.py files under src/: "
            f"{[str(p.relative_to(pkg_dir)) for p in package_py_files]}"
        )
        return errors

    package_py_path = package_py_files[0]
    pkg_instruments_raw, parse_err = extract_instruments_from_package_py(
        package_py_path
    )
    if parse_err:
        errors.append(f"{pkg_dir.name}: {parse_err}")
        return errors

    try:
        pyproject_reqs = {Requirement(r) for r in pyproject_instruments_raw}
    except Exception as exc:
        errors.append(
            f"{pkg_dir.name}: invalid requirement in pyproject.toml instruments extra: {exc}"
        )
        return errors

    try:
        pkg_reqs = {Requirement(r) for r in pkg_instruments_raw}
    except Exception as exc:
        errors.append(
            f"{pkg_dir.name}: invalid requirement in package.py _instruments: {exc}"
        )
        return errors

    if pyproject_reqs != pkg_reqs:
        errors.append(
            f"{pkg_dir.name}: package.py _instruments ({list(pkg_instruments_raw)}) "
            f"does not match pyproject.toml instruments extra ({pyproject_instruments_raw})."
        )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    errors: list[str] = []

    pyprojects = sorted(
        repo_root.glob("instrumentation/*/pyproject.toml"),
    ) + sorted(repo_root.glob("util/*/pyproject.toml"))

    for pyproject_path in pyprojects:
        pkg_dir = pyproject_path.parent
        oldest = pkg_dir / "tests" / "requirements.oldest.txt"
        latest = pkg_dir / "tests" / "requirements.latest.txt"

        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

        # Check 1: instruments extra in pyproject.toml matches _instruments in package.py
        errors.extend(check_instruments_match(pkg_dir, pyproject))

        # Check 2: oldest dependency coverage and redundancy
        if not oldest.exists():
            # Only a gap if the package has an oldest tox factor, signalled by a latest file.
            if latest.exists():
                errors.append(
                    f"{pkg_dir.name}: has tests/requirements.latest.txt but no "
                    f"tests/requirements.oldest.txt - declared floors are never tested."
                )
            continue

        redundant = declared_dep_names(pyproject) & pinned_names(oldest)
        for name in sorted(redundant):
            errors.append(
                f"{pkg_dir.name}: '{name}' is declared in pyproject.toml and also pinned in "
                f"tests/requirements.oldest.txt. Remove the pin - the oldest env derives it from "
                f"the pyproject.toml floor via UV_RESOLUTION=lowest-direct."
            )

    if errors:
        print("Dependency check failed:\n", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1

    print(
        "Dependency checks passed: package.py matches pyproject.toml, no declared deps re-pinned, no missing coverage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
