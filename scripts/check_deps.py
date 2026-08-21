#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Guard dependency invariants across packages.

1. Oldest dependency invariant:
   The oldest tox envs install the lowest versions of each package's *declared* deps straight from
   pyproject.toml via ``UV_RESOLUTION=lowest-direct`` (see AGENTS.md). pyproject.toml is therefore the
   single source of truth for those floors and running the env is what validates them. This script
   covers gaps a passing oldest env can't see:
   - Re-introduced drift: a pyproject-declared dep hand-pinned again in tests/requirements.oldest.txt.
     Such a pin silently overrides the derived floor and can drift from the declared bound; remove it.
   - Missing coverage: a package with an oldest tox factor (a tests/requirements.latest.txt) but no
     tests/requirements.oldest.txt at all, so its declared floors are never exercised.
   - Workspace dependency floors and local installs:
     - If a workspace dependency floor in pyproject.toml is from a previous release cycle,
       it resolves and tests from PyPI; it must NOT be installed from local paths in
       oldest requirements.
     - If a workspace dependency floor matches the current unreleased workspace dev version,
       tests/requirements.oldest.txt must install it locally so tests can run against
       the local development version.

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
from packaging.version import Version

scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from version_utils import get_version_file_path, get_version_from_file


def get_declared_requirements(pyproject: dict) -> list[Requirement]:
    """Return all Requirements declared in [project.dependencies] and optional-dependencies."""
    project = pyproject.get("project", {})
    reqs: list[Requirement] = []
    for dep in project.get("dependencies", []):
        reqs.append(Requirement(dep))
    for deps in project.get("optional-dependencies", {}).values():
        for dep in deps:
            reqs.append(Requirement(dep))
    return reqs


def get_workspace_packages(repo_root: Path) -> dict[str, tuple[Path, str]]:
    """Return map of canonical_name -> (package_dir, current_version_str)."""
    packages: dict[str, tuple[Path, str]] = {}
    pyprojects = sorted(
        repo_root.glob("instrumentation/*/pyproject.toml")
    ) + sorted(repo_root.glob("util/*/pyproject.toml"))
    for pyproject_path in pyprojects:
        pkg_dir = pyproject_path.parent
        try:
            pyproject = tomllib.loads(
                pyproject_path.read_text(encoding="utf-8")
            )
            name = pyproject.get("project", {}).get("name")
            if not name:
                continue
            v_path = get_version_file_path(pkg_dir)
            if v_path and v_path.exists():
                version_str = get_version_from_file(v_path)
                packages[canonicalize_name(name)] = (pkg_dir, version_str)
        except Exception:
            continue
    return packages


def parse_local_workspace_lines(
    oldest_path: Path,
    repo_root: Path,
    workspace_packages: dict[str, tuple[Path, str]],
) -> dict[str, str]:
    """Find lines in a requirements file that install local workspace packages."""
    workspace_dirs = {
        pkg_dir.resolve(): name
        for name, (pkg_dir, _) in workspace_packages.items()
    }
    local_pkgs: dict[str, str] = {}

    for raw in oldest_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        path_str = line
        is_editable = line.startswith(("-e", "--editable"))
        if is_editable:
            parts = line.split(maxsplit=1)
            path_str = parts[1].strip() if len(parts) > 1 else ""

        path_clean = (
            path_str.split("[", 1)[0]
            .strip()
            .replace("{toxinidir}", str(repo_root))
        )

        matched_pkg: str | None = None
        for base in (oldest_path.parent, oldest_path.parent.parent, repo_root):
            resolved = (base / path_clean).resolve()
            if resolved in workspace_dirs:
                matched_pkg = workspace_dirs[resolved]
                break
            if (resolved / "pyproject.toml").is_file():
                try:
                    data = tomllib.loads(
                        (resolved / "pyproject.toml").read_text(
                            encoding="utf-8"
                        )
                    )
                    name = data.get("project", {}).get("name")
                    if name:
                        matched_pkg = canonicalize_name(name)
                        break
                except Exception:
                    pass

        if matched_pkg:
            local_pkgs[matched_pkg] = line
        elif is_editable:
            local_pkgs[line] = line

    return local_pkgs


def check_workspace_dependencies(
    pkg_dir: Path,
    pyproject: dict,
    oldest_path: Path | None,
    repo_root: Path,
    workspace_packages: dict[str, tuple[Path, str]],
) -> list[str]:
    """Verify declared workspace dependency floors match PyPI release state or current .dev version."""
    errors: list[str] = []

    local_pkgs = (
        parse_local_workspace_lines(oldest_path, repo_root, workspace_packages)
        if oldest_path and oldest_path.exists()
        else {}
    )
    valid_locals: set[str] = set()
    reported_locals: set[str] = set()

    declared_reqs = get_declared_requirements(pyproject)
    pkg_name = canonicalize_name(pyproject.get("project", {}).get("name", ""))

    for req in declared_reqs:
        canonical_name = canonicalize_name(req.name)
        if (
            canonical_name not in workspace_packages
            or canonical_name == pkg_name
        ):
            continue

        target_dir, current_version_str = workspace_packages[canonical_name]
        try:
            current_version = Version(current_version_str)
        except Exception:
            continue

        lower_bound_str: str | None = None
        for spec in req.specifier:
            if spec.operator in (">=", "==", "~="):
                lower_bound_str = spec.version
                break

        if lower_bound_str is None:
            errors.append(
                f"{pkg_dir.name}: workspace dependency '{req.name}' is missing a lower bound (e.g. '>= {current_version_str}')."
            )
            continue

        try:
            floor_version = Version(lower_bound_str)
        except Exception:
            errors.append(
                f"{pkg_dir.name}: invalid version specifier '{lower_bound_str}' for workspace dependency '{req.name}'."
            )
            continue

        # Floor is from a previous cycle, published on PyPI (e.g. 1.0b0, 1.1b0, 1.1b0.dev)
        if floor_version < current_version:
            if canonical_name in local_pkgs:
                reported_locals.add(canonical_name)
                errors.append(
                    f"{pkg_dir.name}: '{req.name}' declared floor is '{lower_bound_str}' (released version on PyPI), "
                    f"but tests/requirements.oldest.txt installs it locally with '{local_pkgs[canonical_name]}'. "
                    f"Remove the local/editable install so tests run against the declared floor from PyPI."
                )
        # Floor is the current unreleased workspace dev version (e.g. 1.2b0.dev)
        elif floor_version == current_version:
            if canonical_name not in local_pkgs:
                errors.append(
                    f"{pkg_dir.name}: declared floor for '{req.name}' is unreleased '{lower_bound_str}', "
                    f"but tests/requirements.oldest.txt is missing a local/editable install (-e). "
                    f"Add local/editable install to tests/requirements.oldest.txt while under development."
                )
            else:
                valid_locals.add(canonical_name)
        # Floor is higher than current workspace version
        else:
            errors.append(
                f"{pkg_dir.name}: declared floor '{lower_bound_str}' for '{req.name}' exceeds "
                f"current workspace version '{current_version_str}'."
            )

    for name, line in local_pkgs.items():
        if name not in valid_locals and name not in reported_locals:
            errors.append(
                f"{pkg_dir.name}: '{line}' in tests/requirements.oldest.txt is not permitted. "
                f"Local/editable installs in oldest requirements are only allowed for workspace dependencies declaring an unreleased .dev floor."
            )

    return errors


def declared_dep_names(pyproject: dict) -> set[str]:
    """Canonical names of every dep declared in [project.dependencies] and optional-dependencies."""
    return {
        canonicalize_name(r.name) for r in get_declared_requirements(pyproject)
    }


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

    workspace_packages = get_workspace_packages(repo_root)

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

        # Check 3: workspace dependencies floor and editable install validation
        errors.extend(
            check_workspace_dependencies(
                pkg_dir,
                pyproject,
                oldest,
                repo_root,
                workspace_packages,
            )
        )

    if errors:
        print("Dependency check failed:\n", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1

    print(
        "Dependency checks passed: package.py matches pyproject.toml, no declared deps re-pinned, no missing coverage, workspace dependency floors validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
