#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys
from pathlib import Path

# We can import packaging as it is guaranteed to be in the environment
try:
    import tomllib
except ImportError:
    # Fallback to tomli if run on python < 3.11 outside tox (though tox-uv environment handles it)
    try:
        import tomli as tomllib
    except ImportError:
        print(
            "Error: tomllib (Python 3.11+) or tomli is required to run this script."
        )
        sys.exit(1)

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def get_package_version(pyproject_path):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    if "version" in project:
        return project["version"]

    # Dynamic version via hatch
    tool = data.get("tool", {})
    hatch = tool.get("hatch", {})
    hatch_version = hatch.get("version", {})
    version_path_relative = hatch_version.get("path")
    if not version_path_relative:
        raise ValueError(
            f"Could not find static version or hatch version path in {pyproject_path}"
        )

    pyproject_dir = os.path.dirname(pyproject_path)
    version_path = os.path.join(pyproject_dir, version_path_relative)
    with open(version_path, "r", encoding="utf-8") as vf:
        content = vf.read()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise ValueError(f"Could not find __version__ in {version_path}")


def get_lower_bound(requirement):
    lower_bound = None
    for spec in requirement.specifier:
        if spec.operator in (">=", "==", "~="):
            try:
                version = Version(spec.version)
            except Exception:
                try:
                    version = Version(re.sub(r"\.dev$", ".dev0", spec.version))
                except Exception:
                    continue
            if lower_bound is None or version > lower_bound:
                lower_bound = version
    return lower_bound


def normalize_dev_version(version_str):
    return re.sub(r"\.dev\d*$", "", version_str)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    errors = 0

    # Find all pyproject.toml files under instrumentation/ and util/
    pyprojects = []
    for pattern in [
        "instrumentation/*/pyproject.toml",
        "util/*/pyproject.toml",
    ]:
        pyprojects.extend(repo_root.glob(pattern))

    for pyproject_path in pyprojects:
        pkg_dir = pyproject_path.parent
        oldest_req_path = pkg_dir / "tests" / "requirements.oldest.txt"

        # Only check packages that have tests/requirements.oldest.txt
        if not oldest_req_path.exists():
            continue

        print(f"Checking consistency for: {pkg_dir.name}")

        # 1. Parse dependencies from pyproject.toml
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project_data = data.get("project", {})
        pyproject_deps = []

        # Direct dependencies
        if "dependencies" in project_data:
            pyproject_deps.extend(project_data["dependencies"])

        # Optional dependencies (e.g. instruments)
        opt_deps = project_data.get("optional-dependencies", {})
        for extra, deps in opt_deps.items():
            pyproject_deps.extend(deps)

        # Map canonicalized package name to its requirement object and lower bound
        pyproject_bounds = {}
        for dep_str in pyproject_deps:
            req = Requirement(dep_str)
            name = canonicalize_name(req.name)
            lower_bound = get_lower_bound(req)
            if lower_bound:
                pyproject_bounds[name] = (req, lower_bound)

        # 2. Parse requirements.oldest.txt
        pinned_versions = {}
        editable_installs = {}  # canonicalized name -> path string

        with open(oldest_req_path, "r", encoding="utf-8") as f:
            for line in f:
                # Strip inline comments
                line = line.split("#")[0].strip()
                if not line:
                    continue

                # Check for editable install
                if line.startswith("-e"):
                    # e.g., "-e util/opentelemetry-util-genai" or "-e instrumentation/pkg[instruments]"
                    path_str = line[2:].strip().split("[", 1)[0].strip()
                    target_pyproject = repo_root / path_str / "pyproject.toml"
                    if target_pyproject.exists():
                        try:
                            with open(target_pyproject, "rb") as pf:
                                target_data = tomllib.load(pf)
                            target_name = canonicalize_name(
                                target_data.get("project", {}).get("name", "")
                            )
                            if target_name:
                                editable_installs[target_name] = path_str
                        except Exception as e:
                            print(
                                f"  Error parsing editable target {target_pyproject}: {e}"
                            )
                    continue

                # Parse standard requirement pin
                try:
                    req = Requirement(line)
                    name = canonicalize_name(req.name)
                    # Extract the exact version pinned with '=='
                    pin = None
                    for spec in req.specifier:
                        if spec.operator == "==":
                            pin = Version(spec.version)
                            break
                    if pin:
                        pinned_versions[name] = pin
                except Exception:
                    # Ignore lines we can't parse as standard requirements (like -r, -c, etc.)
                    continue

        # 3. Compare pyproject.toml lower bounds with requirements.oldest.txt
        for name, (req, lower_bound) in pyproject_bounds.items():
            # Skip self-references in requirements.oldest.txt (e.g. -e instrumentation/...)
            if name == canonicalize_name(project_data.get("name", "")):
                continue

            if name in editable_installs:
                # Local workspace dependency
                target_path = editable_installs[name]
                target_pyproject = repo_root / target_path / "pyproject.toml"
                target_version_str = get_package_version(target_pyproject)

                # Normalize dev versions
                norm_lower_bound = normalize_dev_version(str(lower_bound))
                norm_target_version = normalize_dev_version(target_version_str)

                if norm_lower_bound != norm_target_version:
                    print(
                        f"  [ERROR] Workspace dependency '{name}' mismatch:\n"
                        f"    - pyproject.toml declares: {req}\n"
                        f"    - Workspace '{name}' version is: {target_version_str} (at {target_path})\n"
                        f"    - Expected lower bound in pyproject.toml to match workspace version (ignoring .dev suffix)"
                    )
                    errors += 1
            elif name in pinned_versions:
                # PyPI dependency
                pin = pinned_versions[name]
                if pin != lower_bound:
                    print(
                        f"  [ERROR] Dependency '{name}' mismatch:\n"
                        f"    - pyproject.toml declares lower bound: {lower_bound} ({req})\n"
                        f"    - requirements.oldest.txt pins: {pin}\n"
                        f"    - They must match exactly."
                    )
                    errors += 1
            else:
                # Warning: direct dependency exists in pyproject.toml but not pinned in requirements.oldest.txt
                # This could be a gap in testing the oldest version
                print(
                    f"  [WARNING] Dependency '{name}' is declared in pyproject.toml but has no matching pin "
                    f"or editable install in tests/requirements.oldest.txt"
                )

    if errors > 0:
        print(f"\nFound {errors} inconsistency error(s).", file=sys.stderr)
        sys.exit(1)

    print("\nAll oldest dependency checks passed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
