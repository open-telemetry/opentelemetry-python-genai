#!/usr/bin/env python3
# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Version utility tool for checking and bumping package versions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def find_package_dir(repo_root: Path, package_name: str) -> Path | None:
    for path in sorted(repo_root.glob("instrumentation/*")) + sorted(
        repo_root.glob("util/*")
    ):
        if path.is_dir() and path.name == package_name:
            return path
    return None


def get_version_file_path(package_dir: Path) -> Path | None:
    pyproject_path = package_dir / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version_rel_path = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("version", {})
        .get("path")
    )
    if not version_rel_path:
        return None
    return package_dir / version_rel_path


def get_version_from_file(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", content)
    if not match:
        raise ValueError(f"__version__ assignment not found in {file_path}")
    return match.group(1)


def set_version_in_file(file_path: Path, new_version: str) -> None:
    content = file_path.read_text(encoding="utf-8")
    new_content = re.sub(
        r"(__version__\s*=\s*[\"']).*?([\"'])",
        rf"\g<1>{new_version}\g<2>",
        content,
    )
    file_path.write_text(new_content, encoding="utf-8")


def get_repo_version(ini_path: Path) -> str:
    content = ini_path.read_text(encoding="utf-8")
    match = re.search(r"\[prerelease\]\s*\n\s*version\s*=\s*(\S+)", content)
    if not match:
        raise ValueError(f"version under [prerelease] not found in {ini_path}")
    return match.group(1)


def set_repo_version(ini_path: Path, new_version: str) -> None:
    content = ini_path.read_text(encoding="utf-8")
    prerelease_pos = content.find("[prerelease]")
    if prerelease_pos == -1:
        raise ValueError("Could not find [prerelease] section in eachdist.ini")

    version_match = re.search(r"version\s*=\s*(\S+)", content[prerelease_pos:])
    if not version_match:
        raise ValueError("Could not find version entry under [prerelease]")

    old_line = version_match.group(0)
    new_line = f"version={new_version}"
    updated_section = content[prerelease_pos:].replace(old_line, new_line, 1)
    new_content = content[:prerelease_pos] + updated_section
    ini_path.write_text(new_content, encoding="utf-8")


def parse_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^([0-9]+)\.([0-9]+)", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2))


def bump_minor(version: str) -> str:
    base = version.split(".dev")[0]
    match_semver = re.match(r"^([0-9]+)\.([0-9]+)\.[0-9]+$", base)
    if match_semver:
        major, minor = match_semver.groups()
        return f"{major}.{int(minor) + 1}.0.dev"
    match_beta = re.match(r"^([0-9]+)\.([0-9]+)b([0-9]+)$", base)
    if match_beta:
        major, minor, _ = match_beta.groups()
        return f"{major}.{int(minor) + 1}b0.dev"
    raise ValueError(f"Unexpected version format for minor bump: {version}")


def bump_major(version: str) -> str:
    base = version.split(".dev")[0]
    match_semver = re.match(r"^([0-9]+)\.[0-9]+\.[0-9]+$", base)
    if match_semver:
        major = match_semver.group(1)
        return f"{int(major) + 1}.0.0.dev"
    match_beta = re.match(r"^([0-9]+)\.[0-9]+b[0-9]+$", base)
    if match_beta:
        major = match_beta.group(1)
        return f"{int(major) + 1}.0b0.dev"
    raise ValueError(f"Unexpected version format for major bump: {version}")


def bump_patch(version: str) -> str:
    base = version.split(".dev")[0]
    match_semver = re.match(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$", base)
    if match_semver:
        major, minor, patch = match_semver.groups()
        return f"{major}.{minor}.{int(patch) + 1}"
    match_beta = re.match(r"^([0-9]+)\.([0-9]+)b([0-9]+)$", base)
    if match_beta:
        major, minor, patch = match_beta.groups()
        return f"{major}.{minor}b{int(patch) + 1}"
    raise ValueError(f"Unexpected version format for patch bump: {version}")


def run_check(repo_root: Path, ini_path: Path) -> int:
    try:
        target_version = get_repo_version(ini_path)
        target_major, target_minor = parse_major_minor(target_version)
    except Exception as err:
        print(
            f"Error reading/parsing target version from eachdist.ini: {err}",
            file=sys.stderr,
        )
        return 1

    pyprojects = sorted(
        repo_root.glob("instrumentation/*/pyproject.toml"),
    ) + sorted(repo_root.glob("util/*/pyproject.toml"))

    if not pyprojects:
        print("No packages with pyproject.toml found.", file=sys.stderr)
        return 1

    errors: list[str] = []

    for pyproject_path in pyprojects:
        pkg_dir = pyproject_path.parent
        try:
            version_file_path = get_version_file_path(pkg_dir)
            if version_file_path is None:
                continue
        except Exception as err:
            errors.append(f"Failed to parse {pyproject_path}: {err}")
            continue

        if not version_file_path.exists():
            errors.append(
                f"{version_file_path.relative_to(repo_root)} does not exist."
            )
            continue

        try:
            version = get_version_from_file(version_file_path)
            major, minor = parse_major_minor(version)
            if (major, minor) != (target_major, target_minor):
                errors.append(
                    f"{version_file_path.relative_to(repo_root)} version is {version}. "
                    f"Expected major.minor to be {target_major}.{target_minor} (defined in eachdist.ini)."
                )
        except Exception as err:
            errors.append(
                f"Error checking {version_file_path.relative_to(repo_root)}: {err}"
            )

    if errors:
        print("Version major/minor alignment check failed:", file=sys.stderr)
        for err in errors:
            print(f"  [ERROR] {err}", file=sys.stderr)
        return 1

    print(
        f"All version.py files are aligned with target major.minor ({target_major}.{target_minor})."
    )
    return 0


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Version utility tool for checking and bumping package versions."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to run"
    )

    # 'check' subcommand
    subparsers.add_parser(
        "check", help="Check major/minor version alignment against config."
    )

    # 'bump' subcommand
    bump_parser = subparsers.add_parser(
        "bump", help="Bump package or repository versions."
    )
    bump_parser.add_argument(
        "--package",
        help="Package to bump. If omitted, bumps the entire repository.",
    )
    group = bump_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--release",
        action="store_true",
        help="Prepare for release (drop .dev suffix).",
    )
    group.add_argument(
        "--dev",
        action="store_true",
        help="Bump to next dev version (minor).",
    )
    group.add_argument(
        "--patch",
        action="store_true",
        help="Bump patch component.",
    )
    group.add_argument(
        "--major",
        action="store_true",
        help="Bump major component.",
    )

    args = parser.parse_args(argv)

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    ini_path = repo_root / "eachdist.ini"

    if args.command == "check":
        return run_check(repo_root, ini_path)

    # Bump logic
    if args.package:
        pkg_dir = find_package_dir(repo_root, args.package)
        if not pkg_dir:
            print(
                f"Error: package '{args.package}' not found.", file=sys.stderr
            )
            return 1
        version_file = get_version_file_path(pkg_dir)
        if not version_file:
            print(
                f"Error: package '{args.package}' does not use dynamic versioning.",
                file=sys.stderr,
            )
            return 1
        if not version_file.exists():
            print(
                f"Error: version file '{version_file}' does not exist.",
                file=sys.stderr,
            )
            return 1
        current_version = get_version_from_file(version_file)
    else:
        current_version = get_repo_version(ini_path)

    if args.release:
        if not current_version.endswith(".dev"):
            print(
                f"Version {current_version} does not have .dev suffix. Nothing to do.",
                file=sys.stderr,
            )
            return 0
        next_version = current_version.removesuffix(".dev")
    elif args.dev:
        next_version = bump_minor(current_version)
    elif args.patch:
        next_version = bump_patch(current_version)
    else:  # --major
        next_version = bump_major(current_version)

    if args.package:
        print(
            f"Updating package '{args.package}' version from {current_version} to {next_version}"
        )
        set_version_in_file(version_file, next_version)
    else:
        print(
            f"Updating repository-wide version from {current_version} to {next_version}"
        )
        set_repo_version(ini_path, next_version)

        pyprojects = sorted(
            repo_root.glob("instrumentation/*/pyproject.toml"),
        ) + sorted(repo_root.glob("util/*/pyproject.toml"))

        for pyproject_path in pyprojects:
            try:
                pkg_version_file = get_version_file_path(pyproject_path.parent)
                if pkg_version_file is None:
                    continue
                print(
                    f"Updating {pkg_version_file.relative_to(repo_root)} to {next_version}"
                )
                set_version_in_file(pkg_version_file, next_version)
            except Exception as err:
                print(
                    f"Warning: failed to update {pyproject_path}: {err}",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
