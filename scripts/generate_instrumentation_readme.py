#!/usr/bin/env python3

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("instrumentation_readme_generator")

_prefix = "opentelemetry-instrumentation-"


def get_release_workflow_packages(root_path: str) -> set[str]:
    packages = set()

    # 1. Parse .github/workflows/release-package.yml options
    workflow_path = os.path.join(
        root_path,
        ".github",
        "workflows",
        "release-package.yml",
    )
    if os.path.exists(workflow_path):
        with open(workflow_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- opentelemetry-"):
                    pkg_name = line.lstrip("- ").strip("'\"")
                    packages.add(pkg_name)

    # 2. Parse eachdist.ini release_packages
    ini_path = os.path.join(root_path, "eachdist.ini")
    if os.path.exists(ini_path):
        with open(ini_path, "r", encoding="utf-8") as f:
            in_release_packages = False
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    in_release_packages = line == "[release_packages]"
                    continue
                if in_release_packages and line.startswith("opentelemetry-"):
                    packages.add(line)

    return packages


def get_pypi_status(package_name: str) -> str:
    url = f"https://pypi.org/pypi/{package_name}/json"
    req = urllib.request.Request(
        url, headers={"User-Agent": "opentelemetry-readme-generator"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            version = data.get("info", {}).get("version")
            if version:
                return f"[{version}](https://pypi.org/project/{package_name}/)"
            return "skeleton"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "skeleton"
        raise


def main(base_instrumentation_path):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(root_path, "README.md")

    released_rows = []
    unreleased_rows = []

    release_workflow_packages = get_release_workflow_packages(root_path)

    for instrumentation in sorted(os.listdir(base_instrumentation_path)):
        instrumentation_path = os.path.join(
            base_instrumentation_path, instrumentation
        )
        if not os.path.isdir(
            instrumentation_path
        ) or not instrumentation.startswith(_prefix):
            continue

        src_dir = os.path.join(
            instrumentation_path, "src", "opentelemetry", "instrumentation"
        )
        genai_dir = os.path.join(src_dir, "genai")
        if os.path.isdir(genai_dir):
            src_dir = genai_dir
        src_pkgs = [
            f
            for f in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, f))
        ]
        assert len(src_pkgs) == 1
        name = src_pkgs[0]

        pkg_info = {}
        version_filename = os.path.join(
            src_dir,
            name,
            "package.py",
        )
        with open(version_filename, encoding="utf-8") as fh:
            exec(fh.read(), pkg_info)

        instruments_and = pkg_info.get("_instruments", ())
        instruments_any = pkg_info.get("_instruments_any", ())
        instruments_all = ()
        if not instruments_and and not instruments_any:
            instruments_all = (name,)
        else:
            instruments_all = tuple(instruments_and + instruments_any)

        # Get local package version from version.py
        version_info = {}
        version_py_path = os.path.join(src_dir, name, "version.py")
        if os.path.exists(version_py_path):
            with open(version_py_path, encoding="utf-8") as fh:
                exec(fh.read(), version_info)
        local_version = version_info.get("__version__", "")

        status = get_pypi_status(instrumentation)

        if status == "skeleton":
            if instrumentation in release_workflow_packages:
                unreleased_status = "to be released"
            else:
                unreleased_status = "skeleton"
            unreleased_rows.append(
                f"| [{instrumentation}](./instrumentation/{instrumentation}) | {', '.join(instruments_all)} | {local_version} | {unreleased_status} |"
            )
        else:
            released_rows.append(
                f"| [{instrumentation}](./instrumentation/{instrumentation}) | {', '.join(instruments_all)} | {status} |"
            )

    generated_content = []

    if released_rows:
        generated_content.append("## Released Instrumentations\n")
        generated_content.append(
            "| Instrumentation | Supported Package | Version |"
        )
        generated_content.append(
            "| --------------- | ----------------- | ------- |"
        )
        generated_content.extend(released_rows)
        generated_content.append("")

    if unreleased_rows:
        generated_content.append("## Unreleased Instrumentations\n")
        generated_content.append(
            "| Instrumentation | Supported Package | Version | Status |"
        )
        generated_content.append(
            "| --------------- | ----------------- | ------- | ------ |"
        )
        generated_content.extend(unreleased_rows)
        generated_content.append("")

    table_block = "\n".join(generated_content)

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            readme_text = fh.read()

        pattern = r"(<!--\s*instrumentations\s*-->)(.*?)(<!--\s*end\s*-->)"
        replacement = f"\\1\n{table_block}\\3"
        new_readme_text, count = re.subn(
            pattern, replacement, readme_text, flags=re.DOTALL
        )

        if count > 0:
            with open(readme_path, "w", encoding="utf-8") as fh:
                fh.write(new_readme_text)
        else:
            logger.warning(
                "Could not find <!-- instrumentations --> ... <!-- end --> in README.md"
            )

    # Clean up instrumentation/README.md if present
    instr_readme = os.path.join(base_instrumentation_path, "README.md")
    if os.path.exists(instr_readme):
        os.remove(instr_readme)


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instrumentation_path = os.path.join(root_path, "instrumentation")
    main(instrumentation_path)
