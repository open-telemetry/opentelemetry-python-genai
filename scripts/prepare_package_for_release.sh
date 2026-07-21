#!/bin/bash

# Prepare a single package for release on the current branch: drop the .dev suffix
# from version.py and run towncrier. Called from release prepare workflows.

set -euo pipefail

package="${1:?usage: prepare_package_for_release.sh PACKAGE}"

path="./$(./scripts/eachdist.py find-package --package "$package")"
changelog="${path}/CHANGELOG.md"

if [[ ! -f "$changelog" ]]; then
  echo "missing ${changelog}"
  exit 1
fi

version_dev="$(./scripts/eachdist.py version --package "$package")"

if [[ ! "$version_dev" =~ ^([0-9]+)\.([0-9]+)[\.|b]{1}([0-9]+).*\.dev$ ]]; then
  echo "unexpected version: ${version_dev}"
  exit 1
fi

version="${version_dev%.dev}"

version_file="$(find "$path" -type f -path "**/version.py")"
file_count="$(echo "$version_file" | wc -l | tr -d ' ')"

if [[ "$file_count" -ne 1 ]]; then
  echo "Error: expected one version file, found ${file_count}"
  echo "$version_file"
  exit 1
fi

sed -i -E "s/__version__\\s*=\\s*\"${version}\\.dev\"/__version__ = \"${version}\"/g" "$version_file"

uv run tox -e generate
uv run towncrier build --yes --version "$version" --dir "$(dirname "$changelog")"

echo "Prepared ${package} for release v${version}"
