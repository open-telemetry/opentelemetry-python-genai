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

uv run ./scripts/version_utils.py bump --package "$package" --release

uv run tox -e generate
uv run towncrier build --yes --version "$version" --dir "$(dirname "$changelog")"

echo "Prepared ${package} for release v${version}"
