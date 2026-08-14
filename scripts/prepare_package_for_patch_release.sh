#!/bin/bash

# Prepare a patch release for a package on a backport branch: bump the patch
# version and run towncrier. Expects the current version without a .dev suffix.

set -euo pipefail

package="${1:?usage: prepare_package_for_patch_release.sh PACKAGE}"

path="./$(./scripts/eachdist.py find-package --package "$package")"
changelog="${path}/CHANGELOG.md"

if [[ ! -f "$changelog" ]]; then
  echo "missing ${changelog}"
  exit 1
fi
uv run ./scripts/version_utils.py bump --package "$package" --patch

next_version="$(./scripts/eachdist.py version --package "$package")"

uv run tox -e generate
uv run towncrier build --yes --version "$next_version" --dir "$(dirname "$changelog")"

echo "Prepared ${package} for patch release v${next_version}"
