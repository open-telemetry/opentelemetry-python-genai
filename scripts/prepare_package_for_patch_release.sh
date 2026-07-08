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

version="$(./scripts/eachdist.py version --package "$package")"

version_file="$(find "$path" -type f -path "**/version.py")"
file_count="$(echo "$version_file" | wc -l | tr -d ' ')"

if [[ "$file_count" -ne 1 ]]; then
  echo "Error: expected one version file, found ${file_count}"
  echo "$version_file"
  exit 1
fi

if [[ "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  next_version="${major}.${minor}.$((patch + 1))"
elif [[ "$version" =~ ^([0-9]+)\.([0-9]+)b([0-9]+)$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  next_version="${major}.${minor}b$((patch + 1))"
else
  echo "unexpected version: '${version}'"
  exit 1
fi

sed -i -E "s/__version__\\s*=\\s*\"${version}\"/__version__ = \"${next_version}\"/g" "$version_file"

tox -e generate
towncrier build --yes --version "$next_version" --dir "$(dirname "$changelog")"

echo "Prepared ${package} for patch release v${next_version}"
