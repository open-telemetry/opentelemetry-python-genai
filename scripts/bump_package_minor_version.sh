#!/bin/bash

# Bump a package to the next minor .dev version. Called by the bump-minor
# workflow when a maintainer wants to move a package to the next minor line
# (e.g., 1.0b3.dev -> 1.1b0.dev, or 1.0.5 -> 1.1.0.dev).
#
# Accepts the current version whether or not it carries a .dev suffix.

set -euo pipefail

package="${1:?usage: bump_package_minor_version.sh PACKAGE}"

path="./$(./scripts/eachdist.py find-package --package "$package")"
version="$(./scripts/eachdist.py version --package "$package")"
version_file="$(find "$path" -type f -path "**/version.py")"
file_count="$(echo "$version_file" | wc -l | tr -d ' ')"

if [[ "$file_count" -ne 1 ]]; then
  echo "Error: expected one version file, found ${file_count}"
  echo "$version_file"
  exit 1
fi

base_version="${version%.dev}"

if [[ "$base_version" =~ ^([0-9]+)\.([0-9]+)\.[0-9]+$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  next_version="${major}.$((minor + 1)).0.dev"
elif [[ "$base_version" =~ ^([0-9]+)\.([0-9]+)b[0-9]+$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  next_version="${major}.$((minor + 1))b0.dev"
else
  echo "unexpected version: ${version}"
  exit 1
fi

sed -i -E "s/__version__\\s*=\\s*\"${version}\"/__version__ = \"${next_version}\"/g" "$version_file"
echo "Bumped ${package} from ${version} to ${next_version}"
