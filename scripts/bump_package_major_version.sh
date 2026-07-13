#!/bin/bash

# Bump a package to the next major .dev version. Called by the bump-major
# workflow when a maintainer wants to move a package to the next major line
# (e.g., 1.5b2.dev -> 2.0b0.dev, or 1.5.3 -> 2.0.0.dev).
#
# Accepts the current version whether or not it carries a .dev suffix.

set -euo pipefail

package="${1:?usage: bump_package_major_version.sh PACKAGE}"

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

if [[ "$base_version" =~ ^([0-9]+)\.[0-9]+\.[0-9]+$ ]]; then
  major="${BASH_REMATCH[1]}"
  next_version="$((major + 1)).0.0.dev"
elif [[ "$base_version" =~ ^([0-9]+)\.[0-9]+b[0-9]+$ ]]; then
  major="${BASH_REMATCH[1]}"
  next_version="$((major + 1)).0b0.dev"
else
  echo "unexpected version: ${version}"
  exit 1
fi

sed -i -E "s/__version__\\s*=\\s*\"${version}\"/__version__ = \"${next_version}\"/g" "$version_file"
echo "Bumped ${package} from ${version} to ${next_version}"
