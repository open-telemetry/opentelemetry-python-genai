#!/bin/bash

# Bump a package from a released version (no .dev suffix) to the next .dev
# version by incrementing the patch component. Called after a release
# publishes to move version.py forward.
#
# Major and minor bumps are maintainer-led and go through the dedicated
# bump_package_minor_version.sh / bump_package_major_version.sh scripts and
# their corresponding workflows.

set -euo pipefail

package="${1:?usage: bump_package_dev_version.sh PACKAGE}"

path="./$(./scripts/eachdist.py find-package --package "$package")"
version="$(./scripts/eachdist.py version --package "$package")"
version_file="$(find "$path" -type f -path "**/version.py")"

if [[ "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  next_version="${major}.${minor}.$((patch + 1)).dev"
elif [[ "$version" =~ ^([0-9]+)\.([0-9]+)b([0-9]+)$ ]]; then
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  next_version="${major}.${minor}b$((patch + 1)).dev"
else
  echo "unexpected version: ${version}"
  exit 1
fi

sed -i -E "s/__version__\\s*=\\s*\"${version}\"/__version__ = \"${next_version}\"/g" "$version_file"
echo "Bumped ${package} from ${version} to ${next_version}"
