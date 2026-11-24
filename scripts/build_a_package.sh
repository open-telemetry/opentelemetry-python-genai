#!/bin/sh

# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script builds wheels for a single package during the release workflows
# (see .github/workflows/_release-build.yml). The resulting dist/ artifact is
# then published to PyPI via Trusted Publishing in the top-level caller.

set -ev

if [ -n "${PACKAGE_NAME:-}" ]; then
  pkg_name="${PACKAGE_NAME}"
  pkg_version="${VERSION}"
elif [ -n "${GITHUB_REF:-}" ]; then
  pkg_name_and_version="${GITHUB_REF#refs/tags/*}"
  pkg_name="${pkg_name_and_version%==*}"
  pkg_version="${pkg_name_and_version#opentelemetry-*==}"
else
  echo 'Failed to run script, missing workflow env variable GITHUB_REF or PACKAGE_NAME and VERSION.'
  exit 1
fi

basedir=$(git rev-parse --show-toplevel)
cd "$basedir"

# Ensure no untracked or modified files exist in the checkout before building
if [ -n "$(git status --porcelain)" ]; then
  echo "Error! Working tree is dirty before build:"
  git status --porcelain
  exit 1
fi

# Validate version against PEP 440 conventions: https://packaging.pypa.io/en/latest/version.html
uv run --locked --only-group packaging --project "$basedir" \
  python -c "import sys; from packaging.version import Version; Version(sys.argv[1])" "$pkg_version"

distdir="${basedir}/dist"
mkdir -p "$distdir"
rm -rf "${distdir:?}/"*

directory_with_package=$(./scripts/eachdist.py find-package --package "$pkg_name")

if [ ! -f "$directory_with_package/pyproject.toml" ]; then
  echo "Error! pyproject.toml not found for $pkg_name, can't build."
  exit 1
fi

cd "$directory_with_package"

uv run --locked --only-group build --project "$basedir" \
  uv build --sdist --no-build-isolation --offline --out-dir "$distdir"

cd "$distdir"

pkg_tar_gz_file="$(printf '%s' "$pkg_name" | sed 's/-/_/g')-${pkg_version}.tar.gz"

echo "Checking if $pkg_tar_gz_file exists in dist/ directory."

# print the list of files in current directory
ls

if [ ! -f "$pkg_tar_gz_file" ]; then
  echo 'Error! Tag version does not match version built using latest package files.'
  exit 1
fi

# Build a wheel for the source distribution
uv run --locked --only-group build --project "$basedir" \
  uv build --wheel --no-build-isolation --offline --out-dir "$distdir" "$pkg_tar_gz_file"

# Audit built wheel contents against startup-hook (.pth / sitecustomize.py)
# and *.data/ injection attacks (e.g., LiteLLM v1.107.0 supply-chain vector).
python3 - "$distdir" "$pkg_name" "$pkg_version" <<'PYEOF'
import pathlib
import sys
import zipfile

distdir = pathlib.Path(sys.argv[1])
pkg_name_norm = sys.argv[2].replace("-", "_")
pkg_version = sys.argv[3]
expected_dist_info = f"{pkg_name_norm}-{pkg_version}.dist-info/"

wheels = list(distdir.glob(f"{pkg_name_norm}-{pkg_version}-*.whl"))
if len(wheels) != 1:
    sys.exit(f"Error! Expected exactly 1 wheel in {distdir}, found: {wheels}")

with zipfile.ZipFile(wheels[0]) as zf:
    for name in zf.namelist():
        basename = pathlib.PurePosixPath(name).name
        if name.endswith(".pth") or basename in {"sitecustomize.py", "usercustomize.py"}:
            sys.exit(f"Error! Forbidden startup hook file in wheel: {name}")
        if not (name.startswith("opentelemetry/") or name.startswith(expected_dist_info)):
            sys.exit(
                f"Error! Unexpected top-level path in wheel: {name} "
                f"(expected 'opentelemetry/' or '{expected_dist_info}')"
            )

print(f"Verified wheel contents for {wheels[0].name} ({len(zf.namelist())} files).")
PYEOF
