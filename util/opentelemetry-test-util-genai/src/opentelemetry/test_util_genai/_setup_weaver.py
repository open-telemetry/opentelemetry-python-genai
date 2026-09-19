# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision advice policies and the semconv registry for weaver.

The registry source is ``open-telemetry/semantic-conventions-genai``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import lzma
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Bounds the fetch of the registry tarballs so a slow/unreachable
# GitHub doesn't hang conformance runs until the OS-level socket timeout.
_FETCH_TIMEOUT_SECONDS = 60

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "versions.env").is_file() and (
            ancestor / "policies"
        ).is_dir():
            return ancestor
    raise RuntimeError(
        f"Could not locate the genai workspace root (walked up from {here} "
        "looking for versions.env + policies/)."
    )


def _load_version_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise RuntimeError(f"Invalid version pin in {path}: {raw_line!r}")
        pins[key.strip()] = value.strip().strip('"').strip("'")
    return pins


def _cache_dir() -> Path:
    override = os.environ.get("SEMCONV_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "otel-conformance" / "semconv"


def _download_and_extract(url: str, target: Path, label: str) -> None:
    """Download ``url`` (a .tar.gz) and extract its single top-level dir into ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(target.parent), prefix=f"{label}-"
    ) as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "src.tar.gz"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        logger.info("Fetching %s from %s", label, url)
        try:
            with (
                urllib.request.urlopen(
                    url, timeout=_FETCH_TIMEOUT_SECONDS
                ) as response,
                archive_path.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
        except (TimeoutError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"Failed to fetch {label} from {url}: {exc}"
            ) from exc
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")

        entries = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(entries) != 1:
            raise RuntimeError(
                f"Unexpected layout in {label} archive: "
                f"{[p.name for p in entries]}"
            )
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(entries[0]), str(target))


def _localize_manifest_dependencies(
    genai_root: Path, cache_root: Path
) -> None:
    """Download git dependencies as tarballs and rewrite registry_path to local dirs.

    Avoids Weaver cloning dependencies over git/HTTPS at runtime, which can
    hit network throttling or exceed WeaverLiveCheck startup timeouts in CI.
    """
    manifest = genai_root / "model" / "manifest.yaml"
    text = manifest.read_text(encoding="utf-8")
    pattern = re.compile(
        r"registry_path:\s*https://github\.com/open-telemetry/([^/]+)\.git@([^\s\[]+)(?:\[([^\]]+)\])?"
    )

    def _replace(match: re.Match[str]) -> str:
        repo_name = match.group(1)
        tag = match.group(2)
        subpath = match.group(3) or ""
        target = cache_root / f"{repo_name}-{tag}"
        if not target.is_dir():
            url = f"https://github.com/open-telemetry/{repo_name}/archive/refs/tags/{tag}.tar.gz"
            _download_and_extract(url, target, label=f"{repo_name}-{tag}")
        local_path = (target / subpath).resolve().as_posix()
        return f"registry_path: {local_path}"

    new_text, count = pattern.subn(_replace, text)
    if count > 0:
        manifest.write_text(new_text, encoding="utf-8")


def _asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    architecture = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    if system == "Darwin":
        return f"weaver-{architecture}-apple-darwin.tar.xz"
    if system == "Linux":
        return f"weaver-{architecture}-unknown-linux-gnu.tar.xz"
    if system == "Windows" and architecture == "x86_64":
        return "weaver-x86_64-pc-windows-msvc.zip"
    raise RuntimeError(f"Unsupported Weaver platform: {system} {machine}")


def ensure_weaver(version: str | None = None) -> Path:
    """Return the path to the Weaver executable, installing it if necessary."""
    override = os.environ.get("WEAVER")
    if override:
        return Path(override)

    if version is None:
        pins = _load_version_pins(_workspace_root() / "versions.env")
        version = pins.get("WEAVER_VERSION", "v0.26.1")

    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "opentelemetry-python-genai"
    )
    executable = (
        cache_root
        / "weaver"
        / version
        / ("weaver.exe" if platform.system() == "Windows" else "weaver")
    )
    if executable.is_file():
        return executable

    asset = _asset_name()
    base_url = (
        f"https://github.com/open-telemetry/weaver/releases/download/"
        f"{version}/{asset}"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / asset
        checksum_file = Path(temp_dir) / f"{asset}.sha256"
        with urllib.request.urlopen(base_url) as response:
            archive.write_bytes(response.read())
        with urllib.request.urlopen(f"{base_url}.sha256") as response:
            checksum_file.write_bytes(response.read())
        expected = checksum_file.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {asset}")

        executable.parent.mkdir(parents=True, exist_ok=True)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as package:
                member = next(
                    name
                    for name in package.namelist()
                    if name.endswith("/weaver.exe")
                )
                executable.write_bytes(package.read(member))
        else:
            with lzma.open(archive) as compressed:
                with tarfile.open(fileobj=compressed) as package:
                    member = next(
                        item
                        for item in package.getmembers()
                        if item.isfile() and item.name.endswith("/weaver")
                    )
                    source = package.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            f"Unable to extract Weaver from {asset}"
                        )
                    executable.write_bytes(source.read())
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _materialize_dependency_attributes(genai_root: Path) -> None:
    weaver = str(ensure_weaver())

    model = genai_root / "model"
    with tempfile.TemporaryDirectory(dir=str(genai_root)) as tmp:
        schema_path = Path(tmp) / "registry.json"
        try:
            subprocess.run(
                [
                    weaver,
                    "registry",
                    "resolve",
                    "--registry",
                    str(model),
                    "--v2",
                    "--format",
                    "json",
                    "--output",
                    str(schema_path),
                    "--skip-policies",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr) from error
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

    references: set[tuple[str, str]] = set()
    registry = schema["registry"]
    for signal_type in ("spans", "metrics", "events"):
        for signal in registry[signal_type]:
            for attribute in signal.get("attributes", ()):
                source = attribute.get("provenance", {}).get("source")
                if source is not None:
                    references.add((source, attribute["key"]))

    attributes: list[dict[str, object]] = []
    definition_fields = (
        "key",
        "type",
        "examples",
        "brief",
        "note",
        "stability",
        "deprecated",
        "annotations",
    )
    for source, key in sorted(references, key=lambda item: item[1]):
        dependency = schema["dependencies"][source]["registry"]
        attribute = next(
            item for item in dependency["attributes"] if item["key"] == key
        )
        attributes.append(
            {
                field: attribute[field]
                for field in definition_fields
                if attribute.get(field) is not None
            }
        )

    overlay = model / "dependency-attributes.yaml"
    overlay.write_text(
        "file_format: definition/2\nattributes: "
        + json.dumps(attributes, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _provision_genai_root() -> Path:
    """Fetch the pinned genai registry and return its root."""
    pins = _load_version_pins(_workspace_root() / "versions.env")
    try:
        genai_ref = pins["SEMCONV_GENAI_REF"]
    except KeyError as missing:
        raise RuntimeError(
            f"versions.env is missing required pin {missing!s}"
        ) from missing

    cache_root = _cache_dir()
    genai_target = cache_root / f"genai-{genai_ref}"
    stamp = genai_target / ".provisioned"
    dependency_attributes = (
        genai_target / "model" / "dependency-attributes.yaml"
    )
    if stamp.is_file() and dependency_attributes.is_file():
        return genai_target

    if stamp.is_file():
        _materialize_dependency_attributes(genai_target)
        return genai_target

    cache_root.mkdir(parents=True, exist_ok=True)
    genai_archive_url = (
        "https://github.com/open-telemetry/semantic-conventions-genai/"
        f"archive/{genai_ref}.tar.gz"
    )
    _download_and_extract(
        genai_archive_url, genai_target, label="genai-semconv"
    )
    _localize_manifest_dependencies(genai_target, cache_root)
    _materialize_dependency_attributes(genai_target)
    stamp.touch()
    return genai_target


def policies_dir() -> Path:
    """Return the ``policies`` directory with the committed advice ``.rego`` files."""
    return _workspace_root() / "policies"


def advice_data_glob() -> str:
    """Return a ``weaver --advice-data`` glob of the GenAI content JSON schemas."""
    source = _provision_genai_root() / "model" / "gen-ai"
    # gen-ai-tool-definitions.json references the external draft-07 meta-schema,
    # which weaver's rego engine refuses to fetch at eval time; rewrite that one
    # $ref to a local "type": "object" in place (idempotent).
    schema = source / "gen-ai-tool-definitions.json"
    text = schema.read_text(encoding="utf-8")
    patched = text.replace(
        '"$ref": "http://json-schema.org/draft-07/schema#"',
        '"type": "object"',
    )
    if patched != text:
        schema.write_text(patched, encoding="utf-8")
    return str(source / "*.json")


def semconv_registry() -> Path:
    """Return the path to ``<semantic-conventions-genai>/model`` for the pinned ref."""
    return _provision_genai_root() / "model"


def weaver_config_file() -> Path:
    """Return the path to the workspace ``.weaver.toml``."""
    return _workspace_root() / ".weaver.toml"
