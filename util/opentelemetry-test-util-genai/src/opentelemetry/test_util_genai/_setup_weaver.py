# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision advice policies and the semconv registry for weaver.

The registry source is ``open-telemetry/semantic-conventions-genai``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
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
    if stamp.is_file():
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
