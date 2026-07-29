"""Resolve Hi-Scooby models and data declared in configs/resources.yaml."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ResourceError(RuntimeError):
    """Raised when a declared Hi-Scooby resource is unavailable or invalid."""


def _find_config() -> Path:
    override = os.environ.get("HI_SCOOBY_RESOURCES")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ResourceError(
                f"HI_SCOOBY_RESOURCES does not point to a file: {path}"
            )
        return path

    candidates = [Path.cwd() / "configs" / "resources.yaml"]
    candidates.extend(
        parent / "configs" / "resources.yaml"
        for parent in Path(__file__).resolve().parents
    )

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise ResourceError(
        "Could not locate configs/resources.yaml. Run Hi-Scooby from its "
        "repository or set HI_SCOOBY_RESOURCES to the configuration path."
    )


def _override_name(resource_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", resource_name).upper()
    return f"HI_SCOOBY_{normalized}"


class ResourceRegistry:
    """Loaded resource declarations with path and byte-size validation."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else _find_config()
        )

        with self.config_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)

        if not isinstance(document, Mapping):
            raise ResourceError(
                f"Invalid resource configuration: {self.config_path}"
            )
        if document.get("schema_version") != 1:
            raise ResourceError(
                "Unsupported resources.yaml schema_version: "
                f"{document.get('schema_version')!r}"
            )

        resources = document.get("resources")
        alphagenome = document.get("alphagenome")
        if not isinstance(resources, Mapping):
            raise ResourceError("resources.yaml is missing its resources mapping")
        if not isinstance(alphagenome, Mapping):
            raise ResourceError("resources.yaml is missing its alphagenome mapping")

        self._resources = dict(resources)
        self.alphagenome: dict[str, Any] = dict(alphagenome)
        self.root = (
            self.config_path.parent.parent
            if self.config_path.parent.name == "configs"
            else self.config_path.parent
        )
        external_root = os.environ.get("HI_SCOOBY_RESOURCE_ROOT")
        self.external_root = (
            Path(external_root).expanduser().resolve()
            if external_root
            else None
        )
        if self.external_root is not None and not self.external_root.is_dir():
            raise ResourceError(
                "HI_SCOOBY_RESOURCE_ROOT is not a directory: "
                f"{self.external_root}"
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._resources)

    def specification(self, name: str) -> dict[str, Any]:
        try:
            value = self._resources[name]
        except KeyError as error:
            raise ResourceError(f"Unknown resource: {name}") from error
        if not isinstance(value, Mapping):
            raise ResourceError(f"Invalid declaration for resource: {name}")
        return dict(value)

    def resolve(self, name: str, *, verify_bytes: bool = True) -> Path:
        specification = self.specification(name)
        override_variable = _override_name(name)
        override = os.environ.get(override_variable)

        raw_candidates: list[tuple[str, bool]] = []
        if override:
            raw_candidates.append((override, False))
        else:
            for field in ("release_path", "path", "source_path"):
                value = specification.get(field)
                if value:
                    raw_candidates.append((str(value), field == "release_path"))

        candidates: list[Path] = []
        for value, prefer_package in raw_candidates:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                candidates.append(candidate.resolve())
                continue

            roots = [self.root]
            if self.external_root is not None:
                if prefer_package:
                    roots.append(self.external_root)
                else:
                    roots.insert(0, self.external_root)
            candidates.extend((root / candidate).resolve() for root in roots)

        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            rendered = "\n".join(f"  - {path}" for path in candidates)
            status = specification.get("status", "required")
            raise ResourceError(
                f"Resource '{name}' is unavailable (status: {status}).\n"
                f"Checked:\n{rendered}\n"
                "Provide it at one of these paths, set "
                "HI_SCOOBY_RESOURCE_ROOT to the repository/resource root, "
                f"or set {override_variable}."
            )

        expected_bytes = specification.get("bytes")
        if verify_bytes and expected_bytes is not None and selected.is_file():
            actual_bytes = selected.stat().st_size
            if actual_bytes != int(expected_bytes):
                raise ResourceError(
                    f"Resource '{name}' has an unexpected size: "
                    f"{actual_bytes:,} bytes at {selected}; "
                    f"expected {int(expected_bytes):,} bytes."
                )

        print(f"[resources] {name}: {selected}", flush=True)
        return selected

    def resolve_for(
        self,
        purpose: str,
        *,
        verify_bytes: bool = True,
    ) -> dict[str, Path]:
        selected: dict[str, Path] = {}
        for name in self.names:
            specification = self.specification(name)
            if purpose in specification.get("required_for", ()):
                selected[name] = self.resolve(
                    name,
                    verify_bytes=verify_bytes,
                )
        return selected


def load_resources(config_path: Path | None = None) -> ResourceRegistry:
    """Load the configured Hi-Scooby resource registry."""
    return ResourceRegistry(config_path)
