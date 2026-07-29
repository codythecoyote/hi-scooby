from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from .common import atomic_json, source_record


PREPARATION_SOURCE_KEYS = (
    "tiles",
    "cooler",
    "filtered_pairs",
    "membership",
    "contexts",
    "centroids",
    "gene_annotation",
    "ccre_registry",
    "fasta",
    "mappability",
    "embeddings",
    "embedding_manifest",
    "reference_topk",
)


def _verify_expected_annotation_releases(config: dict[str, Any]) -> None:
    annotations = config.get("annotations", {})
    expected = {
        "ccre_registry": ("md5", annotations.get("ccre_registry_md5")),
        "gene_annotation": ("sha256", annotations.get("gene_annotation_sha256")),
    }
    for key, (algorithm, expected_digest) in expected.items():
        if not expected_digest:
            continue
        digest = (
            hashlib.md5(usedforsecurity=False)
            if algorithm == "md5"
            else hashlib.sha256()
        )
        path = Path(config["paths"][key])
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        if digest.hexdigest() != str(expected_digest):
            raise RuntimeError(
                f"{key} does not match the configured release digest: {path}"
            )


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _directory_source_record(path: Path) -> dict[str, Any]:
    def quiet_sha256(file_path: Path, chunk_size: int = 8 << 20) -> str:
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                file_digest.update(chunk)
        return file_digest.hexdigest()

    files = []
    digest = hashlib.sha256()
    total_size = 0
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = child.stat()
        record = {
            "path": str(child.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": quiet_sha256(child),
        }
        relative = child.relative_to(path).as_posix()
        files.append({"relative_path": relative, **record})
        total_size += int(record["size_bytes"])
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(record["mtime_ns"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    if not files:
        raise ValueError(f"Required source directory is empty: {path}")
    return {
        "path": str(path.resolve()),
        "kind": "recursive_directory_manifest",
        "file_count": len(files),
        "total_size_bytes": total_size,
        "recursive_sha256": digest.hexdigest(),
        "files": files,
    }


def _contract_header(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    tracked_changes = _git_output(
        repo_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_changes:
        raise RuntimeError(
            "Preparation requires a clean tracked worktree so code provenance "
            f"is reproducible:\n{tracked_changes}"
        )
    return {
        "schema_version": int(config.get("schema_version", 2)),
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "config": source_record(Path(config["_config_path"])),
        "data_root": str(Path(config["outputs"]["data_root"])),
        "results_root": str(Path(config["outputs"]["results_root"])),
    }


def build_preparation_contract(config: dict[str, Any]) -> dict[str, Any]:
    header = _contract_header(config)
    _verify_expected_annotation_releases(config)
    sources = {}
    for key in PREPARATION_SOURCE_KEYS:
        path = Path(config["paths"][key])
        if not path.exists():
            raise FileNotFoundError(f"Required preparation source is missing: {path}")
        sources[key] = (
            _directory_source_record(path) if path.is_dir() else source_record(path)
        )
    return {**header, "sources": sources}


def create_preparation_contract(
    config: dict[str, Any], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Preparation contract already exists: {output}")
    contract = build_preparation_contract(config)
    atomic_json(output, contract)
    return contract


def verify_preparation_contract(
    config: dict[str, Any], contract_path: Path, *, verify_sources: bool = True
) -> dict[str, Any]:
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Preparation contract is missing: {contract_path}"
        )
    with contract_path.open() as handle:
        recorded = json.load(handle)
    if recorded.get("repair_lineage", {}).get("kind") == (
        "static_annotation_source_repair"
    ):
        _verify_repair_sources(config, recorded)
    current = (
        build_preparation_contract(config)
        if verify_sources
        else {**_contract_header(config), "sources": recorded.get("sources")}
    )
    recorded_core = {
        key: recorded.get(key)
        for key in (
            "schema_version", "git_commit", "config", "data_root",
            "results_root", "sources",
        )
    }
    if not recorded.get("sources") or recorded_core != current:
        raise RuntimeError(
            "Preparation provenance no longer matches the current commit, "
            "configuration, output roots, or source artifacts. Use a new "
            "output root instead of mixing artifacts."
        )
    return recorded


def _verify_recorded_source_metadata(path: Path, recorded: dict[str, Any]) -> None:
    if str(path.resolve()) != recorded.get("path"):
        raise RuntimeError(f"Source path changed since prior contract: {path}")
    if recorded.get("kind") == "recursive_directory_manifest":
        actual_files = {
            child.relative_to(path).as_posix(): child
            for child in path.rglob("*")
            if child.is_file()
        }
        expected_files = {
            row["relative_path"]: row for row in recorded.get("files", [])
        }
        if actual_files.keys() != expected_files.keys():
            raise RuntimeError(f"Source directory membership changed: {path}")
        for relative, child in actual_files.items():
            stat = child.stat()
            expected = expected_files[relative]
            if (
                stat.st_size != int(expected["size_bytes"])
                or stat.st_mtime_ns != int(expected["mtime_ns"])
            ):
                raise RuntimeError(f"Source file metadata changed: {child}")
        return
    stat = path.stat()
    if (
        stat.st_size != int(recorded["size_bytes"])
        or stat.st_mtime_ns != int(recorded["mtime_ns"])
    ):
        raise RuntimeError(f"Source file metadata changed: {path}")


def _verify_repair_sources(config: dict[str, Any], recorded: dict[str, Any]) -> None:
    _verify_expected_annotation_releases(config)
    recorded_sources = recorded.get("sources", {})
    if set(recorded_sources) != set(PREPARATION_SOURCE_KEYS):
        raise RuntimeError("Repaired contract does not contain the exact source set")
    for key in PREPARATION_SOURCE_KEYS:
        path = Path(config["paths"][key])
        expected = recorded_sources[key]
        _verify_recorded_source_metadata(path, expected)
        if key in {"gene_annotation", "ccre_registry"}:
            if source_record(path) != expected:
                raise RuntimeError(f"Repaired annotation source content changed: {path}")


def _verify_static_repair_config_delta(
    config: dict[str, Any], expected_prior_commit: str
) -> None:
    repo_root = Path(config["_repo_root"])
    prior_text = _git_output(
        repo_root,
        "show",
        f"{expected_prior_commit}:configs/raw_contact_ranker.yaml",
    )
    prior = yaml.safe_load(prior_text)
    approved_text = _git_output(
        repo_root,
        "show",
        "HEAD:configs/raw_contact_ranker.yaml",
    )
    approved = yaml.safe_load(approved_text)
    with Path(config["_config_path"]).open() as handle:
        current = yaml.safe_load(handle)
    if not all(isinstance(value, dict) for value in (prior, current, approved)):
        raise RuntimeError("Could not parse prior/current preparation configuration")
    for payload in (prior, current):
        payload.pop("paths", None)
        payload.pop("outputs", None)
    allowed_additions = {
        "annotations": {
            "gene_annotation_name",
            "gene_annotation_sha256",
            "ccre_registry_accession",
            "ccre_registry_url",
            "ccre_registry_md5",
        },
        "evaluation": {
            "minimum_supported_context_fraction",
            "required_anchor_strata",
            "descriptive_anchor_strata",
        },
    }
    missing = object()
    for section, keys in allowed_additions.items():
        values = current.get(section, {})
        approved_values = approved.get(section, {})
        for key in keys:
            if values.get(key, missing) != approved_values.get(key, missing):
                raise RuntimeError(
                    f"Static-annotation repair value differs from approved {section}.{key}"
                )
            values.pop(key, None)
    if current != prior:
        raise RuntimeError(
            "Static-annotation repair refuses unrelated configuration changes "
            f"from {expected_prior_commit}"
        )


def migrate_static_annotation_contract(
    config: dict[str, Any],
    contract_path: Path,
    *,
    expected_prior_commit: str,
) -> dict[str, Any]:
    if not contract_path.is_file():
        raise FileNotFoundError(f"Preparation contract is missing: {contract_path}")
    with contract_path.open() as handle:
        prior = json.load(handle)
    _verify_static_repair_config_delta(config, expected_prior_commit)
    header = _contract_header(config)
    lineage = prior.get("repair_lineage", {})
    if (
        prior.get("git_commit") == header["git_commit"]
        and lineage.get("kind") == "static_annotation_source_repair"
        and lineage.get("prior_git_commit") == expected_prior_commit
    ):
        for key in ("config", "data_root", "results_root"):
            if prior.get(key) != header[key]:
                raise RuntimeError(
                    f"Existing annotation-repair contract changed at {key}"
                )
        _verify_repair_sources(config, prior)
        return prior
    if prior.get("git_commit") != expected_prior_commit:
        raise RuntimeError(
            "Static-annotation repair expected prior commit "
            f"{expected_prior_commit}, found {prior.get('git_commit')}"
        )
    if (
        prior.get("data_root") != header["data_root"]
        or prior.get("results_root") != header["results_root"]
    ):
        raise RuntimeError("Repair cannot change preparation output roots")
    prior_sources = prior.get("sources", {})
    sources: dict[str, Any] = {}
    reused = []
    added = []
    for key in PREPARATION_SOURCE_KEYS:
        path = Path(config["paths"][key])
        if key in prior_sources:
            _verify_recorded_source_metadata(path, prior_sources[key])
            sources[key] = prior_sources[key]
            reused.append(key)
        else:
            if not path.exists():
                raise FileNotFoundError(f"Required repair source is missing: {path}")
            sources[key] = (
                _directory_source_record(path) if path.is_dir() else source_record(path)
            )
            added.append(key)
    backup = contract_path.with_name(
        "preparation_contract.before_static_annotation_repair.json"
    )
    if backup.exists():
        raise FileExistsError(f"Repair contract backup already exists: {backup}")
    shutil.copy2(contract_path, backup)
    migrated = {
        **header,
        "sources": sources,
        "repair_lineage": {
            "kind": "static_annotation_source_repair",
            "prior_git_commit": prior["git_commit"],
            "prior_contract_backup": str(backup.resolve()),
            "reused_source_keys": reused,
            "added_source_keys": added,
            "reused_source_validation": (
                "absolute path, directory membership, size, and mtime matched; "
                "prior SHA-256 records retained"
            ),
        },
    }
    atomic_json(contract_path, migrated)
    return migrated
