from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
import yaml


REQUIRED_PAIR_COLUMNS = (
    "pair_id",
    "chrom",
    "bin_i",
    "bin_j",
    "distance_bp",
    "distance_bin",
    "distance_band",
    "tile_id",
    "tile_row",
    "split",
)


DEFAULT_DISTANCE_BANDS = (
    {
        "id": "250-500",
        "minimum_bp_inclusive": 250_000,
        "maximum_bp_exclusive": 500_000,
    },
    {
        "id": "500-995",
        "minimum_bp_inclusive": 500_000,
        "maximum_bp_exclusive": 1_000_000,
    },
)


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    repo_root = path.parents[1]
    config["_config_path"] = str(path)
    config["_repo_root"] = str(repo_root)
    output_overrides = {
        "data_root": os.environ.get("RANKER_DATA_ROOT"),
        "results_root": os.environ.get("RANKER_RESULTS_ROOT"),
    }
    reference_override = os.environ.get("RANKER_REFERENCE_TOPK")
    if reference_override:
        config.setdefault("paths", {})["reference_topk"] = reference_override
    for key, value in output_overrides.items():
        if value:
            config.setdefault("outputs", {})[key] = value
    for section in ("paths", "outputs"):
        for key, value in config.get(section, {}).items():
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            config[section][key] = str(candidate.resolve())
    return config


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    show_progress = size >= 64 << 20
    if show_progress:
        from tqdm.auto import tqdm

        bar = tqdm(
            total=size,
            desc=f"Hash {path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
    else:
        bar = None
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if bar is not None:
                bar.update(len(chunk))
    if bar is not None:
        bar.close()
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file():
        result["sha256"] = sha256_file(path)
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def record_warning(
    warning_rows: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    project_breaking: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    row = {
        "code": code,
        "message": message,
        "project_breaking": project_breaking,
    }
    if details:
        row["details"] = details
    warning_rows.append(row)
    warnings.warn(f"{code}: {message}", stacklevel=2)


def enforce_or_warn(
    condition: bool,
    warning_rows: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    strict: bool,
    project_breaking: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    if condition:
        return
    if strict or project_breaking:
        raise RuntimeError(f"{code}: {message}")
    record_warning(
        warning_rows,
        code,
        message,
        project_breaking=project_breaking,
        details=details,
    )


def validate_pair_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_PAIR_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Pair table lacks required columns: {sorted(missing)}")
    if frame["pair_id"].duplicated().any():
        raise ValueError("pair_id is not unique")
    if frame.duplicated(["chrom", "bin_i", "bin_j"]).any():
        raise ValueError("Canonical genomic pair is duplicated")
    if np.any(frame["bin_i"].to_numpy() >= frame["bin_j"].to_numpy()):
        raise ValueError("Pair anchors are not strictly canonical")


def configured_distance_bands(
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, int | str], ...]:
    """Return validated half-open distance bands.

    The legacy 5 kb configuration predates explicit band metadata. Keeping a
    legacy default here preserves that reviewed behavior while new resolution
    profiles must declare their exact lattice contract.
    """
    source = (
        config.get("distance_bands", DEFAULT_DISTANCE_BANDS)
        if config is not None
        else DEFAULT_DISTANCE_BANDS
    )
    bands: list[dict[str, int | str]] = []
    previous_stop: int | None = None
    for row in source:
        band_id = str(row["id"])
        start = int(row["minimum_bp_inclusive"])
        stop = int(row["maximum_bp_exclusive"])
        if not band_id or start < 0 or stop <= start:
            raise ValueError(f"Invalid distance band: {row!r}")
        if previous_stop is not None and start != previous_stop:
            raise ValueError("Distance bands must be ordered and contiguous")
        bands.append(
            {
                "id": band_id,
                "minimum_bp_inclusive": start,
                "maximum_bp_exclusive": stop,
            }
        )
        previous_stop = stop
    if not bands:
        raise ValueError("At least one distance band is required")
    return tuple(bands)


def distance_range_bp(config: dict[str, Any]) -> tuple[int, int]:
    """Return the configured half-open modeled distance range."""
    minimum = int(config["minimum_distance_bp"])
    if "maximum_distance_bp_exclusive" in config:
        maximum_exclusive = int(config["maximum_distance_bp_exclusive"])
    else:
        maximum_exclusive = int(config["maximum_distance_bp"]) + int(
            config["bin_size_bp"]
        )
    if minimum < 0 or maximum_exclusive <= minimum:
        raise ValueError("Invalid modeled distance range")
    if minimum % int(config["bin_size_bp"]) or maximum_exclusive % int(
        config["bin_size_bp"]
    ):
        raise ValueError("Modeled distances must align to bin_size_bp")
    return minimum, maximum_exclusive


def distance_band(
    distance_bp: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    values = np.asarray(distance_bp, dtype=np.int64)
    output = np.full(values.shape, "", dtype=object)
    for row in configured_distance_bands(config):
        selected = (
            (values >= int(row["minimum_bp_inclusive"]))
            & (values < int(row["maximum_bp_exclusive"]))
        )
        if np.any(output[selected] != ""):
            raise ValueError("Configured distance bands overlap")
        output[selected] = str(row["id"])
    if np.any(output == ""):
        unknown = np.unique(values[output == ""])[:10].tolist()
        raise ValueError(f"Distances fall outside configured bands: {unknown}")
    return output.astype(str)


def resolution_contract(config: dict[str, Any]) -> dict[str, Any]:
    minimum, maximum_exclusive = distance_range_bp(config)
    return {
        "bin_size_bp": int(config["bin_size_bp"]),
        "minimum_distance_bp_inclusive": minimum,
        "maximum_distance_bp_exclusive": maximum_exclusive,
        "distance_bands": [dict(row) for row in configured_distance_bands(config)],
    }


def iter_parquet(path: Path, columns: Iterable[str] | None = None):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=list(columns) if columns else None):
        yield batch.to_pandas()


def selected_zarr_row(
    array,
    row: int,
    pair_ids: np.ndarray,
    *,
    pair_count: int,
    dtype,
) -> np.ndarray:
    """Read only authorized pair IDs from a row of a pair-aligned Zarr array."""
    if int(array.shape[1]) != int(pair_count):
        raise ValueError("Selected Zarr pair_count does not match the array")
    ids = np.asarray(pair_ids, np.int64)
    values = selected_zarr_values(array, row, ids, dtype=dtype)
    output = np.zeros(pair_count, dtype=dtype)
    output[ids] = values
    return output


def selected_zarr_values(
    array,
    row: int,
    pair_ids: np.ndarray,
    *,
    dtype,
) -> np.ndarray:
    """Read sorted pair IDs as contiguous runs, without touching other splits."""
    ids = np.asarray(pair_ids, np.int64)
    pair_count = int(array.shape[1])
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= pair_count):
        raise ValueError("Invalid selected Zarr pair IDs")
    if len(ids) and np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Selected Zarr pair IDs must be sorted and unique")
    output = np.empty(len(ids), dtype=dtype)
    if not len(ids):
        return output
    boundaries = np.flatnonzero(ids[1:] != ids[:-1] + 1) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(ids)]
    for start, stop in zip(starts, stops, strict=True):
        first = int(ids[start])
        last = int(ids[stop - 1]) + 1
        output[start:stop] = np.asarray(
            array[int(row), first:last], dtype=dtype
        )
    return output


def update_manifest(output_root: Path, section: str, payload: Any) -> None:
    path = output_root / "feature_manifest.json"
    if path.exists():
        with path.open() as handle:
            manifest = json.load(handle)
    else:
        manifest = {"schema_version": 1}
    manifest[section] = json_ready(payload)
    atomic_json(path, manifest)
