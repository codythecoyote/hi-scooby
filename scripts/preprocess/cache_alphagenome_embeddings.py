#!/usr/bin/env python
"""Cache the pinned AlphaGenome mouse pair embeddings used for training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED_SHAPE = (512, 512, 128)
EXPECTED_DTYPE = np.dtype("float16")
EXPECTED_WINDOW_BP = 1_048_576


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen AlphaGenome pair-embedding cache shared by "
            "phase1_v2 and the sparse 10 kb ranker."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/external/mm10_embeddings/"
            "window_manifest.mm10.parquet"
        ),
    )
    parser.add_argument(
        "--phase1-tiles",
        type=Path,
        default=Path("data/processed/multiome/tiles.parquet"),
    )
    parser.add_argument(
        "--sparse-tiles",
        type=Path,
        default=Path(
            "data/processed/raw_contact_ranker_10kb_inputs/"
            "tiles.10kb.parquet"
        ),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/external/mm10_genome/mm10.fa"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/mm10_embeddings"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Optional local pinned AlphaGenome checkpoint. If omitted, the "
            "pinned Hugging Face revision is resolved."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate embeddings that already pass validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing cache files without loading AlphaGenome.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Development-only deterministic limit on required windows.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _load_required_windows(
    manifest_path: Path,
    tile_paths: tuple[Path, ...],
) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    required_manifest_columns = {
        "window_id",
        "chrom",
        "window_start",
        "window_end",
        "window_length_bp",
        "alphagenome_embedding_file",
        "is_within_chrom_bounds",
    }
    missing = required_manifest_columns - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Embedding manifest is missing columns: {sorted(missing)}"
        )
    if manifest["window_id"].duplicated().any():
        raise ValueError("Embedding manifest contains duplicate window IDs")
    if manifest["alphagenome_embedding_file"].duplicated().any():
        raise ValueError("Embedding manifest contains duplicate filenames")
    if not manifest["is_within_chrom_bounds"].astype(bool).all():
        raise ValueError("Embedding manifest contains an out-of-bounds window")
    if not manifest["window_length_bp"].eq(EXPECTED_WINDOW_BP).all():
        raise ValueError("Embedding manifest contains a non-1,048,576 bp window")

    tile_frames = []
    for tile_path in tile_paths:
        tiles = pd.read_parquet(
            tile_path,
            columns=["chrom", "input_start", "input_end"],
        )
        tile_frames.append(tiles)
    required_coordinates = (
        pd.concat(tile_frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["chrom", "input_start", "input_end"])
        .reset_index(drop=True)
    )
    if len(required_coordinates) == 0:
        raise ValueError("Tile tables contain no AlphaGenome windows")

    selected = required_coordinates.merge(
        manifest,
        left_on=["chrom", "input_start", "input_end"],
        right_on=["chrom", "window_start", "window_end"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = selected["_merge"].ne("both")
    if unmatched.any():
        examples = selected.loc[
            unmatched,
            ["chrom", "input_start", "input_end"],
        ].head().to_dict("records")
        raise ValueError(
            f"{int(unmatched.sum())} required windows are absent from the "
            f"manifest; examples: {examples}"
        )
    selected = selected.drop(columns=["_merge"])
    return selected


def _validate_embedding(path: Path, *, check_finite: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{path}: shape {array.shape}; expected {EXPECTED_SHAPE}"
        )
    if np.dtype(array.dtype) != EXPECTED_DTYPE:
        raise ValueError(
            f"{path}: dtype {array.dtype}; expected {EXPECTED_DTYPE}"
        )
    if check_finite and not np.isfinite(array).all():
        raise ValueError(f"{path}: embedding contains non-finite values")


def _write_embedding(path: Path, pair: np.ndarray) -> None:
    if pair.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"AlphaGenome returned shape {pair.shape}; expected "
            f"{EXPECTED_SHAPE}"
        )
    if not np.isfinite(pair).all():
        raise ValueError("AlphaGenome returned non-finite pair embeddings")
    stored = np.asarray(pair, dtype=EXPECTED_DTYPE)
    temporary = path.with_name(
        f".{path.stem}.partial-{os.getpid()}{path.suffix}"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, stored, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_embedding(temporary, check_finite=True)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    manifest_path = _resolve(args.manifest)
    phase1_tiles = _resolve(args.phase1_tiles)
    sparse_tiles = _resolve(args.sparse_tiles)
    fasta_path = _resolve(args.fasta)
    output_path = _resolve(args.output)
    for path, label in (
        (manifest_path, "embedding manifest"),
        (phase1_tiles, "phase-1 tile table"),
        (sparse_tiles, "sparse tile table"),
    ):
        _require_file(path, label)
    if not args.validate_only:
        _require_file(fasta_path, "mm10 FASTA")
    if args.checkpoint is not None:
        checkpoint_path = _resolve(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing AlphaGenome checkpoint: {checkpoint_path}"
            )
    else:
        checkpoint_path = None

    print("[1/3] Resolving required AlphaGenome windows", flush=True)
    windows = _load_required_windows(
        manifest_path,
        (phase1_tiles, sparse_tiles),
    )
    if args.limit is not None:
        windows = windows.iloc[: args.limit].copy()
        print(
            f"[development] Limiting cache operation to {len(windows)} "
            "window(s)",
            flush=True,
        )
    output_path.mkdir(parents=True, exist_ok=True)
    bytes_per_embedding = int(np.prod(EXPECTED_SHAPE)) * EXPECTED_DTYPE.itemsize
    print(
        f"      {len(windows):,} unique windows; "
        f"{bytes_per_embedding * len(windows) / (1024 ** 3):.1f} GiB "
        "uncompressed cache",
        flush=True,
    )

    existing = 0
    missing_rows = []
    print("[2/3] Checking the existing cache", flush=True)
    for row in tqdm(
        windows.itertuples(index=False),
        total=len(windows),
        desc="Validate embedding cache",
        unit="window",
    ):
        path = output_path / str(row.alphagenome_embedding_file)
        if path.exists() and not args.overwrite:
            _validate_embedding(path, check_finite=args.validate_only)
            existing += 1
        else:
            missing_rows.append(row)
    print(
        f"      {existing:,} valid existing; "
        f"{len(missing_rows):,} to generate",
        flush=True,
    )

    if args.validate_only:
        if missing_rows:
            examples = [
                str(row.alphagenome_embedding_file)
                for row in missing_rows[:5]
            ]
            raise FileNotFoundError(
                f"Cache is missing {len(missing_rows):,} required "
                f"embeddings; examples: {examples}"
            )
        print("[3/3] Existing embedding cache is valid", flush=True)
        return 0

    if not missing_rows:
        print("[3/3] Embedding cache is already complete", flush=True)
        return 0

    print("[3/3] Running pinned AlphaGenome forwards", flush=True)
    from hi_scooby.alphagenome import load_pair_embedder

    embedder = load_pair_embedder(
        fasta_path,
        checkpoint_path=checkpoint_path,
    )
    for row in tqdm(
        missing_rows,
        total=len(missing_rows),
        desc="Cache AlphaGenome embeddings",
        unit="window",
    ):
        pair = embedder.embed_interval(
            str(row.chrom),
            int(row.window_start),
            int(row.window_end),
        )
        _write_embedding(
            output_path / str(row.alphagenome_embedding_file),
            pair,
        )

    print(
        f"AlphaGenome embedding cache complete: {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
