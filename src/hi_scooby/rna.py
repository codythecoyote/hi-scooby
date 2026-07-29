"""Validation and loading of Hi-Scooby wide RNA-count inputs."""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm


_BARCODE_PATTERN = re.compile(
    r"^rna(?P<library_number>[0-9]+)_(?P<cell_barcode>.+)$"
)


@dataclass(frozen=True)
class RNAInputSummary:
    """Validated structural metadata for a wide RNA-count table."""

    path: Path
    n_cells: int
    n_genes: int
    genes: tuple[str, ...]
    cell_type_counts: dict[str, int]
    library_ids: tuple[str, ...]


def inspect_wide_rna(path: str | Path) -> RNAInputSummary:
    """Validate the structural contract of a gzipped wide RNA table.

    The function scans the file without materializing its count matrix.
    Count values are parsed when the RNA encoder input is constructed.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"RNA input does not exist: {resolved}")

    seen_barcodes: set[str] = set()
    cell_type_counts: Counter[str] = Counter()
    library_ids: set[str] = set()
    n_cells = 0

    with gzip.open(resolved, "rt") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("RNA input is empty")

        header = header_line.rstrip("\n\r").split("\t")
        if header[:2] != ["barcode", "cell_type"]:
            raise ValueError(
                "RNA header must begin with: barcode, cell_type"
            )

        genes = tuple(header[2:])
        if not genes:
            raise ValueError("RNA input contains no gene columns")
        if any(not gene for gene in genes):
            raise ValueError("RNA input contains an empty gene name")
        if len(set(genes)) != len(genes):
            raise ValueError("RNA input contains duplicate gene names")

        expected_count_fields = len(genes)

        rows = tqdm(
            enumerate(handle, start=2),
            desc="Validating RNA",
            unit="cells",
            dynamic_ncols=True,
        )

        for line_number, line in rows:
            stripped = line.rstrip("\n\r")
            if not stripped:
                raise ValueError(f"Blank RNA row at line {line_number}")

            fields = stripped.split("\t", 2)
            if len(fields) != 3:
                raise ValueError(f"Malformed RNA row at line {line_number}")

            barcode, cell_type, count_text = fields
            if not barcode:
                raise ValueError(f"Missing barcode at line {line_number}")
            if barcode in seen_barcodes:
                raise ValueError(f"Duplicate barcode: {barcode}")
            seen_barcodes.add(barcode)

            if not cell_type:
                raise ValueError(
                    f"Missing cell_type for {barcode} at line {line_number}"
                )

            match = _BARCODE_PATTERN.fullmatch(barcode)
            if match is None:
                raise ValueError(
                    "Barcode does not match rnaNN_<barcode>: "
                    f"{barcode}"
                )

            library_number = int(match.group("library_number"))
            library_ids.add(f"DNA{library_number:02d}")

            observed_count_fields = count_text.count("\t") + 1
            if observed_count_fields != expected_count_fields:
                raise ValueError(
                    f"RNA row {line_number} has "
                    f"{observed_count_fields:,} count fields; expected "
                    f"{expected_count_fields:,}"
                )

            cell_type_counts[cell_type] += 1
            n_cells += 1

    if n_cells == 0:
        raise ValueError("RNA input contains no cell rows")

    return RNAInputSummary(
        path=resolved,
        n_cells=n_cells,
        n_genes=len(genes),
        genes=genes,
        cell_type_counts=dict(sorted(cell_type_counts.items())),
        library_ids=tuple(sorted(library_ids)),
    )


@dataclass(frozen=True)
class RNAEncoderInput:
    """RNA counts aligned to the frozen encoder's gene order."""

    counts: sparse.csr_matrix
    barcodes: tuple[str, ...]
    cell_types: tuple[str, ...]
    library_ids: tuple[str, ...]
    genes: tuple[str, ...]


def aggregate_cell_type_centroids(
    latent: np.ndarray,
    cell_types: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Average per-cell latent embeddings into stable cell-type centroids."""
    latent_array = np.asarray(latent, dtype=np.float32)
    labels = np.asarray(cell_types, dtype=object)

    if latent_array.ndim != 2:
        raise ValueError(
            f"latent must have shape [cells, dimensions]; "
            f"found {latent_array.shape}"
        )
    if latent_array.shape[0] != len(labels):
        raise ValueError(
            "latent rows and cell-type labels do not align: "
            f"{latent_array.shape[0]} versus {len(labels)}"
        )
    if latent_array.shape[0] == 0 or latent_array.shape[1] == 0:
        raise ValueError("latent must contain cells and dimensions")
    if not np.isfinite(latent_array).all():
        raise ValueError("latent contains non-finite values")
    if any(not str(label) for label in labels):
        raise ValueError("cell_types contains an empty label")

    latent_columns = [
        f"latent_{index}" for index in range(latent_array.shape[1])
    ]
    frame = pd.DataFrame(latent_array, columns=latent_columns)
    frame["cell_type"] = labels.astype(str)
    grouped = frame.groupby("cell_type", sort=True, observed=True)
    means = grouped[latent_columns].mean()
    counts = grouped.size()

    return pd.DataFrame(
        {
            "cell_type": means.index.astype(str),
            "n_cells": counts.loc[means.index].to_numpy(np.int64),
            "embedding": [
                row.astype(np.float32, copy=False)
                for row in means.to_numpy(np.float32)
            ],
        }
    ).reset_index(drop=True)


def load_aligned_rna_counts(
    path: str | Path,
    expected_genes: tuple[str, ...] | list[str],
) -> RNAEncoderInput:
    """Load RNA counts in the exact feature order expected by the encoder."""

    summary = inspect_wide_rna(path)
    genes = tuple(str(gene) for gene in expected_genes)

    if not genes:
        raise ValueError("Encoder gene list is empty")
    if len(set(genes)) != len(genes):
        raise ValueError("Encoder gene list contains duplicates")

    input_positions = {
        gene: position for position, gene in enumerate(summary.genes)
    }
    missing = [gene for gene in genes if gene not in input_positions]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"RNA input is missing {len(missing):,} encoder genes: {preview}"
        )

    selected_positions = np.asarray(
        [input_positions[gene] for gene in genes],
        dtype=np.int64,
    )

    barcodes: list[str] = []
    cell_types: list[str] = []
    library_ids: list[str] = []
    data_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    indptr = [0]

    with gzip.open(summary.path, "rt") as handle:
        next(handle)

        rows = tqdm(
            enumerate(handle, start=2),
            total=summary.n_cells,
            desc="Loading RNA counts",
            unit="cells",
            dynamic_ncols=True,
        )

        for line_number, line in rows:
            barcode, cell_type, count_text = (
                line.rstrip("\n\r").split("\t", 2)
            )

            counts = np.fromstring(
                count_text,
                sep="\t",
                dtype=np.int64,
            )
            if counts.size != summary.n_genes:
                raise ValueError(
                    f"RNA row {line_number} contains a malformed count value"
                )
            if np.any(counts < 0):
                raise ValueError(
                    f"RNA row {line_number} contains a negative count"
                )

            selected = counts[selected_positions]
            if selected.size and selected.max() > np.iinfo(np.int32).max:
                raise ValueError(
                    f"RNA row {line_number} exceeds int32 count range"
                )

            nonzero = np.flatnonzero(selected)
            data_chunks.append(selected[nonzero].astype(np.int32, copy=False))
            index_chunks.append(nonzero.astype(np.int32, copy=False))
            indptr.append(indptr[-1] + len(nonzero))

            match = _BARCODE_PATTERN.fullmatch(barcode)
            if match is None:
                raise ValueError(
                    f"Invalid barcode during count loading: {barcode}"
                )

            barcodes.append(barcode)
            cell_types.append(cell_type)
            library_number = int(match.group("library_number"))
            library_ids.append(f"DNA{library_number:02d}")

    if len(barcodes) != summary.n_cells:
        raise RuntimeError(
            "RNA row count changed between validation and loading"
        )

    data = (
        np.concatenate(data_chunks)
        if data_chunks
        else np.empty(0, dtype=np.int32)
    )
    indices = (
        np.concatenate(index_chunks)
        if index_chunks
        else np.empty(0, dtype=np.int32)
    )

    matrix = sparse.csr_matrix(
        (
            data,
            indices,
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(summary.n_cells, len(genes)),
        dtype=np.int32,
    )

    return RNAEncoderInput(
        counts=matrix,
        barcodes=tuple(barcodes),
        cell_types=tuple(cell_types),
        library_ids=tuple(library_ids),
        genes=genes,
    )
