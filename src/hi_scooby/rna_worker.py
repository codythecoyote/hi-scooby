"""Private RNA-SCVI inference worker for the historical RNA environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import rich.pretty  # Registers rich.pretty for scvi-tools 1.1.x.
import scvi

from hi_scooby.rna import (
    aggregate_cell_type_centroids,
    load_aligned_rna_counts,
)


EXPECTED_LATENT_DIM = 14


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a Hi-Scooby wide RNA-count TSV with the frozen "
            "historical RNA-SCVI model."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"RNA input does not exist: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"RNA-SCVI model directory does not exist: {model_path}"
        )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite RNA output: {output_path}"
        )

    genes_path = model_path / "encoder_genes.parquet"
    if not genes_path.is_file():
        raise FileNotFoundError(
            f"RNA-SCVI model lacks encoder_genes.parquet: {model_path}"
        )
    gene_frame = pd.read_parquet(genes_path)
    if tuple(gene_frame.columns) != ("gene_index", "gene"):
        raise ValueError(
            "encoder_genes.parquet must contain gene_index and gene"
        )
    expected_index = np.arange(len(gene_frame), dtype=np.int64)
    if not np.array_equal(
        gene_frame["gene_index"].to_numpy(np.int64),
        expected_index,
    ):
        raise ValueError("Encoder gene indices are not contiguous")
    genes = tuple(gene_frame["gene"].astype(str))

    print("[RNA-SCVI 1/4] Validating and aligning RNA counts", flush=True)
    encoder_input = load_aligned_rna_counts(input_path, genes)

    print("[RNA-SCVI 2/4] Constructing query AnnData", flush=True)
    adata = ad.AnnData(
        X=encoder_input.counts,
        obs=pd.DataFrame(
            {
                "cell_type": encoder_input.cell_types,
                "library_id": encoder_input.library_ids,
            },
            index=pd.Index(encoder_input.barcodes, name="barcode"),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    adata.obs["library_id"] = adata.obs["library_id"].astype("category")

    print("[RNA-SCVI 3/4] Loading frozen encoder on CPU", flush=True)
    model = scvi.model.SCVI.load(
        str(model_path),
        adata=adata,
        accelerator="cpu",
        device="auto",
    )
    latent = model.get_latent_representation(
        adata=adata,
        give_mean=True,
    ).astype(np.float32)
    expected_shape = (adata.n_obs, EXPECTED_LATENT_DIM)
    if latent.shape != expected_shape:
        raise RuntimeError(
            f"RNA-SCVI latent shape is {latent.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(latent).all():
        raise RuntimeError("RNA-SCVI latent embedding contains non-finite values")

    print("[RNA-SCVI 4/4] Aggregating and saving cell-type centroids", flush=True)
    centroids = aggregate_cell_type_centroids(
        latent,
        encoder_input.cell_types,
    )
    if int(centroids["n_cells"].sum()) != adata.n_obs:
        raise RuntimeError("RNA centroid counts do not cover every input cell")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        centroids.to_parquet(
            temporary_path,
            index=False,
            compression="zstd",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"[RNA-SCVI] Wrote {len(centroids)} cell-type centroids "
        f"covering {adata.n_obs:,} cells: {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
