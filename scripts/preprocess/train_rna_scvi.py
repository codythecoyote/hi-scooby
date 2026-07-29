#!/usr/bin/env python
"""Rebuild the frozen 14D RNA-SCVI encoder and cell-type centroids."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rich.pretty  # Registers rich.pretty for scvi-tools 1.1.x.
import scvi
import torch


EXPECTED_CELLS = 3_802
EXPECTED_GENES = 7_652
EXPECTED_LIBRARIES = 39
EXPECTED_CELL_TYPES = 22
LATENT_DIM = 14
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the historical 14D RNA-SCVI encoder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/multiome/multivi_input.h5ad"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/rna_scvi_14"),
    )
    parser.add_argument(
        "--reference-centroids",
        type=Path,
        default=Path(
            "data/processed/multiome/train_in/"
            "rna_scvi_14_cell_type_centroids.parquet"
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=("gpu", "cpu"),
        default="gpu",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--centroid-tolerance",
        type=float,
        default=1e-4,
    )
    return parser.parse_args()


def build_centroids(
    latent: np.ndarray,
    cell_types: pd.Series,
) -> pd.DataFrame:
    latent_columns = [f"latent_{index}" for index in range(latent.shape[1])]
    latent_by_cell = pd.DataFrame(latent, columns=latent_columns)
    latent_by_cell["cell_type"] = cell_types.astype(str).to_numpy()

    centroid_wide = (
        latent_by_cell.groupby("cell_type", sort=True)[latent_columns].mean()
    )
    cell_counts = latent_by_cell.groupby("cell_type", sort=True).size()

    return pd.DataFrame(
        {
            "cell_type": centroid_wide.index,
            "n_cells": cell_counts.loc[centroid_wide.index].to_numpy(),
            "embedding": [
                row.astype(np.float32)
                for row in centroid_wide.to_numpy()
            ],
        }
    ).reset_index(drop=True)


def compare_centroids(
    rebuilt: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance: float,
) -> dict[str, object]:
    rebuilt_types = tuple(rebuilt["cell_type"].astype(str))
    reference_types = tuple(reference["cell_type"].astype(str))
    types_equal = rebuilt_types == reference_types

    rebuilt_counts = rebuilt["n_cells"].to_numpy(dtype=np.int64)
    reference_counts = reference["n_cells"].to_numpy(dtype=np.int64)
    counts_equal = (
        rebuilt_counts.shape == reference_counts.shape
        and np.array_equal(rebuilt_counts, reference_counts)
    )

    if not types_equal:
        return {
            "accepted": False,
            "cell_types_equal": False,
            "cell_counts_equal": counts_equal,
            "max_abs_difference": None,
            "mean_abs_difference": None,
            "rmse": None,
            "tolerance": tolerance,
        }

    rebuilt_array = np.stack(rebuilt["embedding"]).astype(np.float64)
    reference_array = np.stack(reference["embedding"]).astype(np.float64)
    difference = rebuilt_array - reference_array

    max_abs = float(np.max(np.abs(difference)))
    mean_abs = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))

    return {
        "accepted": bool(counts_equal and max_abs <= tolerance),
        "cell_types_equal": True,
        "cell_counts_equal": counts_equal,
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "rmse": rmse,
        "tolerance": tolerance,
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reference_path = args.reference_centroids.expanduser().resolve()

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing encoder output: {output_path}"
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing training AnnData: {input_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing reference centroids: {reference_path}"
        )
    if args.devices < 1:
        raise ValueError("--devices must be at least one")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training requested, but torch.cuda.is_available() is false"
        )

    print("[1/6] Loading paired-cell training AnnData", flush=True)
    adata_train = ad.read_h5ad(input_path)

    required_obs = {"cell_id", "cell_type", "library_id"}
    missing_obs = required_obs.difference(adata_train.obs.columns)
    if missing_obs:
        raise ValueError(f"Missing obs columns: {sorted(missing_obs)}")
    if "feature_types" not in adata_train.var:
        raise ValueError("AnnData var is missing feature_types")

    gene_mask = adata_train.var["feature_types"].eq("Gene Expression")
    adata_rna = adata_train[:, gene_mask].copy()
    adata_rna.obs["library_id"] = (
        adata_rna.obs["library_id"]
        .astype("category")
        .cat.remove_unused_categories()
    )

    observed = {
        "cells": int(adata_rna.n_obs),
        "genes": int(adata_rna.n_vars),
        "libraries": int(adata_rna.obs["library_id"].nunique()),
        "cell_types": int(adata_rna.obs["cell_type"].nunique()),
    }
    expected = {
        "cells": EXPECTED_CELLS,
        "genes": EXPECTED_GENES,
        "libraries": EXPECTED_LIBRARIES,
        "cell_types": EXPECTED_CELL_TYPES,
    }
    if observed != expected:
        raise ValueError(
            f"Unexpected RNA training contract: {observed}; "
            f"expected {expected}"
        )

    print(
        "[2/6] Registering RNA counts with library_id batch correction",
        flush=True,
    )
    scvi.settings.seed = SEED
    scvi.model.SCVI.setup_anndata(
        adata_rna,
        batch_key="library_id",
    )

    model = scvi.model.SCVI(
        adata_rna,
        n_latent=LATENT_DIM,
    )
    model.view_anndata_setup()

    print(
        f"[3/6] Training SCVI on {args.accelerator} "
        f"with seed {SEED}",
        flush=True,
    )
    model.train(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
    )

    print("[4/6] Extracting posterior-mean embeddings", flush=True)
    latent = model.get_latent_representation(
        give_mean=True,
    ).astype(np.float32)

    if latent.shape != (EXPECTED_CELLS, LATENT_DIM):
        raise RuntimeError(f"Unexpected latent shape: {latent.shape}")
    if not np.isfinite(latent).all():
        raise RuntimeError("Latent embedding contains non-finite values")

    centroids = build_centroids(latent, adata_rna.obs["cell_type"])
    reference = pd.read_parquet(reference_path)
    comparison = compare_centroids(
        centroids,
        reference,
        args.centroid_tolerance,
    )

    print("[5/6] Saving encoder and frozen metadata", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path, overwrite=False)

    embedding = pd.DataFrame(
        {
            "cell_id": adata_rna.obs["cell_id"].astype(str).to_numpy(),
            "obs_names": adata_rna.obs_names.astype(str),
            "embedding": list(latent),
        }
    )
    embedding.to_parquet(
        output_path / "rna_scvi_14_embedding.parquet",
        index=False,
    )
    centroids.to_parquet(
        output_path / "rna_scvi_14_cell_type_centroids.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "gene_index": np.arange(adata_rna.n_vars, dtype=np.int32),
            "gene": adata_rna.var_names.astype(str),
        }
    ).to_parquet(
        output_path / "encoder_genes.parquet",
        index=False,
    )

    manifest = {
        "schema_version": 1,
        "input_resource": "rna_scvi_training_anndata",
        "shape": [int(adata_rna.n_obs), int(adata_rna.n_vars)],
        "latent_dim": LATENT_DIM,
        "seed": SEED,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "versions": {
            "python": platform.python_version(),
            "scvi-tools": version("scvi-tools"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "anndata": version("anndata"),
            "lightning": version("lightning"),
        },
        "centroid_comparison": comparison,
    }
    (output_path / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("[6/6] Centroid comparison", flush=True)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)

    if not comparison["accepted"]:
        print(
            "Encoder was saved for diagnosis, but its latent coordinate "
            "system does not reproduce the phase-1 reference centroids.",
            flush=True,
        )
        return 2

    print(f"RNA encoder accepted: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
