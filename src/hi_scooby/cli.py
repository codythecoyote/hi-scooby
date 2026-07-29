"""Command-line interface for Hi-Scooby."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from hi_scooby import __version__


def _default_output_path(rna_counts: Path, mode: str) -> Path:
    name = rna_counts.name
    for suffix in (".gz", ".tsv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return Path.cwd() / "hi_scooby_output" / f"{name}.{mode}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hi-scooby",
        description=(
            "Generate smoothed 5 kb and/or sparse 10 kb chromatin "
            "contact-map predictions from a wide RNA-count TSV."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    commands = parser.add_subparsers(dest="command")

    predict = commands.add_parser(
        "predict",
        help="Run contact-map inference.",
    )
    predict.add_argument(
        "rna_counts",
        type=Path,
        help=(
            "Gzipped wide TSV whose first columns are barcode and cell_type."
        ),
    )
    predict.add_argument(
        "--mode",
        choices=("smooth", "sparse", "both"),
        default="both",
        help="Prediction mode; defaults to both.",
    )
    predict.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    predict.add_argument(
        "--contact-depth",
        type=int,
        default=1_000_000,
        help=(
            "Filtered cis-pair depth for sparse expected counts and NB2 "
            "draws; defaults to 1,000,000."
        ),
    )
    predict.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the sparse NB2 simulated count; defaults to 0.",
    )
    predict.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "PyTorch downstream-head device; defaults to automatic "
            "selection. Live AlphaGenome still requires a JAX GPU."
        ),
    )

    return parser


def _development_overrides():
    """Load explicitly marked local-only inputs used by smoke checks."""
    from hi_scooby.inference.runner import cached_embedding_provider

    cache_path = os.environ.get("HI_SCOOBY_DEVELOPMENT_EMBEDDING_CACHE")
    centroid_path = os.environ.get("HI_SCOOBY_DEVELOPMENT_CENTROIDS")
    tile_limit_text = os.environ.get("HI_SCOOBY_DEVELOPMENT_TILE_LIMIT")

    provider = None
    if cache_path:
        provider = cached_embedding_provider(cache_path)

    centroids = None
    if centroid_path:
        import pandas as pd

        resolved = Path(centroid_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                "HI_SCOOBY_DEVELOPMENT_CENTROIDS is not a file: "
                f"{resolved}"
            )
        centroids = pd.read_parquet(resolved)

    tile_limit = None
    if tile_limit_text:
        try:
            tile_limit = int(tile_limit_text)
        except ValueError as error:
            raise ValueError(
                "HI_SCOOBY_DEVELOPMENT_TILE_LIMIT must be an integer"
            ) from error
        if tile_limit <= 0:
            raise ValueError(
                "HI_SCOOBY_DEVELOPMENT_TILE_LIMIT must be positive"
            )

    if provider is not None or centroids is not None or tile_limit is not None:
        print(
            "[development] Local smoke-test overrides are active; this is "
            "not a live full-genome inference run.",
            flush=True,
        )
    return provider, centroids, tile_limit


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "predict":
        if args.contact_depth <= 0:
            parser.error("--contact-depth must be positive")

        from hi_scooby.inference.runner import run_inference

        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else _default_output_path(
                args.rna_counts,
                args.mode,
            ).resolve()
        )
        device = None if args.device == "auto" else args.device

        try:
            provider, centroids, tile_limit = _development_overrides()
            result = run_inference(
                args.rna_counts,
                output,
                mode=args.mode,
                contact_depth=args.contact_depth,
                seed=args.seed,
                device=device,
                tile_limit=tile_limit,
                centroids=centroids,
                embedding_provider=provider,
            )
        except KeyboardInterrupt:
            print("\nhi-scooby: interrupted", file=sys.stderr)
            return 130
        except Exception as error:
            print(
                f"hi-scooby: error: {error}",
                file=sys.stderr,
            )
            return 2

        print(f"Hi-Scooby prediction complete: {result}", flush=True)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
