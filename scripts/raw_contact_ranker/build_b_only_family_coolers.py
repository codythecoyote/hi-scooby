#!/usr/bin/env python3
"""Build raw 10 kb family coolers from pseudoreplicate-B cells only.

The whole-cell split is taken verbatim from the frozen split-0 assignment
table. Raw pairs are streamed once and only B-half contacts from the two
supported families are retained. The resulting cooler candidate mass is
audited against the independently materialized counts_b Zarr arrays.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import cooler
import numpy as np
import pandas as pd
import zarr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_contact_ranker.common import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    source_record,
)
from raw_contact_ranker.evidence import (  # noqa: E402
    _cooler_candidate_event_total,
    _load_pair_arrays,
)


FAMILY_MEMBERS = {
    "cortical_IT": {
        "L23IT-1",
        "L23IT-2",
        "L23IT-3",
        "L45IT",
        "L5IT",
        "L6IT",
    },
    "corticofugal": {"L5ET", "L6CT", "PT"},
}
CELL_TO_FAMILY = {
    member: family
    for family, members in FAMILY_MEMBERS.items()
    for member in members
}
LIBRARY_PATTERN = re.compile(r"_DNA_(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/raw_contact_ranker_10kb.yaml",
    )
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--half", choices=("A", "B"), default="B")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "data/processed/raw_contact_ranker_10kb_v1"
            / "b_only_family_peakachu_10kb_v1"
        ),
    )
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def assignment_lookup(
    assignments: pd.DataFrame,
    *,
    split: int,
) -> tuple[
    dict[str, dict[str, tuple[str, str, str]]],
    dict[str, dict[str, int]],
]:
    selected = assignments.loc[assignments["split"].eq(split)].copy()
    if selected.empty:
        raise RuntimeError(f"No assignments for split {split}")
    lookup: dict[str, dict[str, tuple[str, str, str]]] = defaultdict(dict)
    cells: dict[str, dict[str, int]] = {
        family: {"A": 0, "B": 0} for family in FAMILY_MEMBERS
    }
    for row in selected.itertuples(index=False):
        cell_type = str(row.cell_type)
        family = CELL_TO_FAMILY.get(cell_type)
        if family is None:
            continue
        library = str(row.library_id)
        barcode = str(row.dna_barcode)
        if barcode in lookup[library]:
            raise RuntimeError(
                f"Duplicate barcode {barcode} in library {library}"
            )
        half = str(row.half)
        lookup[library][barcode] = (family, half, cell_type)
        cells[family][half] += 1
    return dict(lookup), cells


def stream_half_pairs(
    pairs_root: Path,
    lookup: dict[str, dict[str, tuple[str, str, str]]],
    *,
    half: str,
    output_dir: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    outputs = {
        family: output_dir / f"{family}.half_{half}.pairs4.tsv.gz"
        for family in FAMILY_MEMBERS
    }
    handles = {
        family: gzip.open(path, "wb", compresslevel=5)
        for family, path in outputs.items()
    }
    stats: dict[str, Any] = {
        "raw_data_rows": 0,
        "malformed_rows": 0,
        "unmatched_library_or_barcode_rows": 0,
        "family_rows_by_half": {
            family: {"A": 0, "B": 0} for family in FAMILY_MEMBERS
        },
        "written_rows": {family: 0 for family in FAMILY_MEMBERS},
        "raw_cell_type_mismatches": 0,
        "raw_cell_type_mismatch_examples": [],
        "input_files": [],
    }
    try:
        paths = sorted(pairs_root.glob("*.cis_uu_autosomes.pairs.gz"))
        if not paths:
            raise FileNotFoundError(
                f"No filtered pairs found beneath {pairs_root}"
            )
        for path_index, path in enumerate(paths, start=1):
            match = LIBRARY_PATTERN.search(path.name)
            if match is None:
                continue
            library = f"DNA{int(match.group(1)):02d}"
            barcode_lookup = lookup.get(library)
            if barcode_lookup is None:
                continue
            file_rows = 0
            file_selected = Counter()
            print(
                f"[b-only-cooler] {path_index:02d}/{len(paths):02d} "
                f"{path.name}",
                flush=True,
            )
            with gzip.open(path, "rb") as source:
                for line in source:
                    if line.startswith(b"#"):
                        continue
                    stats["raw_data_rows"] += 1
                    file_rows += 1
                    fields = line.rstrip().split(b"\t")
                    if len(fields) < 9:
                        stats["malformed_rows"] += 1
                        continue
                    barcode = fields[0].split(b":", 1)[0].decode()
                    assignment = barcode_lookup.get(barcode)
                    if assignment is None:
                        stats["unmatched_library_or_barcode_rows"] += 1
                        continue
                    family, assigned_half, cell_type = assignment
                    observed_cell_type = fields[8].decode()
                    if observed_cell_type != cell_type:
                        stats["raw_cell_type_mismatches"] += 1
                        if (
                            len(stats["raw_cell_type_mismatch_examples"])
                            < 20
                        ):
                            stats["raw_cell_type_mismatch_examples"].append(
                                {
                                    "library_id": library,
                                    "dna_barcode": barcode,
                                    "assignment_cell_type": cell_type,
                                    "raw_pair_cell_type": observed_cell_type,
                                }
                            )
                    stats["family_rows_by_half"][family][assigned_half] += 1
                    if assigned_half != half:
                        continue
                    record = b"\t".join(
                        (fields[1], fields[2], fields[3], fields[4])
                    ) + b"\n"
                    handles[family].write(record)
                    stats["written_rows"][family] += 1
                    file_selected[family] += 1
            stats["input_files"].append(
                {
                    "path": str(path.resolve()),
                    "raw_rows": int(file_rows),
                    "written_rows": dict(file_selected),
                }
            )
    finally:
        for handle in handles.values():
            handle.close()
    if stats["malformed_rows"]:
        raise RuntimeError("Malformed raw pair rows were encountered")
    if not all(stats["written_rows"].values()):
        raise RuntimeError("At least one family has no selected contacts")
    return outputs, stats


def write_autosome_chromsizes(
    source_cooler: Path,
    output: Path,
) -> pd.Series:
    contact_map = cooler.Cooler(str(source_cooler))
    chromsizes = contact_map.chromsizes
    autosomes = [
        f"chr{index}" for index in range(1, 20)
        if f"chr{index}" in chromsizes.index
    ]
    selected = chromsizes.loc[autosomes].astype(np.int64)
    selected.to_csv(output, sep="\t", header=False)
    return selected


def create_coolers(
    pair_paths: dict[str, Path],
    *,
    chromsizes: Path,
    output_dir: Path,
    bin_size: int,
    chunksize: int,
) -> dict[str, Path]:
    executable = Path(sys.executable).parent / "cooler"
    if not executable.is_file():
        located = shutil.which("cooler")
        if located is None:
            raise FileNotFoundError("cooler executable not found")
        executable = Path(located)
    outputs = {}
    for family, pair_path in pair_paths.items():
        output = output_dir / f"{family}.half_B.10kb.cool"
        command = [
            str(executable),
            "cload",
            "pairs",
            f"{chromsizes}:{bin_size}",
            str(pair_path),
            str(output),
            "--zero-based",
            "--assembly",
            "mm10",
            "-c1",
            "1",
            "-p1",
            "2",
            "-c2",
            "3",
            "-p2",
            "4",
            "--input-copy-status",
            "unique",
            "--chunksize",
            str(chunksize),
            "--temp-dir",
            "/tmp",
        ]
        print(
            f"[b-only-cooler] Creating {output.name}", flush=True
        )
        subprocess.run(command, check=True)
        outputs[family] = output
    return outputs


def audit_candidate_mass(
    config: dict[str, Any],
    coolers: dict[str, Path],
) -> dict[str, dict[str, int | bool]]:
    source_cooler = Path(config["paths"]["cooler"])
    canonical = (
        Path(config["outputs"]["data_root"]) / "canonical_pairs.parquet"
    )
    pair_arrays = _load_pair_arrays(canonical, source_cooler, config)
    sorted_keys = pair_arrays["key"]
    n_bins = int(pair_arrays["n_genome_bins"][0])
    evidence = zarr.open_group(
        str(
            Path(config["outputs"]["data_root"])
            / "pseudoreplicate_evidence.zarr"
        ),
        mode="r",
    )
    context_ids = list(map(str, evidence.attrs["context_ids"]))
    output = {}
    for family, members in FAMILY_MEMBERS.items():
        indices = [context_ids.index(member) for member in sorted(members)]
        expected = sum(
            int(
                np.asarray(
                    evidence["counts_b"][index], dtype=np.uint64
                ).sum()
            )
            for index in indices
        )
        observed = _cooler_candidate_event_total(
            coolers[family], sorted_keys, n_bins
        )
        output[family] = {
            "expected_counts_b_candidate_events": expected,
            "observed_cooler_candidate_events": observed,
            "exact_match": bool(expected == observed),
        }
        if expected != observed:
            raise RuntimeError(
                f"{family} cooler candidate mass {observed} != "
                f"counts_b {expected}"
            )
    return output


def main() -> None:
    args = parse_args()
    if args.split != 0 or args.half != "B":
        raise ValueError(
            "This confirmatory build is fixed to split 0, half B"
        )
    config = load_config(args.config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["outputs"]["data_root"])
    assignments_path = data_root / "pseudoreplicate_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    lookup, cell_counts = assignment_lookup(
        assignments, split=args.split
    )
    pair_paths, stream_stats = stream_half_pairs(
        Path(config["paths"]["filtered_pairs"]),
        lookup,
        half=args.half,
        output_dir=output_dir,
    )
    chromsizes_path = output_dir / "mm10.autosomes.chrom.sizes"
    chromsizes = write_autosome_chromsizes(
        Path(config["paths"]["cooler"]), chromsizes_path
    )
    cooler_paths = create_coolers(
        pair_paths,
        chromsizes=chromsizes_path,
        output_dir=output_dir,
        bin_size=int(config["bin_size_bp"]),
        chunksize=args.chunksize,
    )
    cooler_stats = {}
    for family, path in cooler_paths.items():
        contact_map = cooler.Cooler(str(path))
        total = int(
            contact_map.pixels()[:]["count"].to_numpy(np.uint64).sum()
        )
        expected = int(stream_stats["written_rows"][family])
        cooler_stats[family] = {
            "source_rows": expected,
            "cooler_total_count": total,
            "count_conserved": bool(total == expected),
            "nnz": int(contact_map.info["nnz"]),
            "nbins": int(contact_map.info["nbins"]),
            "sha256": sha256_file(path),
        }
        if total != expected:
            raise RuntimeError(
                f"{family} cooler total {total} != source rows {expected}"
            )
    candidate_audit = audit_candidate_mass(config, cooler_paths)
    report = {
        "schema_version": 1,
        "purpose": (
            "Raw 10 kb family contact maps constructed exclusively from "
            "frozen whole-cell pseudoreplicate B for Peakachu calling."
        ),
        "split": int(args.split),
        "half": args.half,
        "family_members": {
            family: sorted(members)
            for family, members in FAMILY_MEMBERS.items()
        },
        "family_cell_counts": cell_counts,
        "chromsizes": {
            "path": str(chromsizes_path),
            "chromosomes": chromsizes.to_dict(),
        },
        "source": {
            "assignments": source_record(assignments_path),
            "filtered_pairs_root": str(
                Path(config["paths"]["filtered_pairs"]).resolve()
            ),
        },
        "stream": stream_stats,
        "outputs": {
            family: {
                "pairs": source_record(pair_paths[family]),
                "cooler": source_record(cooler_paths[family]),
                **cooler_stats[family],
            }
            for family in FAMILY_MEMBERS
        },
        "candidate_mass_audit": candidate_audit,
        "prepared_test_predictions_accessed": False,
        "raw_b_contacts_are_genomewide": True,
    }
    atomic_json(output_dir / "build_report.json", report)
    print(
        f"[b-only-cooler] Wrote {output_dir / 'build_report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
