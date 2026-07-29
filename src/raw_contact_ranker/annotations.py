from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from tqdm.auto import tqdm

from .common import atomic_json, source_record, update_manifest


# JASPAR MA0139.1 CTCF position counts, converted to log-odds at runtime.
CTCF_COUNTS = np.asarray(
    [
        [87,167,281,56],[291,145,49,106],[76,414,42,59],[167,54,329,41],
        [17,5,557,12],[3,4,578,6],[4,565,6,16],[560,2,15,14],
        [6,4,574,7],[13,12,500,66],[106,45,373,67],[85,82,422,2],
        [150,209,66,166],[141,75,117,258],[98,309,70,114],[303,82,101,105],
        [108,173,194,116],[115,143,213,120],[139,115,209,128],
    ],
    dtype=np.float64,
)


def _pwm() -> tuple[np.ndarray, np.ndarray]:
    probabilities = (CTCF_COUNTS + 0.5) / (CTCF_COUNTS.sum(1, keepdims=True) + 2.0)
    forward = np.log2(probabilities / 0.25)
    complement = np.asarray([3, 2, 1, 0])
    reverse = forward[::-1, complement]
    return forward.astype(np.float32), reverse.astype(np.float32)


def _sequence_features(sequence: str) -> tuple[float, float, float, int]:
    encoded = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
    valid = np.isin(encoded, np.frombuffer(b"ACGT", dtype=np.uint8))
    valid_fraction = float(valid.mean()) if len(encoded) else math.nan
    gc = float(np.isin(encoded, np.frombuffer(b"GC", dtype=np.uint8)).sum() / max(valid.sum(), 1))
    if len(encoded) < len(CTCF_COUNTS):
        return valid_fraction, gc, math.nan, 0
    lookup = np.full(256, -1, np.int8)
    for index, base in enumerate(b"ACGT"):
        lookup[base] = index
    bases = lookup[encoded]
    windows = np.lib.stride_tricks.sliding_window_view(bases, len(CTCF_COUNTS))
    valid_windows = np.all(windows >= 0, axis=1)
    if not valid_windows.any():
        return valid_fraction, gc, math.nan, 0
    forward, reverse = _pwm()
    positions = np.arange(len(CTCF_COUNTS))
    fw = forward[positions, windows[valid_windows]].sum(1)
    rv = reverse[positions, windows[valid_windows]].sum(1)
    fw_max, rv_max = float(fw.max()), float(rv.max())
    return valid_fraction, gc, max(fw_max, rv_max), 1 if fw_max >= rv_max else -1


def _interval_bin_counts(frame: pd.DataFrame, bin_size: int) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for row in frame.itertuples(index=False):
        first = int(row.start) // bin_size
        last = max(first, (int(row.end) - 1) // bin_size)
        for index in range(first, last + 1):
            key = (str(row.chrom), index)
            result[key] = result.get(key, 0) + 1
    return result


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def _interval_bin_counts_from_bed(
    path: Path, bin_size: int
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Malformed BED row in {path}: {line.rstrip()}")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if end <= start:
                continue
            first = start // bin_size
            last = (end - 1) // bin_size
            for index in range(first, last + 1):
                key = (chrom, index)
                result[key] = result.get(key, 0) + 1
    return result


def _tss_bin_counts_from_gtf(
    path: Path, bin_size: int
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed GTF row in {path}: {line.rstrip()}")
            chrom, _, feature, start, end, _, strand, _, _ = fields
            if feature != "gene":
                continue
            # GTF coordinates are one-based and inclusive; internal coordinates
            # are zero-based and half-open.
            tss = int(end) - 1 if strand == "-" else int(start) - 1
            key = (chrom, tss // bin_size)
            result[key] = result.get(key, 0) + 1
    return result


def _motif_count(sequence: str, motifs: tuple[str, ...]) -> int:
    sequence = sequence.upper()
    return sum(sequence.count(motif) for motif in motifs)


def _mappability_bins(
    path: Path,
    bins_by_chrom: dict[str, int],
    bin_size: int,
) -> dict[str, np.ndarray]:
    sums = {chrom: np.zeros(count, np.float64) for chrom, count in bins_by_chrom.items()}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 4 or fields[0] not in sums:
                continue
            chrom, start, end, value = fields[0], int(fields[1]), int(fields[2]), float(fields[3])
            first, last = start // bin_size, (end - 1) // bin_size
            array = sums[chrom]
            for index in range(first, min(last + 1, len(array))):
                overlap = max(0, min(end, (index + 1) * bin_size) - max(start, index * bin_size))
                array[index] += overlap * value
    return {chrom: values / bin_size for chrom, values in sums.items()}


def _anchor_class(tss_i, tss_j, ccre_i, ccre_j, ctcf_i, ctcf_j) -> np.ndarray:
    promoter = tss_i | tss_j
    ccre = ccre_i | ccre_j
    convergent = (ctcf_i == 1) & (ctcf_j == -1)
    output = np.full(len(promoter), "other", dtype=object)
    output[(ctcf_i != 0) | (ctcf_j != 0)] = "ctcf"
    output[ccre] = "ccre"
    output[promoter] = "promoter"
    output[convergent] = "ctcf_convergent"
    return output


def annotate_pairs(
    config: dict[str, Any],
    pairs_path: Path,
) -> dict[str, Any]:
    bin_size = int(config["bin_size_bp"])
    used_anchors: set[tuple[str, int]] = set()
    pair_parquet = pq.ParquetFile(pairs_path)
    for batch in tqdm(
        pair_parquet.iter_batches(columns=["chrom", "bin_i", "bin_j"], batch_size=1_000_000),
        total=math.ceil(pair_parquet.metadata.num_rows / 1_000_000),
        desc="Index candidate anchors",
        unit="batch",
    ):
        frame = batch.to_pandas()
        chrom = frame["chrom"].astype(str).to_numpy()
        used_anchors.update(zip(chrom, frame["bin_i"].to_numpy(int), strict=True))
        used_anchors.update(zip(chrom, frame["bin_j"].to_numpy(int), strict=True))
    bins_by_chrom: dict[str, int] = {}
    for chrom, start in used_anchors:
        bins_by_chrom[chrom] = max(
            bins_by_chrom.get(chrom, 0), int(start) // bin_size + 1
        )
    ccre_counts = _interval_bin_counts_from_bed(
        Path(config["paths"]["ccre_registry"]), bin_size
    )
    tss_counts = _tss_bin_counts_from_gtf(
        Path(config["paths"]["gene_annotation"]), bin_size
    )
    mappability = _mappability_bins(
        Path(config["paths"]["mappability"]), bins_by_chrom, bin_size
    )
    fasta = pysam.FastaFile(config["paths"]["fasta"])

    anchor_rows = []
    for chrom, start in tqdm(
        sorted(used_anchors),
        total=len(used_anchors),
        desc="Annotate genomic bins",
        unit="bin",
    ):
        chrom = str(chrom)
        start = int(start)
        index = start // bin_size
        sequence = fasta.fetch(chrom, start, start + bin_size)
        valid, gc, ctcf_score, ctcf_orientation = _sequence_features(sequence)
        assay_site_count = _motif_count(
            sequence,
            (
                str(config["annotations"]["primary_restriction_motif"]),
                str(config["annotations"]["secondary_restriction_motif"]),
            ),
        )
        map_value = (
            float(mappability[chrom][index])
            if chrom in mappability and index < len(mappability[chrom])
            else math.nan
        )
        if not np.isfinite(ctcf_score) or ctcf_score < float(
            config["annotations"]["ctcf_score_threshold"]
        ):
            ctcf_orientation = 0
        anchor_rows.append(
            {
                "chrom": chrom,
                "bin_start": start,
                "mappability": map_value,
                "valid_sequence_fraction": valid,
                "gc_fraction": gc,
                # CviQI in-situ digestion and NlaIII circularization both
                # affect capture. On fixed 5 kb bins, motif count is density up
                # to a constant and avoids conflating cCRE peaks with exposure.
                "assay_site_density": float(assay_site_count),
                "tss": tss_counts.get((chrom, index), 0) > 0,
                "ccre": ccre_counts.get((chrom, index), 0) > 0,
                "ctcf_score": ctcf_score,
                "ctcf_orientation": ctcf_orientation,
            }
        )
    fasta.close()
    anchors = pd.DataFrame(anchor_rows).set_index(["chrom", "bin_start"])

    temporary = pairs_path.with_suffix(".annotated.parquet.tmp")
    writer: pq.ParquetWriter | None = None
    pair_count = 0
    filtered_pair_count = 0
    missing_counts: dict[str, int] = {}
    anchor_class_counts: dict[str, int] = {}
    multi_label_counts = {
        "anchor_has_promoter": 0,
        "anchor_has_ccre": 0,
        "anchor_has_ctcf": 0,
        "anchor_ctcf_convergent": 0,
        "anchor_ctcf_divergent": 0,
    }
    parquet = pq.ParquetFile(pairs_path)
    input_pair_count = parquet.metadata.num_rows
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=500_000),
            total=math.ceil(parquet.metadata.num_rows / 500_000),
            desc="Join pair annotations",
            unit="batch",
        ):
            frame = batch.to_pandas()
            left_index = pd.MultiIndex.from_arrays([frame["chrom"], frame["bin_i"]])
            right_index = pd.MultiIndex.from_arrays([frame["chrom"], frame["bin_j"]])
            left = anchors.reindex(left_index).reset_index(drop=True)
            right = anchors.reindex(right_index).reset_index(drop=True)
            for suffix, values in (("i", left), ("j", right)):
                frame[f"bin_{suffix}_mappability"] = values["mappability"].to_numpy()
                frame[f"bin_{suffix}_valid_sequence_fraction"] = values["valid_sequence_fraction"].to_numpy()
                frame[f"bin_{suffix}_gc_fraction"] = values["gc_fraction"].to_numpy()
                frame[f"bin_{suffix}_assay_site_density"] = values["assay_site_density"].to_numpy()
                frame[f"bin_{suffix}_tss"] = values["tss"].fillna(False).to_numpy(bool)
                frame[f"bin_{suffix}_ccre"] = values["ccre"].fillna(False).to_numpy(bool)
                frame[f"bin_{suffix}_ctcf_score"] = values["ctcf_score"].to_numpy()
                frame[f"bin_{suffix}_ctcf_orientation"] = values["ctcf_orientation"].fillna(0).to_numpy(np.int8)
                for column in ("mappability", "valid_sequence_fraction", "gc_fraction", "ctcf_score"):
                    missing = ~np.isfinite(frame[f"bin_{suffix}_{column}"].to_numpy(float))
                    frame[f"bin_{suffix}_{column}_missing"] = missing
                    missing_counts[f"bin_{suffix}_{column}"] = missing_counts.get(f"bin_{suffix}_{column}", 0) + int(missing.sum())
            map_i = left["mappability"].to_numpy(float)
            map_j = right["mappability"].to_numpy(float)
            valid_i = left["valid_sequence_fraction"].to_numpy(float)
            valid_j = right["valid_sequence_fraction"].to_numpy(float)
            minimum_map = float(config["annotations"]["minimum_mappability"])
            minimum_valid = float(
                config["annotations"]["minimum_valid_sequence_fraction"]
            )
            eligible = (
                np.isfinite(map_i)
                & np.isfinite(map_j)
                & np.isfinite(valid_i)
                & np.isfinite(valid_j)
                & (map_i >= minimum_map)
                & (map_j >= minimum_map)
                & (valid_i >= minimum_valid)
                & (valid_j >= minimum_valid)
            )
            exposure = (
                np.sqrt(np.clip(map_i * map_j, 1e-12, None))
                * np.sqrt(np.clip(valid_i * valid_j, 1e-6, None))
                * (1.0 + left["assay_site_density"].to_numpy(float))
                * (1.0 + right["assay_site_density"].to_numpy(float))
            )
            exposure_floor = float(config["annotations"]["exposure_floor"])
            exposure_ceiling = float(config["annotations"]["exposure_ceiling"])
            clipped = np.clip(exposure, exposure_floor, exposure_ceiling)
            frame["exposure_clipped"] = eligible & (clipped != exposure)
            frame["exposure"] = clipped
            frame["anchor_has_promoter"] = (
                frame["bin_i_tss"].to_numpy(bool) | frame["bin_j_tss"].to_numpy(bool)
            )
            frame["anchor_has_ccre"] = (
                frame["bin_i_ccre"].to_numpy(bool) | frame["bin_j_ccre"].to_numpy(bool)
            )
            frame["anchor_has_ctcf"] = (
                (frame["bin_i_ctcf_orientation"].to_numpy(np.int8) != 0)
                | (frame["bin_j_ctcf_orientation"].to_numpy(np.int8) != 0)
            )
            frame["anchor_ctcf_convergent"] = (
                (frame["bin_i_ctcf_orientation"].to_numpy(np.int8) == 1)
                & (frame["bin_j_ctcf_orientation"].to_numpy(np.int8) == -1)
            )
            frame["anchor_ctcf_divergent"] = (
                (frame["bin_i_ctcf_orientation"].to_numpy(np.int8) == -1)
                & (frame["bin_j_ctcf_orientation"].to_numpy(np.int8) == 1)
            )
            frame["anchor_class"] = _anchor_class(
                frame["bin_i_tss"].to_numpy(bool),
                frame["bin_j_tss"].to_numpy(bool),
                frame["bin_i_ccre"].to_numpy(bool),
                frame["bin_j_ccre"].to_numpy(bool),
                frame["bin_i_ctcf_orientation"].to_numpy(np.int8),
                frame["bin_j_ctcf_orientation"].to_numpy(np.int8),
            )
            filtered_pair_count += int((~eligible).sum())
            frame = frame.loc[eligible].copy()
            values, counts = np.unique(frame["anchor_class"], return_counts=True)
            for value, count in zip(values, counts, strict=True):
                anchor_class_counts[str(value)] = (
                    anchor_class_counts.get(str(value), 0) + int(count)
                )
            for label in multi_label_counts:
                multi_label_counts[label] += int(frame[label].sum())
            frame["pair_id"] = np.arange(
                pair_count, pair_count + len(frame), dtype=np.int64
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd",
                    use_dictionary=["chrom", "distance_band", "tile_id", "split", "anchor_class"],
                )
            writer.write_table(table)
            pair_count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(pairs_path)
    sources = {
        name: source_record(Path(config["paths"][name]))
        for name in ("gene_annotation", "ccre_registry", "fasta", "mappability")
    }
    report = {
        "pair_count": pair_count,
        "annotated_candidate_bins": len(anchors),
        "input_pair_count": input_pair_count,
        "pairs_filtered_invalid_exposure": filtered_pair_count,
        "pair_count_unchanged": pair_count == input_pair_count,
        "missing_annotation_counts": missing_counts,
        "ctcf_pwm_id": config["annotations"]["ctcf_pwm_id"],
        "ctcf_score_threshold": float(config["annotations"]["ctcf_score_threshold"]),
        "gene_annotation_name": config["annotations"]["gene_annotation_name"],
        "ccre_registry_accession": config["annotations"]["ccre_registry_accession"],
        "anchor_class_counts": anchor_class_counts,
        "multi_label_counts": multi_label_counts,
        "mappability_name": config["annotations"]["mappability_name"],
        "sources": sources,
        "contact_counts_consulted": False,
        "cooler_balance_weights_consulted": False,
        "exposure_definition": "sqrt(mappability_i*mappability_j)*sqrt(valid_i*valid_j)*(1+site_i)*(1+site_j)",
        "assay_site_motifs": [
            config["annotations"]["primary_restriction_motif"],
            config["annotations"]["secondary_restriction_motif"],
        ],
    }
    output_root = Path(config["outputs"]["data_root"])
    atomic_json(output_root / "annotation_report.json", report)
    update_manifest(output_root, "annotations", report)
    return report
