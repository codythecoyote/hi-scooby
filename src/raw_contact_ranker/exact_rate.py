from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import zarr

from .common import (
    atomic_json,
    resolution_contract,
    selected_zarr_row,
    sha256_file,
)
from .data import PairData
from .model import RawContactRanker


def chromosome_fold(chromosome: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("At least two chromosome folds are required")
    text = str(chromosome)
    if not text.startswith("chr") or not text[3:].isdigit():
        raise ValueError(f"Expected an autosomal chromosome label: {text!r}")
    return (int(text[3:]) - 1) % folds


def exact_group_codes(tile_row: np.ndarray, distance_bin: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_row, np.int64)
    distance = np.asarray(distance_bin, np.int64)
    if tile.shape != distance.shape:
        raise ValueError("Tile and distance arrays must align")
    width = int(distance.max(initial=0)) + 1
    return tile * width + distance


def center_by_group(
    score: torch.Tensor,
    group_index: torch.Tensor,
) -> torch.Tensor:
    if score.ndim != 1 or group_index.shape != score.shape:
        raise ValueError("Scores and group indices must be aligned vectors")
    output = torch.empty_like(score)
    for group in torch.unique(group_index):
        selected = group_index == group
        output[selected] = score[selected] - score[selected].mean()
    return output


def exact_conditional_log_likelihood(
    log_exposure: torch.Tensor,
    residual_score: torch.Tensor,
    counts: torch.Tensor,
    group_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total log likelihood, event count, and centered residual."""
    if not (
        log_exposure.shape
        == residual_score.shape
        == counts.shape
        == group_index.shape
    ):
        raise ValueError("Exact likelihood tensors must have identical shapes")
    if torch.any(counts < 0) or not bool(torch.isfinite(log_exposure).all()):
        raise ValueError("Exact likelihood received invalid counts or exposure")
    centered = center_by_group(residual_score, group_index)
    logits = log_exposure + centered
    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    events = counts.sum()
    for group in torch.unique(group_index):
        selected = group_index == group
        group_counts = counts[selected]
        group_events = group_counts.sum()
        if bool((group_events > 0).item()):
            total = total + torch.sum(
                group_counts * torch.log_softmax(logits[selected], dim=0)
            )
    return total, events, centered


@dataclass(frozen=True)
class ExactBatch:
    pair_ids: np.ndarray
    counts: np.ndarray
    group_index: np.ndarray


class ExactGroupDataset:
    def __init__(
        self,
        pairs: pd.DataFrame,
        counts: np.ndarray,
        *,
        split: str,
        fold: int | None = None,
        folds: int = 5,
        train_side: bool = True,
    ) -> None:
        if counts.shape != (len(pairs),):
            raise ValueError("Pooled counts do not align with canonical pairs")
        selected = pairs["split"].eq(split).to_numpy()
        if fold is not None:
            chromosomes = pairs["chrom"].astype(str)
            fold_lookup = {
                value: chromosome_fold(value, folds)
                for value in chromosomes.unique()
            }
            fold_values = chromosomes.map(fold_lookup).to_numpy(np.int16)
            selected &= fold_values != fold if train_side else fold_values == fold
        ids = pairs.loc[selected, "pair_id"].to_numpy(np.int64)
        codes = exact_group_codes(
            pairs.loc[selected, "tile_row"].to_numpy(),
            pairs.loc[selected, "distance_bin"].to_numpy(),
        )
        order = np.argsort(codes, kind="stable")
        ordered_ids = ids[order]
        ordered_codes = codes[order]
        boundaries = np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1
        self.groups = [
            group for group in np.split(ordered_ids, boundaries) if len(group)
        ]
        self.maximum_group_size = max(map(len, self.groups), default=0)
        self.counts = np.asarray(counts)

    def iter_batches(
        self,
        maximum_pairs: int,
        *,
        shuffle_seed: int | None,
    ) -> Iterator[ExactBatch]:
        if maximum_pairs < 1:
            raise ValueError("maximum_pairs must be positive")
        order = np.arange(len(self.groups))
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        batch_groups: list[np.ndarray] = []
        pair_total = 0
        for group_position in order:
            group = self.groups[int(group_position)]
            if batch_groups and pair_total + len(group) > maximum_pairs:
                yield self._pack(batch_groups)
                batch_groups = []
                pair_total = 0
            batch_groups.append(group)
            pair_total += len(group)
        if batch_groups:
            yield self._pack(batch_groups)

    def _pack(self, groups: list[np.ndarray]) -> ExactBatch:
        ids = np.concatenate(groups)
        group_index = np.repeat(
            np.arange(len(groups), dtype=np.int64),
            [len(group) for group in groups],
        )
        return ExactBatch(
            pair_ids=ids,
            counts=self.counts[ids].astype(np.float32),
            group_index=group_index,
        )


def pooled_counts(
    data_root: Path,
    pair_ids: np.ndarray | None = None,
    *,
    progress_desc: str | None = None,
) -> np.ndarray:
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    pair_count = int(evidence["full_count"].shape[1])
    ids = (
        np.arange(pair_count, dtype=np.int64)
        if pair_ids is None
        else np.asarray(pair_ids, np.int64)
    )
    output = np.zeros(pair_count, np.uint64)
    contexts = range(evidence["full_count"].shape[0])
    for context in tqdm(
        contexts,
        total=len(contexts),
        desc=progress_desc,
        unit="context",
        disable=progress_desc is None,
    ):
        output += selected_zarr_row(
            evidence["full_count"],
            context,
            ids,
            pair_count=pair_count,
            dtype=np.uint64,
        )
    return output


def build_data_contract(
    config: dict[str, Any],
    *,
    pair_count: int,
    objective: str,
    feature_set: str,
) -> dict[str, Any]:
    data_root = Path(config["outputs"]["data_root"])
    preparation = data_root / "preparation_contract.json"
    feature_manifest = data_root / "feature_manifest.json"
    if not preparation.is_file() or not feature_manifest.is_file():
        raise FileNotFoundError(
            "Exact-rate checkpoints require preparation and feature manifests"
        )
    return {
        "resolution": resolution_contract(config),
        "pair_count": int(pair_count),
        "objective": str(objective),
        "feature_set": str(feature_set),
        "preparation_contract_sha256": sha256_file(preparation),
        "feature_manifest_sha256": sha256_file(feature_manifest),
    }


def validate_checkpoint_contract(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if int(checkpoint.get("schema_version", 0)) < 2:
        raise RuntimeError("Legacy checkpoint lacks a resolution data contract")
    actual = checkpoint.get("data_contract")
    if actual != expected:
        raise RuntimeError(
            "Checkpoint data contract does not match the prepared dataset: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _model_score(
    model: RawContactRanker,
    pair_data: PairData,
    pair_ids: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = pair_data.model_inputs(pair_ids, device)
    output = model(**inputs)
    learned = output["residual_score"] + output["technical_offset"]
    return learned, inputs["fixed_exposure"]


def evaluate_exact_dataset(
    model: RawContactRanker,
    pair_data: PairData,
    dataset: ExactGroupDataset,
    *,
    device: torch.device,
    maximum_pairs: int,
    collect_predictions: bool = False,
    progress_desc: str = "Evaluate exact-rate batches",
) -> dict[str, Any]:
    model.eval()
    likelihood = 0.0
    baseline = 0.0
    events = 0.0
    prediction_rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in tqdm(
            dataset.iter_batches(maximum_pairs, shuffle_seed=None),
            desc=progress_desc,
            unit="batch",
        ):
            learned, log_exposure = _model_score(
                model, pair_data, batch.pair_ids, device
            )
            counts = torch.as_tensor(batch.counts, device=device)
            groups = torch.as_tensor(batch.group_index, device=device)
            value, event_count, centered = exact_conditional_log_likelihood(
                log_exposure, learned, counts, groups
            )
            fixed, _, _ = exact_conditional_log_likelihood(
                log_exposure, torch.zeros_like(learned), counts, groups
            )
            likelihood += float(value.item())
            baseline += float(fixed.item())
            events += float(event_count.item())
            if collect_predictions:
                logits = log_exposure + centered
                probability = torch.empty_like(logits)
                for group in torch.unique(groups):
                    selected = groups == group
                    probability[selected] = torch.softmax(logits[selected], dim=0)
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "pair_id": batch.pair_ids,
                            "shared_residual_score": centered.cpu().numpy(),
                            "probability_within_owner_tile_exact_distance": (
                                probability.cpu().numpy()
                            ),
                        }
                    )
                )
    return {
        "conditional_log_likelihood": likelihood,
        "baseline_conditional_log_likelihood": baseline,
        "events": events,
        "conditional_log_likelihood_per_event": (
            likelihood / events if events else None
        ),
        "baseline_conditional_log_likelihood_per_event": (
            baseline / events if events else None
        ),
        "gain_per_event": (
            (likelihood - baseline) / events if events else None
        ),
        "predictions": (
            pd.concat(prediction_rows, ignore_index=True)
            if prediction_rows
            else None
        ),
    }


def train_exact_rate(
    config: dict[str, Any],
    *,
    output_dir: Path,
    feature_set: str,
    seed: int,
    epochs: int,
    fold: int | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    if feature_set not in {"annotations", "alphagenome", "combined"}:
        raise ValueError("Unsupported exact-rate feature set")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    data_root = Path(config["outputs"]["data_root"])
    pair_data = PairData(
        data_root / "canonical_pairs.parquet",
        data_root / "pair_features.zarr",
        preload_features=bool(config["model"].get("preload_features", True)),
    )
    training_ids = pair_data.pairs.loc[
        pair_data.pairs["split"].eq("train"), "pair_id"
    ].to_numpy(np.int64)
    counts = pooled_counts(
        data_root,
        training_ids,
        progress_desc="Pool training counts",
    )
    folds = int(config["model"].get("chromosome_folds", 5))
    training = ExactGroupDataset(
        pair_data.pairs,
        counts,
        split="train",
        fold=fold,
        folds=folds,
        train_side=True,
    )
    target_bins = 1_000_000 // int(config["bin_size_bp"])
    minimum_distance_bin = int(config["minimum_distance_bp"]) // int(
        config["bin_size_bp"]
    )
    expected_maximum_group = target_bins - minimum_distance_bin
    if training.maximum_group_size > expected_maximum_group:
        raise RuntimeError(
            "Exact owner-tile distance group exceeds the resolution geometry: "
            f"{training.maximum_group_size} > {expected_maximum_group}"
        )
    tuning = (
        ExactGroupDataset(
            pair_data.pairs,
            counts,
            split="train",
            fold=fold,
            folds=folds,
            train_side=False,
        )
        if fold is not None
        else None
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawContactRanker(
        hidden_dim=int(config["model"]["hidden_dim"]),
        context_dim=int(config["model"]["context_dim"]),
        programs=int(config["model"].get("context_programs", 8)),
        dropout=float(config["model"]["dropout"]),
        feature_set=feature_set,
    ).to(device)
    contract = build_data_contract(
        config,
        pair_count=len(pair_data.pairs),
        objective="exact_local_rate",
        feature_set=feature_set,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_loss = math.inf
    best_epoch: int | None = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(checkpoint, contract)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        metrics_path = resume_from.parent / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError("Resume checkpoint lacks metrics.json")
        with metrics_path.open() as handle:
            prior = json.load(handle)
        history = list(prior["history"])
        best_loss = float(prior["best_validation_loss"])
        best_epoch = int(prior["best_epoch"])
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_pairs = int(config["model"]["batch_size"])
    patience = int(config["model"]["early_stopping_patience"])
    stale_epochs = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        total_ll = 0.0
        total_events = 0.0
        batches = training.iter_batches(
            maximum_pairs,
            shuffle_seed=seed + epoch,
        )
        for batch in tqdm(batches, desc=f"Exact-rate epoch {epoch}", unit="batch"):
            learned, log_exposure = _model_score(
                model, pair_data, batch.pair_ids, device
            )
            counts_tensor = torch.as_tensor(batch.counts, device=device)
            groups = torch.as_tensor(batch.group_index, device=device)
            likelihood, events, _ = exact_conditional_log_likelihood(
                log_exposure, learned, counts_tensor, groups
            )
            if not bool((events > 0).item()):
                continue
            loss = -likelihood / events
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_ll += float(likelihood.detach().item())
            total_events += float(events.item())
        train_loss = -total_ll / total_events if total_events else math.inf
        tuning_metrics = (
            evaluate_exact_dataset(
                model,
                pair_data,
                tuning,
                device=device,
                maximum_pairs=maximum_pairs,
                progress_desc=f"Evaluate tuning epoch {epoch}",
            )
            if tuning is not None
            else None
        )
        validation_loss = (
            -float(tuning_metrics["conditional_log_likelihood_per_event"])
            if tuning_metrics is not None
            and tuning_metrics["conditional_log_likelihood_per_event"] is not None
            else train_loss
        )
        row = {
            "epoch": epoch,
            "train_negative_log_likelihood_per_event": train_loss,
            "validation_negative_log_likelihood_per_event": validation_loss,
            "validation_gain_per_event": (
                tuning_metrics["gain_per_event"] if tuning_metrics else None
            ),
        }
        history.append(row)
        checkpoint = {
            "schema_version": 2,
            "stage": "shared_rate",
            "epoch": epoch,
            "model_config": model.config_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "data_contract": contract,
            "seed": seed,
            "fold": fold,
        }
        temporary = output_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output_dir / "last.pt")
        minimum_delta = float(config["model"]["early_stopping_min_delta"])
        if validation_loss < best_loss - minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            temporary = output_dir / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output_dir / "best.pt")
        else:
            stale_epochs += 1
        report = {
            "schema_version": 1,
            "objective": "exact_local_rate",
            "feature_set": feature_set,
            "fold": fold,
            "seed": seed,
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "data_contract": contract,
        }
        atomic_json(output_dir / "metrics.json", report)
        if tuning is not None and stale_epochs >= patience:
            break
    return report


def evaluate_exact_checkpoint(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    split: str,
    output: Path,
    freeze_test: bool = False,
    frozen_release: Path | None = None,
    test_lock: Path | None = None,
) -> dict[str, Any]:
    if split == "test" and not freeze_test:
        raise ValueError("Test evaluation requires freeze_test=True")
    data_root = Path(config["outputs"]["data_root"])
    pair_data = PairData(
        data_root / "canonical_pairs.parquet",
        data_root / "pair_features.zarr",
        preload_features=bool(config["model"].get("preload_features", True)),
    )
    frozen_hash = None
    if split == "test":
        if frozen_release is None or test_lock is None:
            raise ValueError(
                "Test evaluation requires a frozen release and test lock"
            )
        from .release import verify_test_authorization

        verify_test_authorization(
            frozen_release,
            test_lock,
            checkpoint_sha256=sha256_file(checkpoint_path),
            allow_context_checkpoint=False,
        )
        frozen_hash = sha256_file(frozen_release)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_set = str(checkpoint["data_contract"]["feature_set"])
    expected = build_data_contract(
        config,
        pair_count=len(pair_data.pairs),
        objective="exact_local_rate",
        feature_set=feature_set,
    )
    validate_checkpoint_contract(checkpoint, expected)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawContactRanker(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    selected_ids = pair_data.pairs.loc[
        pair_data.pairs["split"].eq(split), "pair_id"
    ].to_numpy(np.int64)
    dataset = ExactGroupDataset(
        pair_data.pairs,
        pooled_counts(
            data_root,
            selected_ids,
            progress_desc=f"Pool {split} counts",
        ),
        split=split,
    )
    metrics = evaluate_exact_dataset(
        model,
        pair_data,
        dataset,
        device=device,
        maximum_pairs=int(config["model"]["batch_size"]),
        collect_predictions=True,
        progress_desc=f"Evaluate {split} exact-rate batches",
    )
    predictions = metrics.pop("predictions")
    assert isinstance(predictions, pd.DataFrame)
    prediction_path = output.with_suffix(".predictions.parquet")
    predictions.sort_values("pair_id", kind="stable").to_parquet(
        prediction_path, index=False, compression="zstd"
    )
    report = {
        "schema_version": 1,
        "split": split,
        "test_accessed": split == "test",
        "frozen_release_sha256": frozen_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_contract": expected,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        **metrics,
    }
    atomic_json(output, report)
    return report
