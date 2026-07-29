from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from .common import atomic_json
from .data import PairData
from .losses import raw_contact_loss
from .model import RawContactRanker


@dataclass(frozen=True)
class ControlBatch:
    positive_ids: np.ndarray
    event_control_ids: np.ndarray
    rank_control_ids: np.ndarray
    proposal_probabilities: np.ndarray
    context_ids: np.ndarray
    support_weights: np.ndarray
    event_counts: np.ndarray


@dataclass
class ControlDataset:
    positive_ids: np.ndarray
    event_control_ids: np.ndarray
    rank_control_ids: np.ndarray
    proposal_probabilities: np.ndarray
    context_ids: np.ndarray
    support_weights: np.ndarray
    event_counts: np.ndarray

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        *,
        controls_per_event: int,
        rank_controls_per_event: int,
        progress: bool = True,
        load_batch_rows: int = 65_536,
    ) -> "ControlDataset":
        parquet = pq.ParquetFile(path)
        row_count = parquet.metadata.num_rows
        positive_ids = np.empty(row_count, np.int64)
        event_control_ids = np.empty((row_count, controls_per_event), np.int64)
        rank_control_ids = np.empty((row_count, rank_controls_per_event), np.int64)
        proposal_probabilities = np.empty(
            (row_count, controls_per_event), np.float32
        )
        context_ids = np.empty(row_count, np.int64)
        support_weights = np.empty(row_count, np.float32)
        event_counts = np.empty(row_count, np.float32)
        columns = [
            "positive_pair_id",
            "event_control_pair_ids",
            "event_proposal_probabilities",
            "rank_control_pair_ids",
            "context_id",
            "support_weight",
            "event_count",
        ]
        cursor = 0
        bar = tqdm(
            total=row_count,
            desc="Preload sampled controls",
            unit="event",
            unit_scale=True,
            disable=not progress,
        )
        for batch in parquet.iter_batches(
            batch_size=load_batch_rows,
            columns=columns,
        ):
            count = batch.num_rows
            selected = slice(cursor, cursor + count)

            def column(name: str):
                return batch.column(batch.schema.get_field_index(name))

            def scalar(name: str, dtype) -> np.ndarray:
                values = column(name)
                if values.null_count:
                    raise ValueError(f"Control column {name} contains nulls")
                return np.asarray(
                    values.to_numpy(zero_copy_only=False), dtype=dtype
                )

            def fixed_list(name: str, dtype, width: int) -> np.ndarray:
                values = column(name)
                if values.null_count:
                    raise ValueError(f"Control column {name} contains nulls")
                flattened = np.asarray(
                    values.flatten().to_numpy(zero_copy_only=False), dtype=dtype
                )
                expected = count * width
                if len(flattened) != expected:
                    raise ValueError(
                        f"Control column {name} has {len(flattened)} values; "
                        f"expected {expected}"
                    )
                return flattened.reshape(count, width)

            positive_ids[selected] = scalar("positive_pair_id", np.int64)
            event_control_ids[selected] = fixed_list(
                "event_control_pair_ids", np.int64, controls_per_event
            )
            rank_control_ids[selected] = fixed_list(
                "rank_control_pair_ids", np.int64, rank_controls_per_event
            )
            proposal_probabilities[selected] = fixed_list(
                "event_proposal_probabilities", np.float32, controls_per_event
            )
            context_ids[selected] = scalar("context_id", np.int64)
            support_weights[selected] = scalar("support_weight", np.float32)
            event_counts[selected] = scalar("event_count", np.float32)
            cursor += count
            bar.update(count)
        bar.close()
        if cursor != row_count:
            raise RuntimeError(
                f"Loaded {cursor} control rows but Parquet metadata reports {row_count}"
            )
        return cls(
            positive_ids=positive_ids,
            event_control_ids=event_control_ids,
            rank_control_ids=rank_control_ids,
            proposal_probabilities=proposal_probabilities,
            context_ids=context_ids,
            support_weights=support_weights,
            event_counts=event_counts,
        )

    @property
    def nbytes(self) -> int:
        return int(sum(array.nbytes for array in self.__dict__.values()))

    def batch_count(self, batch_size: int) -> int:
        return math.ceil(len(self.positive_ids) / batch_size)

    def iter_epoch(
        self,
        batch_size: int,
        *,
        shuffle_seed: int,
    ) -> Iterator[ControlBatch]:
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(len(self.positive_ids))
        for start in range(0, len(self.positive_ids), batch_size):
            stop = min(start + batch_size, len(self.positive_ids))
            indices = order[start:stop]
            yield ControlBatch(
                positive_ids=self.positive_ids[indices],
                event_control_ids=self.event_control_ids[indices],
                rank_control_ids=self.rank_control_ids[indices],
                proposal_probabilities=self.proposal_probabilities[indices],
                context_ids=self.context_ids[indices],
                support_weights=self.support_weights[indices],
                event_counts=self.event_counts[indices],
            )


def _meaningful_improvement(
    validation_loss: float,
    best_loss: float,
    minimum_delta: float,
) -> bool:
    return validation_loss < best_loss - minimum_delta


def _load_resume_metadata(
    resume_from: Path,
    output_dir: Path,
    checkpoint: dict[str, Any],
    requested_epochs: int,
) -> tuple[list[dict[str, Any]], float, int, int, int]:
    start_epoch = int(checkpoint["epoch"]) + 1
    prior_metrics_path = resume_from.parent / "metrics.json"
    if not prior_metrics_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint lacks prior metrics: {prior_metrics_path}"
        )
    with prior_metrics_path.open() as handle:
        prior_metrics = json.load(handle)
    history = list(prior_metrics.get("history", []))
    best_loss = float(prior_metrics["best_validation_loss"])
    best_epoch = prior_metrics.get("best_epoch")
    if len(history) != start_epoch:
        raise RuntimeError(
            "Resume history does not end at the checkpoint epoch: "
            f"history={len(history)}, start_epoch={start_epoch}"
        )
    if requested_epochs <= start_epoch:
        raise ValueError(
            f"Requested total epochs {requested_epochs} must exceed resumed "
            f"epoch count {start_epoch}"
        )
    if not isinstance(best_epoch, int) or not 0 <= best_epoch < start_epoch:
        raise RuntimeError(
            f"Resume metrics contain an invalid best_epoch: {best_epoch!r}"
        )
    epochs_without_improvement = start_epoch - best_epoch - 1
    prior_best = resume_from.parent / "best.pt"
    if not prior_best.exists():
        raise FileNotFoundError(
            f"Resume run lacks its selected best checkpoint: {prior_best}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prior_best, output_dir / "best.pt")
    return (
        history,
        best_loss,
        best_epoch,
        start_epoch,
        epochs_without_improvement,
    )


def _score(model, pair_data: PairData, ids, device, context_embeddings, contexts):
    inputs = pair_data.model_inputs(ids, device)
    output = model(
        **inputs,
        context_embedding=context_embeddings,
        context_index=contexts if context_embeddings is not None else None,
    )
    rank_score = output["residual_score"] + output["context_delta"]
    return output["log_rate"], rank_score, output


def _batch_objective(
    model,
    pair_data: PairData,
    batch: ControlBatch,
    device: torch.device,
    context_embeddings: torch.Tensor | None,
    config: dict[str, Any],
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context_ids = torch.as_tensor(
        batch.context_ids, dtype=torch.long, device=device
    )
    positive_score, positive_rank_score, _ = _score(
        model,
        pair_data,
        batch.positive_ids,
        device,
        context_embeddings,
        context_ids,
    )
    event_context = context_ids.repeat_interleave(batch.event_control_ids.shape[1])
    control_score, _, _ = _score(
        model,
        pair_data,
        batch.event_control_ids.reshape(-1),
        device,
        context_embeddings,
        event_context,
    )
    control_score = control_score.reshape(batch.event_control_ids.shape)
    rank_context = context_ids.repeat_interleave(batch.rank_control_ids.shape[1])
    _, rank_control_score, _ = _score(
        model,
        pair_data,
        batch.rank_control_ids.reshape(-1),
        device,
        context_embeddings,
        rank_context,
    )
    rank_control_score = rank_control_score.reshape(batch.rank_control_ids.shape)
    regularization = (
        float(config["model"]["rna_l2"]) * model.rna_regularization()
        if stage in {"rna", "full"} else 0.0
    )
    return raw_contact_loss(
        positive_score,
        control_score,
        torch.as_tensor(
            batch.proposal_probabilities, dtype=torch.float32, device=device
        ),
        torch.as_tensor(batch.support_weights, device=device),
        lambda_rank=float(config["model"]["lambda_rank"]),
        rank_positive_score=positive_rank_score,
        rank_control_score=rank_control_score,
        regularization=regularization,
        event_count=torch.as_tensor(batch.event_counts, device=device),
    )


def _evaluate_control_objective(
    model,
    pair_data: PairData,
    controls: ControlDataset,
    *,
    expected_mask: np.ndarray,
    device: torch.device,
    context_embeddings: torch.Tensor | None,
    config: dict[str, Any],
    stage: str,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate one fixed model with global event/rank denominators."""
    model.eval()
    event_numerator = 0.0
    event_denominator = 0.0
    rank_numerator = 0.0
    rank_denominator = 0.0
    with torch.no_grad():
        for batch in controls.iter_epoch(batch_size, shuffle_seed=0):
            for name, ids in (
                ("positive", batch.positive_ids),
                ("event control", batch.event_control_ids),
                ("rank control", batch.rank_control_ids),
            ):
                if not expected_mask[ids].all():
                    raise RuntimeError(
                        f"A non-held-out {name} reached validation selection"
                    )
            _, components = _batch_objective(
                model,
                pair_data,
                batch,
                device,
                context_embeddings,
                config,
                stage,
            )
            event_weight = float(np.sum(batch.event_counts, dtype=np.float64))
            rank_weight = float(np.sum(batch.support_weights, dtype=np.float64))
            event_numerator += float(components["event"].item()) * event_weight
            event_denominator += event_weight
            rank_numerator += float(components["rank"].item()) * rank_weight
            rank_denominator += rank_weight
    if event_denominator <= 0:
        raise RuntimeError("Validation controls contain no positive event mass")
    event = event_numerator / event_denominator
    rank = rank_numerator / rank_denominator if rank_denominator > 0 else 0.0
    regularization = (
        float(config["model"]["rna_l2"])
        * float(model.rna_regularization().detach().item())
        if stage in {"rna", "full"} else 0.0
    )
    total = event + float(config["model"]["lambda_rank"]) * rank + regularization
    return {
        "loss": total,
        "event_loss": event,
        "rank_loss": rank,
        "regularization_loss": regularization,
        "event_denominator": event_denominator,
        "rank_denominator": rank_denominator,
    }


def _overfit_fixture(
    config: dict[str, Any],
    output_dir: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawContactRanker(dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch, controls = 16, 8
    inputs = {
        "pair_embedding": torch.randn(batch, 128, device=device),
        "anchor_i": torch.randn(batch, 128, device=device),
        "anchor_j": torch.randn(batch, 128, device=device),
        "annotations": torch.randn(batch, 16, device=device),
        "pair_annotations": torch.randn(batch, 5, device=device),
        "technical": torch.randn(batch, 5, device=device),
        "fixed_exposure": torch.randn(batch, device=device),
        "fixed_distance_offset": torch.randn(batch, device=device),
        "distance_bp": torch.linspace(250_000, 995_000, batch, device=device),
    }
    control_inputs = {
        key: value.repeat_interleave(controls, 0)
        if key != "pair_embedding"
        else value.repeat_interleave(controls, 0) - 0.5
        for key, value in inputs.items()
    }
    initial = None
    bar = tqdm(range(200), desc=f"Overfit {stage} fixture", unit="step")
    for _ in bar:
        positive_output = model(**inputs)
        control_output = model(**control_inputs)
        positive = positive_output["log_rate"]
        control = control_output["log_rate"].reshape(batch, controls)
        loss, _ = raw_contact_loss(
            positive,
            control,
            torch.full_like(control, 1 / controls),
            torch.ones(batch, device=device),
            lambda_rank=float(config["model"]["lambda_rank"]),
            rank_positive_score=positive_output["residual_score"],
            rank_control_score=control_output["residual_score"].reshape(batch, controls),
        )
        if initial is None:
            initial = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=f"{loss.item():.5f}", refresh=False)
    final = float(loss.item())
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "stage": stage,
        "model_config": model.config_dict(),
        "model_state": model.state_dict(),
        "fixture": True,
    }
    torch.save(checkpoint, output_dir / "fixture.pt")
    report = {"initial_loss": initial, "final_loss": final, "overfit": final < initial * 0.5}
    atomic_json(output_dir / "fixture_metrics.json", report)
    return report


def train(
    config: dict[str, Any],
    *,
    stage: str,
    output_dir: Path,
    initialize_from: Path | None = None,
    resume_from: Path | None = None,
    overfit_fixture: bool = False,
) -> dict[str, Any]:
    if stage not in {"sequence", "rna", "full"}:
        raise ValueError(f"Unsupported training stage: {stage!r}")
    lambda_rank = float(config["model"]["lambda_rank"])
    if not math.isfinite(lambda_rank) or lambda_rank <= 0:
        raise ValueError("model.lambda_rank must be finite and positive")
    configured_minimum_delta = float(
        config["model"].get("early_stopping_min_delta", 0.0)
    )
    if not math.isfinite(configured_minimum_delta) or configured_minimum_delta < 0:
        raise ValueError(
            "model.early_stopping_min_delta must be finite and nonnegative"
        )
    if initialize_from is not None and resume_from is not None:
        raise ValueError("initialize_from and resume_from are mutually exclusive")
    if overfit_fixture:
        return _overfit_fixture(config, output_dir, stage=stage)
    data_root = Path(config["outputs"]["data_root"])
    preload_features = bool(config["model"].get("preload_features", True))
    pair_data = PairData(
        data_root / "canonical_pairs.parquet",
        data_root / "pair_features.zarr",
        preload_features=preload_features,
        progress=True,
    )
    if preload_features:
        tqdm.write(
            f"RAM feature/covariate cache: {pair_data.preloaded_bytes / 2**30:.2f} GiB"
        )
    if stage == "full":
        control_path = data_root / "sampled_local_control_groups.parquet"
        validation_control_path = (
            data_root / "validation_sampled_local_control_groups.parquet"
        )
        missing = [
            path for path in (control_path, validation_control_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Full-model training requires evaluation-aligned local rank "
                "controls. Run resample_local_rank_controls.py first. Missing: "
                + ", ".join(str(path) for path in missing)
            )
        report_path = data_root / "local_rank_resampling_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                "Full-model local controls lack their integrity report: "
                f"{report_path}"
            )
        with report_path.open() as handle:
            local_report = json.load(handle)
        expected_proposal = (
            "same_tile_distance_band_zero_and_lower_replicate_support"
        )
        if local_report.get("rank_proposal") != expected_proposal:
            raise RuntimeError(
                "Full-model local rank controls use an obsolete proposal: "
                f"{local_report.get('rank_proposal')!r}"
            )
        for split, path in (
            ("train", control_path),
            ("validation", validation_control_path),
        ):
            split_report = local_report.get("splits", {}).get(split, {})
            reported_output = Path(split_report.get("output", "")).resolve()
            reported_rows = split_report.get("rows")
            actual_rows = pq.ParquetFile(path).metadata.num_rows
            if reported_output != path.resolve() or reported_rows != actual_rows:
                raise RuntimeError(
                    f"Local rank control integrity report does not match {split} "
                    f"table: {path}"
                )
    else:
        control_path = data_root / "sampled_control_groups.parquet"
        validation_control_path = (
            data_root / "validation_sampled_control_groups.parquet"
        )
    controls = ControlDataset.from_parquet(
        control_path,
        controls_per_event=int(config["sampling"]["controls_per_event"]),
        rank_controls_per_event=int(config["sampling"]["rank_controls_per_event"]),
        progress=True,
    )
    tqdm.write(f"RAM control cache: {controls.nbytes / 2**30:.2f} GiB")
    validation_controls = ControlDataset.from_parquet(
        validation_control_path,
        controls_per_event=int(config["sampling"]["controls_per_event"]),
        rank_controls_per_event=int(config["sampling"]["rank_controls_per_event"]),
        progress=True,
    )
    tqdm.write(
        "RAM validation control cache: "
        f"{validation_controls.nbytes / 2**30:.2f} GiB"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    model = RawContactRanker(
        hidden_dim=int(config["model"]["hidden_dim"]),
        context_dim=int(config["model"]["context_dim"]),
        programs=int(config["model"]["context_programs"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    source_checkpoint = resume_from or initialize_from
    checkpoint = None
    if source_checkpoint:
        checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        if resume_from and checkpoint.get("stage") != stage:
            raise ValueError(
                "Resume checkpoint stage does not match requested stage: "
                f"{checkpoint.get('stage')!r} != {stage!r}"
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
    context_embeddings = None
    if stage in {"rna", "full"}:
        if stage == "rna":
            if initialize_from is None:
                raise ValueError("RNA stage requires a sequence checkpoint")
            model.freeze_sequence()
        contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
            "context_index", kind="stable"
        )
        centroids = pd.read_parquet(config["paths"]["centroids"])
        ordered_centroids = contexts[["cell_type", "context_index"]].merge(
            centroids[["cell_type", "embedding"]],
            on="cell_type", how="left", sort=False, validate="one_to_one",
        ).sort_values("context_index", kind="stable")
        if ordered_centroids["embedding"].isna().any():
            raise ValueError("One or more target contexts lack an RNA centroid")
        context_embeddings = torch.as_tensor(
            np.stack(
                ordered_centroids["embedding"].map(
                    lambda value: np.asarray(value, np.float32)
                )
            ),
            device=device,
        )
    else:
        for parameter in model.rna_projection.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    epochs = int(config["model"]["epochs"])
    batch_size = int(config["model"]["batch_size"])
    batches_per_epoch = controls.batch_count(batch_size)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch: int | None = None
    start_epoch = 0
    epochs_without_improvement = 0
    if resume_from:
        assert checkpoint is not None
        if "optimizer_state" not in checkpoint:
            raise ValueError("Resume checkpoint lacks optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        (
            history,
            best_loss,
            best_epoch,
            start_epoch,
            epochs_without_improvement,
        ) = _load_resume_metadata(resume_from, output_dir, checkpoint, epochs)
    patience = int(config["model"].get("early_stopping_patience", epochs))
    minimum_delta = configured_minimum_delta
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    train_mask = pair_data.pairs["split"].eq("train").to_numpy()
    validation_mask = pair_data.pairs["split"].eq("validation").to_numpy()
    started = time.monotonic()
    progress_updates = int(config["model"].get("progress_updates_per_epoch", 200))
    report_every = max(1, batches_per_epoch // max(progress_updates, 1))
    epoch_bar = tqdm(
        range(start_epoch, epochs), desc=f"Train {stage}", unit="epoch"
    )
    for epoch in epoch_bar:
        model.train()
        latest_loss = float("nan")
        event_numerator = 0.0
        event_denominator = 0.0
        rank_numerator = 0.0
        rank_denominator = 0.0
        regularization_values: list[float] = []
        iterator = controls.iter_epoch(
            batch_size,
            shuffle_seed=int(config["seed"]) + epoch,
        )
        batch_bar = tqdm(
            iterator,
            total=batches_per_epoch,
            desc=f"{stage} epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
            mininterval=5.0,
        )
        for batch_index, batch in enumerate(batch_bar, start=1):
            positive_ids = batch.positive_ids
            event_control_ids = batch.event_control_ids
            rank_control_ids = batch.rank_control_ids
            if not train_mask[positive_ids].all():
                raise RuntimeError("Held-out positive pair reached training")
            if not train_mask[event_control_ids].all():
                raise RuntimeError("Held-out event control pair reached training")
            if not train_mask[rank_control_ids].all():
                raise RuntimeError("Held-out rank control pair reached training")
            loss, components = _batch_objective(
                model, pair_data, batch, device, context_embeddings, config, stage
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise FloatingPointError("Model gradient norm is non-finite") from error
            optimizer.step()
            loss_value = float(loss.item())
            latest_loss = loss_value
            event_weight = float(np.sum(batch.event_counts, dtype=np.float64))
            rank_weight = float(np.sum(batch.support_weights, dtype=np.float64))
            event_numerator += float(components["event"].item()) * event_weight
            event_denominator += event_weight
            rank_numerator += float(components["rank"].item()) * rank_weight
            rank_denominator += rank_weight
            regularization_values.append(float(components["regularization"].item()))
            batch_bar.set_postfix(loss=f"{loss_value:.5f}", refresh=False)
            if batch_index % report_every == 0 or batch_index == batches_per_epoch:
                elapsed = time.monotonic() - started
                completed_batches = epoch * batches_per_epoch + batch_index
                total_batches = epochs * batches_per_epoch
                rate = completed_batches / max(elapsed, 1e-9)
                atomic_json(
                    progress_path,
                    {
                        "status": "running",
                        "stage": stage,
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "batch": batch_index,
                        "batches_per_epoch": batches_per_epoch,
                        "completed_batches": completed_batches,
                        "total_batches": total_batches,
                        "completed_fraction": completed_batches / total_batches,
                        "latest_loss": loss_value,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": (
                            (total_batches - completed_batches) / rate
                            if rate > 0 else None
                        ),
                    },
                )
        train_event_loss = event_numerator / event_denominator
        train_rank_loss = (
            rank_numerator / rank_denominator if rank_denominator > 0 else 0.0
        )
        train_regularization = float(np.mean(regularization_values))
        epoch_loss = (
            train_event_loss
            + float(config["model"]["lambda_rank"]) * train_rank_loss
            + train_regularization
        )
        validation = _evaluate_control_objective(
            model,
            pair_data,
            validation_controls,
            expected_mask=validation_mask,
            device=device,
            context_embeddings=context_embeddings,
            config=config,
            stage=stage,
            batch_size=batch_size,
        )
        validation_loss = validation["loss"]
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "train_loss": epoch_loss,
                "event_loss": train_event_loss,
                "rank_loss": train_rank_loss,
                "regularization_loss": train_regularization,
                "validation_loss": validation_loss,
                "validation_event_loss": validation["event_loss"],
                "validation_rank_loss": validation["rank_loss"],
                "validation_regularization_loss": validation["regularization_loss"],
            }
        )
        checkpoint = {
            "schema_version": 1,
            "stage": stage,
            "model_config": model.config_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if _meaningful_improvement(validation_loss, best_loss, minimum_delta):
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        atomic_json(
            output_dir / "metrics.json",
            {
                "history": history,
                "best_loss": best_loss,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "checkpoint_selection": "held_out_sampled_validation_objective",
                "early_stopping_min_delta": minimum_delta,
            },
        )
        epoch_bar.set_postfix(
            train=f"{epoch_loss:.5f}",
            validation=f"{validation_loss:.5f}",
            best=f"{best_loss:.5f}",
        )
        if epochs_without_improvement >= patience:
            tqdm.write(
                f"Early stop after {len(history)} epochs: no validation-loss "
                f"improvement for {patience} epochs"
            )
            break
    elapsed = time.monotonic() - started
    completed_epochs = len(history)
    atomic_json(
        progress_path,
        {
            "status": "complete",
            "stage": stage,
            "epoch": completed_epochs,
            "epochs": epochs,
            "batch": batches_per_epoch,
            "batches_per_epoch": batches_per_epoch,
            "completed_batches": completed_epochs * batches_per_epoch,
            "total_batches": epochs * batches_per_epoch,
            "completed_fraction": 1.0,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": 0.0,
            "best_loss": best_loss,
            "best_validation_loss": best_loss,
        },
    )
    return {
        "best_loss": best_loss,
        "best_validation_loss": best_loss,
        "epochs": completed_epochs,
        "requested_epochs": epochs,
        "early_stopped": completed_epochs < epochs,
        "checkpoint": str(output_dir / "best.pt"),
        "elapsed_seconds": elapsed,
    }
