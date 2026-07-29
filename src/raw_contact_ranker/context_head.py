from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import zarr

from .common import atomic_json, selected_zarr_row, sha256_file
from .context import output_passes_power
from .data import PairData
from .exact_rate import ExactGroupDataset, build_data_contract
from .model import RawContactRanker


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = np.empty(count, np.float64)
    for rank, key in enumerate(ordered, start=1):
        value = float(p_values[key])
        if not 0 <= value <= 1:
            raise ValueError("P values must be in [0, 1]")
        adjusted[rank - 1] = value * count / rank
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    return {
        key: float(adjusted[index]) for index, key in enumerate(ordered)
    }


def _eligible_units(
    config: dict[str, Any],
    power_gate: Path,
    concordance_gate: Path,
) -> list[dict[str, Any]]:
    power = _read(power_gate)
    concordance = _read(concordance_gate)
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    candidates: list[dict[str, Any]] = [
        {
            "output_id": str(name),
            "output_type": "context",
            "members": [str(name)],
        }
        for name in contexts["cell_type"].astype(str)
    ]
    candidates.extend(
        {
            "output_id": str(pool["id"]),
            "output_type": "pool",
            "members": list(map(str, pool["members"])),
        }
        for pool in config.get("contexts", {}).get("pools", [])
    )
    return [
        row
        for row in candidates
        if output_passes_power(config, power, row["output_id"])
        and concordance["outputs"].get(row["output_id"], {}).get("passed") is True
    ]


def _unit_counts(
    config: dict[str, Any],
    units: list[dict[str, Any]],
    pair_ids: np.ndarray,
) -> np.ndarray:
    data_root = Path(config["outputs"]["data_root"])
    evidence = zarr.open_group(
        str(data_root / "pseudoreplicate_evidence.zarr"), mode="r"
    )
    contexts = pd.read_parquet(config["paths"]["contexts"]).sort_values(
        "context_index", kind="stable"
    )
    index = {
        str(name): row
        for row, name in enumerate(contexts["cell_type"].astype(str))
    }
    pair_count = int(evidence["full_count"].shape[1])
    ids = np.asarray(pair_ids, np.int64)
    context_counts = {}
    for name, row in tqdm(
        index.items(),
        total=len(index),
        desc="Load context-head counts",
        unit="context",
    ):
        context_counts[name] = selected_zarr_row(
            evidence["full_count"],
            row,
            ids,
            pair_count=pair_count,
            dtype=np.uint64,
        )
    return np.stack(
        [
            np.sum(
                np.stack(
                    [
                        context_counts[name]
                        for name in unit["members"]
                    ]
                ),
                axis=0,
                dtype=np.uint64,
            )
            for unit in units
        ],
        axis=1,
    )


def _rna_embeddings(
    config: dict[str, Any],
    units: list[dict[str, Any]],
) -> np.ndarray:
    centroids = pd.read_parquet(config["paths"]["centroids"]).set_index(
        "cell_type"
    )
    values = []
    for unit in units:
        members = [
            np.asarray(centroids.loc[name, "embedding"], np.float32)
            for name in unit["members"]
        ]
        values.append(np.mean(np.stack(members), axis=0))
    output = np.stack(values).astype(np.float32)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise ValueError("Invalid RNA centroid matrix")
    return output


def _derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("RNA permutation requires at least two outputs")
    rng = np.random.default_rng(seed)
    base = np.arange(size)
    for _ in range(10_000):
        candidate = rng.permutation(size)
        if np.all(candidate != base):
            return candidate
    raise RuntimeError("Could not construct deterministic RNA derangement")


def _context_embeddings(
    config: dict[str, Any],
    units: list[dict[str, Any]],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "onehot":
        return np.eye(len(units), dtype=np.float32), {
            "kind": "onehot",
            "source": None,
        }
    rna = _rna_embeddings(config, units)
    metadata: dict[str, Any] = {
        "kind": mode,
        "source": str(config["paths"]["centroids"]),
        "source_sha256": sha256_file(Path(config["paths"]["centroids"])),
    }
    if mode == "rna_permuted":
        permutation = _derangement(len(units), int(config["seed"]) + 71)
        rna = rna[permutation]
        metadata["permutation"] = permutation.tolist()
    elif mode != "rna":
        raise ValueError(f"Unsupported context-head mode: {mode}")
    return rna, metadata


def exact_context_log_likelihood(
    log_exposure: torch.Tensor,
    learned_score: torch.Tensor,
    counts: torch.Tensor,
    group_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if learned_score.ndim != 2:
        raise ValueError("Context score must have shape [pairs, outputs]")
    if counts.shape != learned_score.shape:
        raise ValueError("Context counts and scores do not align")
    if log_exposure.shape != learned_score.shape[:1]:
        raise ValueError("Exposure does not align to context scores")
    centered = torch.empty_like(learned_score)
    likelihood = torch.zeros(
        learned_score.shape[1],
        dtype=learned_score.dtype,
        device=learned_score.device,
    )
    events = counts.sum(dim=0)
    for group in torch.unique(group_index):
        selected = group_index == group
        centered[selected] = (
            learned_score[selected]
            - learned_score[selected].mean(dim=0, keepdim=True)
        )
        logits = log_exposure[selected, None] + centered[selected]
        likelihood = likelihood + torch.sum(
            counts[selected] * torch.log_softmax(logits, dim=0),
            dim=0,
        )
    return likelihood, events, centered


def _context_model(
    shared_checkpoint: dict[str, Any],
    *,
    context_dim: int,
) -> RawContactRanker:
    model_config = dict(shared_checkpoint["model_config"])
    model_config["context_dim"] = context_dim
    model = RawContactRanker(**model_config)
    state = dict(shared_checkpoint["model_state"])
    state.pop("rna_projection.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"rna_projection.weight"} or unexpected:
        raise RuntimeError(
            f"Could not initialize context head: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.freeze_sequence()
    for parameter in model.sequence_score.parameters():
        parameter.requires_grad_(False)
    for parameter in model.technical.parameters():
        parameter.requires_grad_(False)
    return model


def _batch_scores(
    model: RawContactRanker,
    pair_data: PairData,
    pair_ids: np.ndarray,
    embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = pair_data.model_inputs(pair_ids, device)
    output = model(**inputs, context_embedding=embeddings)
    shared = output["residual_score"] + output["technical_offset"]
    learned = shared[:, None] + output["context_delta"]
    return learned, inputs["fixed_exposure"], output["context_delta"]


def _predict_context(
    model: RawContactRanker,
    pair_data: PairData,
    dataset: ExactGroupDataset,
    embeddings: torch.Tensor,
    device: torch.device,
    maximum_pairs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    total_ll = np.zeros(embeddings.shape[0], np.float64)
    total_events = np.zeros(embeddings.shape[0], np.float64)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            dataset.iter_batches(maximum_pairs, shuffle_seed=None),
            desc="Predict context residuals",
            unit="batch",
        ):
            learned, exposure, delta = _batch_scores(
                model, pair_data, batch.pair_ids, embeddings, device
            )
            counts = torch.zeros_like(learned)
            groups = torch.as_tensor(batch.group_index, device=device)
            _, _, centered = exact_context_log_likelihood(
                exposure, learned, counts, groups
            )
            shared_centered = torch.empty_like(exposure)
            for group in torch.unique(groups):
                selected = groups == group
                shared_centered[selected] = (
                    learned[selected, 0] - delta[selected, 0]
                ) - (learned[selected, 0] - delta[selected, 0]).mean()
            centered_delta = centered - shared_centered[:, None]
            frame = pd.DataFrame({"pair_id": batch.pair_ids})
            for index in range(embeddings.shape[0]):
                frame[f"context_delta_{index:03d}"] = (
                    centered_delta[:, index].cpu().numpy().astype(np.float32)
                )
            frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True).sort_values(
            "pair_id", kind="stable"
        ),
        {
            "conditional_log_likelihood": total_ll.tolist(),
            "events": total_events.tolist(),
        },
    )


def train_context_head(
    config: dict[str, Any],
    *,
    shared_checkpoint_path: Path,
    power_gate: Path,
    concordance_gate: Path,
    mode: str,
    output_dir: Path,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    units = _eligible_units(config, power_gate, concordance_gate)
    if len(units) < 2:
        raise RuntimeError("Fewer than two outputs passed context prerequisites")
    embeddings_array, embedding_metadata = _context_embeddings(
        config, units, mode
    )
    data_root = Path(config["outputs"]["data_root"])
    pair_data = PairData(
        data_root / "canonical_pairs.parquet",
        data_root / "pair_features.zarr",
        preload_features=bool(config["model"].get("preload_features", True)),
    )
    training_ids = pair_data.pairs.loc[
        pair_data.pairs["split"].eq("train"), "pair_id"
    ].to_numpy(np.int64)
    counts = _unit_counts(config, units, training_ids)
    shared_checkpoint = torch.load(
        shared_checkpoint_path, map_location="cpu", weights_only=False
    )
    shared_contract = build_data_contract(
        config,
        pair_count=len(pair_data.pairs),
        objective="exact_local_rate",
        feature_set=str(shared_checkpoint["data_contract"]["feature_set"]),
    )
    if shared_checkpoint.get("data_contract") != shared_contract:
        raise RuntimeError("Shared checkpoint data contract is stale")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _context_model(
        shared_checkpoint, context_dim=embeddings_array.shape[1]
    ).to(device)
    embeddings = torch.as_tensor(embeddings_array, device=device)
    train_dataset = ExactGroupDataset(
        pair_data.pairs,
        np.zeros(len(pair_data.pairs), np.uint8),
        split="train",
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    maximum_pairs = int(config["model"]["batch_size"])
    history = []
    for epoch in range(epochs):
        model.train()
        model.sequence.eval()
        ll_total = 0.0
        event_total = 0.0
        for batch in tqdm(
            train_dataset.iter_batches(
                maximum_pairs, shuffle_seed=seed + epoch
            ),
            desc=f"{mode} context epoch {epoch}",
            unit="batch",
        ):
            learned, exposure, _ = _batch_scores(
                model, pair_data, batch.pair_ids, embeddings, device
            )
            batch_counts = torch.as_tensor(
                counts[batch.pair_ids], dtype=torch.float32, device=device
            )
            groups = torch.as_tensor(batch.group_index, device=device)
            likelihood, events, _ = exact_context_log_likelihood(
                exposure, learned, batch_counts, groups
            )
            event_sum = events.sum()
            if not bool((event_sum > 0).item()):
                continue
            loss = -likelihood.sum() / event_sum
            loss = loss + float(config["model"].get("rna_l2", 0.0)) * (
                model.rna_regularization() / max(len(units), 1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            ll_total += float(likelihood.detach().sum().item())
            event_total += float(event_sum.item())
        history.append(
            {
                "epoch": epoch,
                "negative_log_likelihood_per_event": (
                    -ll_total / event_total if event_total else None
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        **shared_contract,
        "objective": "context_exact_local_rate",
        "shared_checkpoint_sha256": sha256_file(shared_checkpoint_path),
        "mode": mode,
        "output_ids": [row["output_id"] for row in units],
        "embedding_metadata": embedding_metadata,
    }
    checkpoint = {
        "schema_version": 2,
        "stage": "context_rate",
        "model_config": model.config_dict(),
        "model_state": model.state_dict(),
        "data_contract": contract,
        "units": units,
        "embeddings": embeddings_array,
        "seed": seed,
        "epochs": epochs,
    }
    checkpoint_path = output_dir / "context_head.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    prediction_paths = {}
    for split in tqdm(
        ("train", "validation"),
        desc="Write context predictions",
        unit="split",
    ):
        dataset = ExactGroupDataset(
            pair_data.pairs,
            np.zeros(len(pair_data.pairs), np.uint8),
            split=split,
        )
        predictions, _ = _predict_context(
            model,
            pair_data,
            dataset,
            embeddings,
            device,
            maximum_pairs,
        )
        path = output_dir / f"{split}.context_predictions.parquet"
        predictions.to_parquet(path, index=False, compression="zstd")
        prediction_paths[split] = str(path)
    report = {
        "schema_version": 1,
        "mode": mode,
        "units": units,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_paths": prediction_paths,
        "data_contract": contract,
        "test_accessed": False,
    }
    atomic_json(output_dir / "context_training.json", report)
    return report


def predict_context_test(
    config: dict[str, Any],
    *,
    checkpoint_path: Path,
    output: Path,
    freeze_test: bool,
    frozen_release: Path,
    test_lock: Path,
) -> dict[str, Any]:
    if not freeze_test:
        raise ValueError("Context test prediction requires freeze_test=True")
    from .release import verify_test_authorization

    verify_test_authorization(
        frozen_release,
        test_lock,
        checkpoint_sha256=sha256_file(checkpoint_path),
        allow_context_checkpoint=True,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    data_root = Path(config["outputs"]["data_root"])
    pair_data = PairData(
        data_root / "canonical_pairs.parquet",
        data_root / "pair_features.zarr",
        preload_features=bool(config["model"].get("preload_features", True)),
    )
    expected = {
        **build_data_contract(
            config,
            pair_count=len(pair_data.pairs),
            objective="exact_local_rate",
            feature_set=str(checkpoint["data_contract"]["feature_set"]),
        ),
        "objective": "context_exact_local_rate",
        "shared_checkpoint_sha256": checkpoint["data_contract"][
            "shared_checkpoint_sha256"
        ],
        "mode": checkpoint["data_contract"]["mode"],
        "output_ids": checkpoint["data_contract"]["output_ids"],
        "embedding_metadata": checkpoint["data_contract"][
            "embedding_metadata"
        ],
    }
    if checkpoint["data_contract"] != expected:
        raise RuntimeError("Context checkpoint data contract is stale")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawContactRanker(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    embeddings = torch.as_tensor(
        checkpoint["embeddings"], dtype=torch.float32, device=device
    )
    dataset = ExactGroupDataset(
        pair_data.pairs,
        np.zeros(len(pair_data.pairs), np.uint8),
        split="test",
    )
    predictions, _ = _predict_context(
        model,
        pair_data,
        dataset,
        embeddings,
        device,
        int(config["model"]["batch_size"]),
    )
    predictions.to_parquet(output, index=False, compression="zstd")
    report = {
        "schema_version": 1,
        "split": "test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(output),
        "prediction_sha256": sha256_file(output),
        "test_accessed": True,
        "frozen_release_sha256": sha256_file(frozen_release),
    }
    atomic_json(output.with_suffix(".json"), report)
    return report


def merge_context_release_predictions(
    config: dict[str, Any],
    *,
    context_gate_path: Path,
    test_predictions: dict[str, Path],
    frozen_release: Path,
    test_lock: Path,
    output: Path,
) -> dict[str, Any]:
    """Merge frozen train/validation heads with their one-shot test scores."""
    from .release import verify_test_lock

    verify_test_lock(frozen_release, test_lock)
    gate = _read(context_gate_path)
    accepted = {
        output_id: row
        for output_id, row in gate["outputs"].items()
        if row.get("accepted") is True
    }
    if not accepted:
        raise RuntimeError("No accepted context outputs require release predictions")
    mode_test = {
        mode: pd.read_parquet(path).sort_values("pair_id", kind="stable")
        for mode, path in test_predictions.items()
    }
    data_root = Path(config["outputs"]["data_root"])
    pairs = pd.read_parquet(
        data_root / "canonical_pairs.parquet", columns=["pair_id"]
    ).sort_values("pair_id", kind="stable")
    release = pd.DataFrame(
        {"pair_id": pairs["pair_id"].to_numpy(np.int64)}
    )
    for output_id, row in tqdm(
        accepted.items(),
        total=len(accepted),
        desc="Merge accepted context predictions",
        unit="output",
    ):
        mode = str(row["selected_mode"])
        column = str(row["prediction_column"])
        pieces = [
            pd.read_parquet(
                row["training_prediction_path"],
                columns=["pair_id", column],
            ),
            pd.read_parquet(
                row["validation_prediction_path"],
                columns=["pair_id", column],
            ),
            mode_test[mode][["pair_id", column]],
        ]
        values = pd.concat(pieces, ignore_index=True).sort_values(
            "pair_id", kind="stable"
        )
        if not np.array_equal(
            values["pair_id"].to_numpy(np.int64),
            release["pair_id"].to_numpy(np.int64),
        ):
            raise ValueError(
                f"Context release predictions do not cover every pair: {output_id}"
            )
        release[column] = values[column].to_numpy(np.float32)
    expected_output = {
        str(row["prediction_path"]) for row in accepted.values()
    }
    if expected_output != {str(output)}:
        raise RuntimeError(
            "Frozen context gate authorizes a different release prediction path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    release.to_parquet(output, index=False, compression="zstd")
    report = {
        "schema_version": 1,
        "context_gate": str(context_gate_path),
        "context_gate_sha256": sha256_file(context_gate_path),
        "test_predictions": {
            mode: {"path": str(path), "sha256": sha256_file(path)}
            for mode, path in test_predictions.items()
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
        "accepted_outputs": sorted(accepted),
        "test_accessed": True,
    }
    atomic_json(output.with_suffix(".json"), report)
    return report
