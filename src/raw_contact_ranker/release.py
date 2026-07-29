from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .common import atomic_json, sha256_file


def _read(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def freeze_release(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    validation_evaluation: Path,
    power_gate: Path,
    topology_gate: Path,
    calibration_gate: Path,
    selection: Path,
    rollout: Path,
    context_gate: Path | None,
    output: Path,
) -> dict[str, Any]:
    """Freeze every tunable input before the only permitted test access."""
    if output.exists():
        raise FileExistsError(f"Frozen release already exists: {output}")
    power = _read(power_gate)
    topology = _read(topology_gate)
    calibration = _read(calibration_gate)
    selected = _read(selection)
    evaluation = _read(validation_evaluation)
    checks = {
        "power_eligible": power.get("eligible") is True,
        "topology_promoted": topology.get("promoted") is True,
        "calibration_accepted": calibration.get("accepted") is True,
        "feature_selected": selected.get("topology_training_authorized") is True,
        "validation_only": all(
            artifact.get("test_accessed") is False
            for artifact in (topology, calibration, selected, evaluation)
        ),
        "checkpoint_matches_evaluation": (
            evaluation.get("checkpoint_sha256") == sha256_file(checkpoint)
        ),
        "checkpoint_matches_calibration": (
            calibration.get("checkpoint_sha256") == sha256_file(checkpoint)
        ),
        "validation_predictions_match_topology": (
            topology.get("prediction_sha256")
            == sha256_file(Path(evaluation["prediction_path"]))
        ),
        "validation_predictions_match_calibration": (
            calibration.get("validation_predictions_sha256")
            == sha256_file(Path(evaluation["prediction_path"]))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Release freeze refused failed validation gates: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    data_root = Path(config["outputs"]["data_root"])
    inputs = {
        "config": Path(config["_config_path"]),
        "preparation_contract": data_root / "preparation_contract.json",
        "feature_manifest": data_root / "feature_manifest.json",
        "checkpoint": checkpoint,
        "validation_evaluation": validation_evaluation,
        "power_gate": power_gate,
        "topology_gate": topology_gate,
        "calibration_gate": calibration_gate,
        "selection": selection,
        "rollout": rollout,
    }
    if context_gate is not None:
        context = _read(context_gate)
        if context.get("test_accessed") is not False:
            raise RuntimeError("Context extension gate touched test before freeze")
        inputs["context_gate"] = context_gate
    report = {
        "schema_version": 1,
        "claim": (
            "scHiCAR_expected_long_range_contact_rate_at_10kb_per_million_assay_pairs"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "checks": checks,
        "frozen": True,
        "test_accessed": False,
    }
    atomic_json(output, report)
    return report


def acquire_test_lock(frozen_release: Path, lock_path: Path) -> dict[str, Any]:
    """Irreversibly record the one allowed test access before reading test."""
    frozen = _read(frozen_release)
    if not frozen.get("frozen") or frozen.get("test_accessed"):
        raise RuntimeError("Release is not eligible for one-shot test access")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "The one-shot test lock already exists; retesting is forbidden"
        ) from error
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "frozen_release": str(frozen_release),
                "frozen_release_sha256": sha256_file(frozen_release),
                "test_access_authorized": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return _read(lock_path)


def verify_test_authorization(
    frozen_release: Path,
    test_lock: Path,
    *,
    checkpoint_sha256: str,
    allow_context_checkpoint: bool,
) -> dict[str, Any]:
    """Bind a test read to the immutable release, lock, and checkpoint."""
    frozen = verify_test_lock(frozen_release, test_lock)
    allowed = {frozen["inputs"]["checkpoint"]["sha256"]}
    if allow_context_checkpoint and "context_gate" in frozen["inputs"]:
        context_record = frozen["inputs"]["context_gate"]
        context_path = Path(context_record["path"])
        if sha256_file(context_path) != context_record["sha256"]:
            raise RuntimeError("Frozen context gate changed before test")
        context = _read(context_path)
        allowed.update(
            str(row["checkpoint_sha256"])
            for row in context.get("mode_reports", {}).values()
        )
    if checkpoint_sha256 not in allowed:
        raise RuntimeError("Checkpoint is not authorized by the frozen release")
    return frozen


def verify_test_lock(
    frozen_release: Path,
    test_lock: Path,
) -> dict[str, Any]:
    """Verify that the irreversible test lock authorizes this release."""
    frozen = _read(frozen_release)
    lock = _read(test_lock)
    if not frozen.get("frozen") or frozen.get("test_accessed"):
        raise RuntimeError("Invalid frozen release for test access")
    if lock.get("frozen_release_sha256") != sha256_file(frozen_release):
        raise RuntimeError("Test lock does not match the frozen release")
    return frozen


def finalize_test_gate(
    *,
    frozen_release: Path,
    test_lock: Path,
    exact_evaluation: Path,
    topology_gate: Path,
    calibration_gate: Path,
    context_test_gate: Path | None,
    context_prediction_report: Path | None,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Final test gate already exists: {output}")
    frozen = _read(frozen_release)
    lock = _read(test_lock)
    exact = _read(exact_evaluation)
    topology = _read(topology_gate)
    calibration = _read(calibration_gate)
    checkpoint_hash = frozen["inputs"]["checkpoint"]["sha256"]
    checks = {
        "release_hash_matches_lock": (
            lock.get("frozen_release_sha256") == sha256_file(frozen_release)
        ),
        "exact_test_only": exact.get("split") == "test"
        and exact.get("test_accessed") is True,
        "checkpoint_unchanged": exact.get("checkpoint_sha256") == checkpoint_hash,
        "exact_release_unchanged": (
            exact.get("frozen_release_sha256")
            == sha256_file(frozen_release)
        ),
        "topology_test_only": topology.get("split") == "test"
        and topology.get("test_accessed") is True,
        "topology_promoted": topology.get("promoted") is True,
        "calibration_test_only": calibration.get("split") == "test"
        and calibration.get("test_accessed") is True,
        "calibration_accepted": calibration.get("accepted") is True,
        "calibration_checkpoint_unchanged": (
            calibration.get("validation_checkpoint_sha256") == checkpoint_hash
        ),
        "calibration_rollout_unchanged": (
            calibration.get("rollout_sha256")
            == frozen["inputs"]["rollout"]["sha256"]
        ),
        "calibration_release_unchanged": (
            calibration.get("frozen_release_sha256")
            == sha256_file(frozen_release)
        ),
        "test_predictions_match_topology": (
            topology.get("prediction_sha256")
            == sha256_file(Path(exact["prediction_path"]))
        ),
        "exact_prediction_hash_recorded": (
            exact.get("prediction_sha256")
            == sha256_file(Path(exact["prediction_path"]))
        ),
        "test_predictions_match_calibration": (
            calibration.get("prediction_sha256")
            == sha256_file(Path(exact["prediction_path"]))
        ),
    }
    artifacts = {
        "frozen_release": frozen_release,
        "test_lock": test_lock,
        "exact_evaluation": exact_evaluation,
        "topology_gate": topology_gate,
        "calibration_gate": calibration_gate,
    }
    frozen_context = frozen["inputs"].get("context_gate")
    if frozen_context is not None:
        if context_test_gate is None or context_prediction_report is None:
            raise RuntimeError(
                "Frozen context extension requires final context test artifacts"
            )
        context = _read(context_test_gate)
        context_predictions = _read(context_prediction_report)
        checks.update(
            {
                "context_test_only": context.get("test_accessed") is True,
                "context_test_accepted": context.get("accepted") is True,
                "context_validation_gate_unchanged": (
                    context.get("validation_context_gate_sha256")
                    == frozen_context["sha256"]
                ),
                "context_prediction_gate_unchanged": (
                    context_predictions.get("context_gate_sha256")
                    == frozen_context["sha256"]
                ),
                "context_predictions_include_test": (
                    context_predictions.get("test_accessed") is True
                ),
                "context_predictions_unchanged_for_calibration": (
                    calibration.get("context_delta_sha256", {}).get(
                        context_predictions.get("output")
                    )
                    == context_predictions.get("output_sha256")
                ),
            }
        )
        artifacts["context_test_gate"] = context_test_gate
        artifacts["context_prediction_report"] = context_prediction_report
    elif context_test_gate is not None or context_prediction_report is not None:
        raise RuntimeError("Unexpected context test artifacts for shared-only freeze")
    report = {
        "schema_version": 1,
        "claim": frozen["claim"],
        "checks": checks,
        "accepted": bool(checks) and all(checks.values()),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "test_accessed": True,
        "retuning_permitted": False,
    }
    atomic_json(output, report)
    return report
