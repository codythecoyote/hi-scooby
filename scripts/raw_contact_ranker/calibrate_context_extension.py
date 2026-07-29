#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.calibration import calibrate_context_extension
from raw_contact_ranker.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--shared-calibration", type=Path, required=True)
    parser.add_argument("--context-gate", type=Path, required=True)
    parser.add_argument("--shared-train-predictions", type=Path, required=True)
    parser.add_argument(
        "--shared-validation-predictions", type=Path, required=True
    )
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        "[rate10k] Recalibrating validation-promoted context ranks with "
        "frozen shared-band dispersions",
        flush=True,
    )
    calibration, gate = calibrate_context_extension(
        load_config(args.config),
        shared_calibration_path=args.shared_calibration.resolve(),
        context_gate_path=args.context_gate.resolve(),
        shared_train_predictions=args.shared_train_predictions.resolve(),
        shared_validation_predictions=(
            args.shared_validation_predictions.resolve()
        ),
        calibration_output=args.calibration_output.resolve(),
        gate_output=args.gate_output.resolve(),
    )
    print(
        json.dumps(
            {
                "shared_accepted": calibration["accepted"],
                "context_outputs_accepted": sum(
                    row["accepted"] for row in gate["outputs"].values()
                ),
                "calibration_output": str(
                    args.calibration_output.resolve()
                ),
                "gate_output": str(args.gate_output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
