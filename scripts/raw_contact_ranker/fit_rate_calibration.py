#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import default_config
from raw_contact_ranker.calibration import fit_rate_calibration
from raw_contact_ranker.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"[rate10k] Fitting train-only NB2 calibration and validating it: "
        f"checkpoint={args.checkpoint.resolve()}",
        flush=True,
    )
    report = fit_rate_calibration(
        load_config(args.config),
        train_predictions=args.train_predictions.resolve(),
        validation_predictions=args.validation_predictions.resolve(),
        checkpoint=args.checkpoint.resolve(),
        output=args.output.resolve(),
    )
    passed = sum(report["checks"].values())
    print(
        f"[rate10k] Rate calibration accepted: {report['accepted']} "
        f"({passed}/{len(report['checks'])} shared checks passed); "
        f"report={args.output.resolve()}",
        flush=True,
    )
    if not report["accepted"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
